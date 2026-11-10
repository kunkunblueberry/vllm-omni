# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Physical per-layer slot allocation for Noisy PP cell versions."""

from __future__ import annotations

import torch

from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.noisy_pp_plan import Cell
from vllm_omni.experimental.ar_diffusion.kv_cache.paged import allocate_kv_pool_with_views

VersionKey = tuple[str, int, int, int]  # req, chunk, step, block


class VersionPool:
    """Per-layer version slots. ``capacity`` is slots per layer (independent free stacks)."""

    def __init__(
        self,
        *,
        capacity: int,
        chunk_blocks: int,
        block_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_size: int,
        dtype: torch.dtype,
        device: torch.device,
        layer_offset: int = 0,
    ) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be positive, got {capacity}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        self.capacity = capacity
        self.chunk_blocks = chunk_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.layer_offset = layer_offset
        self.device = torch.device(device)
        kv_pools, k_pools, v_pools = allocate_kv_pool_with_views(
            num_blocks=capacity * chunk_blocks,
            block_size=block_size,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_size,
            dtype=dtype,
            device=device,
        )
        self.kv_pools = kv_pools
        self.k_pools = k_pools
        self.v_pools = v_pools
        self.free: list[list[int]] = [list(range(capacity - 1, -1, -1)) for _ in range(num_layers)]
        self.keys: dict[VersionKey, int] = {}

    def alloc(self, key: VersionKey) -> int:
        if key in self.keys:
            return self.keys[key]
        layer = self.local_layer(key)
        free = self.free[layer]
        if not free:
            raise RuntimeError(
                f"VersionPool exhausted on layer={layer} (per-layer capacity={self.capacity}, "
                f"num_layers={self.num_layers})"
            )
        slot = free.pop()
        self.keys[key] = slot
        return slot

    def release(self, key: VersionKey) -> None:
        slot = self.keys.pop(key, None)
        if slot is not None:
            self.free[self.local_layer(key)].append(slot)

    def slot_of(self, key: VersionKey) -> int:
        return self.keys[key]

    def has(self, key: VersionKey) -> bool:
        return key in self.keys

    @property
    def has_free(self) -> bool:
        return any(bool(stack) for stack in self.free)

    def block_ids(self, slot: int) -> list[int]:
        start = slot * self.chunk_blocks
        return list(range(start, start + self.chunk_blocks))

    def local_layer(self, key: VersionKey) -> int:
        return _layer_of(key, num_layers=self.num_layers, layer_offset=self.layer_offset)

    def kv_block_views(self, key: VersionKey, num_blocks: int) -> tuple[torch.Tensor, torch.Tensor]:
        """K/V views ``[num_blocks, block_size, H, D]`` for one version."""
        slot = self.slot_of(key)
        layer = self.local_layer(key)
        start = slot * self.chunk_blocks
        end = start + num_blocks
        return self.kv_pools[layer][0][start:end], self.kv_pools[layer][1][start:end]

    def tensor_dict(self, key: VersionKey, num_blocks: int) -> dict[str, torch.Tensor]:
        k, v = self.kv_block_views(key, num_blocks)
        return {
            "k": k.contiguous(),
            "v": v.contiguous(),
            "layer": torch.tensor([self.local_layer(key)], dtype=torch.int32),
        }

    def copy_into(self, key: VersionKey, payload: dict[str, torch.Tensor]) -> None:
        slot = self.slot_of(key)
        layer = int(payload["layer"].reshape(-1)[0].item()) if "layer" in payload else self.local_layer(key)
        start = slot * self.chunk_blocks
        end = start + payload["k"].shape[0]
        self.kv_pools[layer][0][start:end].copy_(payload["k"])
        self.kv_pools[layer][1][start:end].copy_(payload["v"])


def _version_key(req: str, version: Cell) -> VersionKey:
    return (req, version[0], version[1], version[2])


def _layer_of(key: VersionKey, *, num_layers: int, layer_offset: int = 0) -> int:
    """Absolute block → local pool layer; remote versions use ``block % num_layers``."""
    block = key[3]
    local = block - layer_offset
    if not 0 <= local < num_layers:
        local = block % num_layers
    return local

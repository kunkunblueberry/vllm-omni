# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Noisy PP cell-version cache allocation and session-structure ownership."""

from __future__ import annotations

import torch

from vllm_omni.experimental.ar_diffusion.kv_cache.noisy_pool import VersionPool
from vllm_omni.experimental.ar_diffusion.kv_cache.noisy_spec import ARDiffusionNoisyKVSpec
from vllm_omni.experimental.ar_diffusion.kv_cache.noisy_transport import NoisyKVTransport


class NoisyKVCache:
    def __init__(
        self,
        spec: ARDiffusionNoisyKVSpec,
        *,
        dtype: torch.dtype,
        device: torch.device,
        layer_groups: int,
        max_batch_size: int,
        step_replicas: int = 1,
        gpu_memory_fraction: float = 1.0,
        available_bytes: int | None = None,
        layer_offset: int = 0,
        blocks_per_rank: int = 1,
    ) -> None:
        if not 0.0 < float(gpu_memory_fraction) <= 1.0:
            raise ValueError(f"gpu_memory_fraction must be in (0, 1], got {gpu_memory_fraction}")
        self.spec = spec
        self.layer_groups = layer_groups
        self.max_batch_size = max(1, max_batch_size)
        self.blocks_per_rank = max(1, blocks_per_rank)
        chunk_blocks = spec.max_chunk_tokens // spec.block_size
        # Capacity is per layer. Same-Step needs at most H+2 resident versions:
        # one fixed clean sink, up to H recent chunks, and the current write;
        # chunk 0 is deduplicated while inside the recent window. The G/step
        # margins below also cover pipeline overlap and transfers.
        per_req = max(spec.max_history_chunks + 2 + layer_groups, spec.max_history_chunks + step_replicas)
        # Retention shares this pool, and a Same-Step window holds one version per
        # denoise step rather than one per chunk, so the retained set can be wider
        # than the chunk count. Size for the widest window the spec allows.
        per_req = max(per_req, spec.history_versions + 2 + layer_groups)
        desired = self.max_batch_size * per_req
        bytes_per_version = (
            2
            * chunk_blocks
            * spec.block_size
            * spec.num_kv_heads
            * spec.head_size
            * torch.empty((), dtype=dtype).element_size()
        )
        bytes_all_layers = bytes_per_version * max(1, spec.num_layers)
        if available_bytes is not None:
            budget = int(available_bytes * float(gpu_memory_fraction))
            if bytes_all_layers <= 0:
                raise RuntimeError("invalid NoisyKV geometry: bytes_all_layers <= 0")
            max_by_budget = budget // bytes_all_layers
            if max_by_budget < 1:
                raise RuntimeError(
                    f"NoisyKV budget too small for one per-layer version slot: "
                    f"budget={budget} bytes_all_layers={bytes_all_layers}"
                )
            if max_by_budget < desired:
                raise RuntimeError(
                    f"NoisyKV budget fits {max_by_budget} versions but configured schedule requires {desired}: "
                    f"budget={budget} required_bytes={desired * bytes_all_layers}"
                )
        self.capacity = desired
        self.pool = VersionPool(
            capacity=self.capacity,
            chunk_blocks=chunk_blocks,
            block_size=spec.block_size,
            num_layers=spec.num_layers,
            num_kv_heads=spec.num_kv_heads,
            head_size=spec.head_size,
            dtype=dtype,
            device=device,
            layer_offset=layer_offset,
        )
        self.transport = NoisyKVTransport(self.pool)
        self.resident_peak = 0
        self.bytes_per_version = bytes_per_version
        self.reserved_bytes = self.capacity * bytes_all_layers
        # One session may hold at most the window this pool was sized for:
        # ``per_req`` versions on each of ``num_layers`` layers. A pool key counts
        # one block's layer, so the allowance is stated in the same units
        # :meth:`session_usage` reports -- admission can then charge a session
        # exactly what the physical pool would let it occupy, and no more.
        self.session_grant_versions = per_req * max(1, spec.num_layers)
        self.session_grant_bytes = per_req * bytes_all_layers
        # First-request schedule fingerprint per outer session_id / pool key.
        self._session_structural_keys: dict[str, tuple[object, ...]] = {}

    def bind_session_structure(self, kv_id: str, key: tuple[object, ...]) -> None:
        """Record or check the schedule fingerprint for a persistent session pool key."""
        existing = self._session_structural_keys.get(kv_id)
        if existing is None:
            self._session_structural_keys[kv_id] = key
            return
        if existing != key:
            raise ValueError(
                f"session {kv_id!r} schedule structural_key mismatch: "
                f"retained={existing!r} request={key!r}; "
                "release_noisy_kv and restart, or keep D/S/ordering/H/blocks fixed"
            )

    def clear_session_structure(self, kv_id: str) -> None:
        self._session_structural_keys.pop(kv_id, None)

    def session_usage(self, kv_id: str) -> tuple[int, int]:
        """(versions, bytes) the pool key ``kv_id`` currently holds.

        ``versions`` is the logical retained-version count and the second value the
        footprint of those versions. A pool key holds one block's layer, so each
        version costs ``bytes_per_version``. Both are what a capacity ledger needs
        to reconcile a session's grant against reality -- and what it charges at
        admission, since a session is granted exactly the window the pool sized for
        it. The physical layout stays private to this pool.
        """
        versions = sum(1 for key in self.pool.keys if key[0] == kv_id)
        return versions, versions * self.bytes_per_version

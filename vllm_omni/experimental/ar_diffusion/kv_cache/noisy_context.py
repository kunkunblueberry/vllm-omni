# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Paged-attention layer view constructed from one Noisy PP cell version."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention import (
    ARDiffusionPagedLayerInputs,
    _layer_idx_tensor,
)


@dataclass
class NoisyLayerContext:
    is_ar_diffusion_paged_context: ClassVar[bool] = True
    layer_idx: int
    key_pool: torch.Tensor
    value_pool: torch.Tensor
    block_size: int
    video_slots: torch.Tensor
    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    max_query_len: int
    max_seq_len: int
    seq_len: int
    # Cached paged-attn view (avoids realloc on the hot path).
    _inputs: ARDiffusionPagedLayerInputs | None = None
    _empty_action_slots: ClassVar[dict[tuple[str, int | None], torch.Tensor]] = {}

    def _action_slots(self) -> torch.Tensor:
        dev = self.video_slots.device
        key = (dev.type, dev.index)
        buf = NoisyLayerContext._empty_action_slots.get(key)
        if buf is None or buf.device != dev:
            buf = self.video_slots.new_empty(0, dtype=torch.long)
            NoisyLayerContext._empty_action_slots[key] = buf
        return buf

    def to_layer_inputs(self) -> ARDiffusionPagedLayerInputs:
        if self._inputs is not None:
            return self._inputs
        self._inputs = ARDiffusionPagedLayerInputs(
            layer_idx=_layer_idx_tensor(self.layer_idx),
            key_pool=self.key_pool,
            value_pool=self.value_pool,
            block_size=self.block_size,
            seq_len=self.seq_len,
            video_slots=self.video_slots,
            action_slots=self._action_slots(),
            block_table=self.block_table,
            query_start_loc=self.query_start_loc,
            seq_lens=self.seq_lens,
            max_query_len=self.max_query_len,
            max_seq_len=self.max_seq_len,
        )
        return self._inputs

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Configuration and geometry for Noisy PP's cell-version KV pool."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ARDiffusionNoisyKVSpec:
    num_layers: int
    num_kv_heads: int
    head_size: int
    block_size: int
    max_chunk_tokens: int
    max_history_chunks: int
    # Versions per chunk a session keeps across requests: one for the diagonal
    # orderings (the clean KV of each chunk), one per denoise step plus the
    # chunk-0 sink for Same-Step, whose consumer reads the version matching its
    # own step. Defaults to the diagonal case.
    max_history_versions: int | None = None

    @property
    def history_versions(self) -> int:
        """Retained versions per layer one rank may hold at once."""
        return int(self.max_history_versions or self.max_history_chunks)

    def __post_init__(self) -> None:
        if self.max_history_versions is not None and self.max_history_versions <= 0:
            raise ValueError(f"max_history_versions must be positive, got {self.max_history_versions}")
        for name in (
            "num_layers",
            "num_kv_heads",
            "head_size",
            "block_size",
            "max_chunk_tokens",
            "max_history_chunks",
        ):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.max_chunk_tokens % self.block_size != 0:
            raise ValueError("max_chunk_tokens must be a multiple of block_size")

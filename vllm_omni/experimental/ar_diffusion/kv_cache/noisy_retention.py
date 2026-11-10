# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Noisy PP's ordering-specific cross-request cell retention policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.noisy_pp_plan import Cell, CellSchedule, Ordering

if TYPE_CHECKING:
    from vllm_omni.experimental.ar_diffusion.kv_cache.noisy_manager import NoisyKVCache


def clean_history_cells(schedule: CellSchedule, *, num_blocks: int) -> set[Cell]:
    """History cells to retain across requests for ``schedule``.

    Keeps the trailing ``H`` window ``[chunk_end-H, chunk_end)``. The diagonal
    orderings read the clean KV of each retained chunk, so one version per chunk
    is enough; a Same-Step consumer reads the version matching its own denoise
    step, so every denoise step of the window has to survive. Same-Step
    additionally keeps absolute chunk-0 clean sink cells for the whole session —
    later plans still demand ``(0, D, *)`` even after the trailing window has
    moved past chunk 0.
    """
    if num_blocks < 1:
        return set()
    t_clean = schedule.num_denoise_steps
    cells: set[Cell] = set()
    h = schedule.kv_history_chunks
    if h >= 1:
        end = schedule.chunk_end
        start = max(0, end - h)
        steps = range(schedule.num_denoise_steps) if schedule.ordering is Ordering.SAME_STEP else (t_clean,)
        cells |= {(chunk, step, block) for chunk in range(start, end) for step in steps for block in range(num_blocks)}
    if schedule.ordering is Ordering.SAME_STEP:
        cells |= {(0, t_clean, block) for block in range(num_blocks)}
    return cells


def prior_history_in_pool(
    cache: NoisyKVCache,
    kv_id: str,
    *,
    chunk_base: int,
) -> frozenset[Cell]:
    """Resident cells for ``kv_id`` with ``chunk < chunk_base`` (outer session supplies base)."""
    if chunk_base <= 0:
        return frozenset()
    return frozenset(
        (chunk, step, block) for sid, chunk, step, block in cache.pool.keys if sid == kv_id and chunk < chunk_base
    )

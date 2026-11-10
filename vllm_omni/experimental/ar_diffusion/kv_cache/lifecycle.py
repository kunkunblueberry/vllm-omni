# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Backend-neutral allocation-retention operations for AR-Diffusion KV.

Storage backends own their allocation identity, synchronization, publication,
rollback and actual release. This module owns the shared set operation: retain
only allocations named by the committed history, releasing all other owned
allocations exactly once. It deliberately knows nothing about tokens, pages,
segments, Cells or tensors.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar

AllocationT = TypeVar("AllocationT")


def release_unretained(
    allocations: Iterable[AllocationT],
    retained: Iterable[AllocationT],
    release: Callable[[AllocationT], None],
    *,
    before_release: Callable[[AllocationT], None] | None = None,
) -> None:
    """Release owned allocations absent from ``retained``, in allocation order.

    Snapshot candidates before callbacks, since release may mutate ownership.
    If supplied, complete ``before_release`` for every candidate before freeing
    any storage. Noisy uses that phase to retire pending receives; segmented
    commits already synchronize publication before entering this operation.
    The caller supplies only its own allocations, validates retention and ensures
    compute/communication last-use. Retained ordering stays backend-specific.
    """
    keep = set(retained)
    dropped = tuple(allocation for allocation in allocations if allocation not in keep)
    if before_release is not None:
        for allocation in dropped:
            before_release(allocation)
    for allocation in dropped:
        release(allocation)

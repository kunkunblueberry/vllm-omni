# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Session capacity reservation, one ledger per worker.

The roadmap splits KV capacity into two jobs (2.3): admission reserves the
capacity a request needs, and execution only uses what was reserved. A session's
retained history counts against its ongoing holding, so a follow-up request
reserves the *increment* rather than the whole window.

The ledger is installed by the party that owns the pool, because that is the only
party that knows the geometry the budget is expressed in. ``NoisyKVCache`` states
what a pool key holds and how wide a window one session may occupy, so the runner
installs a ledger over the pool's own reservation when it builds it: grants are
charged against the bytes the pool actually reserved, and a ledger that lives in
the worker holding the storage cannot outlive the replica that owns it.

A worker never grows a session past its grant: when the demand does not fit, the
caller gets ``ReservationDeniedError`` and the request is refused at admission
instead of taking capacity from another session.

Admission is only half of the loop. After a batch the worker reports what the
session actually holds (``CapacityUsage``: logical versions plus worker bytes) and
the ledger reconciles it against the grant. A worker that holds more than it was
granted is an overcommit: the session fails closed instead of silently borrowing
capacity from another session. Releasing a key returns its grant, so reset, close,
and a failed forward all leave the budget where they found it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vllm.logger import init_logger

logger = init_logger(__name__)


class ReservationDeniedError(RuntimeError):
    """The session's incremental demand does not fit the remaining budget."""


@dataclass(frozen=True)
class CapacityDemand:
    """The total holding a request needs for its session.

    Absolute, not incremental: a session that already holds history passes the
    whole window it needs, and :meth:`SessionCapacityLedger.reserve` charges only
    the part not already held. That keeps the caller from having to know what the
    session currently owns, and it makes the demand directly comparable with a
    worker's reported ``CapacityUsage``.
    """

    versions: int
    bytes: int

    def __post_init__(self) -> None:
        if self.versions < 0 or self.bytes < 0:
            raise ValueError(f"capacity demand must be non-negative, got {self}")


@dataclass(frozen=True)
class CapacityGrant:
    """What the scheduler reserved for a session."""

    session_id: str
    versions: int
    bytes: int
    incremental: bool


@dataclass(frozen=True)
class CapacityUsage:
    """What a worker reports a session actually holds after a batch.

    ``versions`` is the logical retained-version count and ``bytes`` the
    worker-side footprint. Both are strategy-neutral: the paged layout and the
    Noisy ``VersionPool`` report the same pair even though their storage and
    logical block granularity differ.
    """

    versions: int
    bytes: int

    def __post_init__(self) -> None:
        if self.versions < 0 or self.bytes < 0:
            raise ValueError(f"capacity usage must be non-negative, got {self}")


class ReservationOvercommitError(ReservationDeniedError):
    """The worker holds more than the scheduler granted this session."""


@dataclass
class SessionCapacityLedger:
    """Per-session capacity reservations against one device budget."""

    budget_bytes: int
    _held: dict[str, int] = field(default_factory=dict)
    _granted_versions: dict[str, int] = field(default_factory=dict)
    _grants: dict[str, CapacityGrant] = field(default_factory=dict)
    _usage: dict[str, CapacityUsage] = field(default_factory=dict)
    _overcommit: dict[str, CapacityUsage] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.budget_bytes < 0:
            raise ValueError(f"budget_bytes must be non-negative, got {self.budget_bytes}")

    def held_bytes(self, session_id: str) -> int:
        return self._held.get(session_id, 0)

    def used_bytes(self) -> int:
        return sum(self._held.values())

    def free_bytes(self) -> int:
        return self.budget_bytes - self.used_bytes()

    def reserve(self, session_id: str, demand: CapacityDemand) -> CapacityGrant:
        """Reserve what ``demand`` asks for, or refuse it.

        ``demand`` is absolute (the holding the request needs), so the ledger
        charges only ``demand.bytes - held``. A session that already holds enough
        only needs a temporary grant: its history is counted once, not per
        request. Granting is all-or-nothing on both halves of the pair.
        """
        held = self._held.get(session_id, 0)
        incremental = demand.bytes > held
        needed = demand.bytes - held if incremental else 0
        if needed > self.free_bytes():
            raise ReservationDeniedError(
                f"session {session_id!r} needs {needed} more bytes but only "
                f"{self.free_bytes()} of {self.budget_bytes} are free"
            )
        if incremental:
            self._held[session_id] = demand.bytes
        self._granted_versions[session_id] = demand.versions
        grant = CapacityGrant(
            session_id=session_id,
            versions=demand.versions,
            bytes=self._held.get(session_id, demand.bytes),
            incremental=incremental,
        )
        self._grants[session_id] = grant
        return grant

    def grant(self, session_id: str) -> CapacityGrant | None:
        return self._grants.get(session_id)

    def reconcile(self, session_id: str, usage: CapacityUsage) -> CapacityUsage:
        """Accept a worker's reported usage, or fail closed on an overcommit.

        The grant is the contract; a worker that reports more than it was granted
        cannot be absorbed by another session's budget, so the ledger records the
        overcommit and raises. The caller decides the session's fate (this module
        keeps accounting only).
        """
        held = self._held.get(session_id, 0)
        granted_versions = self._granted_versions.get(session_id, 0)
        over_bytes = usage.bytes > held
        over_versions = usage.versions > granted_versions
        if over_bytes or over_versions:
            self._overcommit[session_id] = usage
            exceeded = []
            if over_bytes:
                exceeded.append(f"{usage.bytes} bytes > {held} granted")
            if over_versions:
                exceeded.append(f"{usage.versions} versions > {granted_versions} granted")
            raise ReservationOvercommitError(f"session {session_id!r} reports {', '.join(exceeded)}")
        self._usage[session_id] = usage
        self._overcommit.pop(session_id, None)
        return usage

    def usage(self, session_id: str) -> CapacityUsage | None:
        """Last usage the worker reported for ``session_id``."""
        return self._usage.get(session_id)

    def overcommit(self, session_id: str) -> CapacityUsage | None:
        """The overcommitted usage that failed this session, if any."""
        return self._overcommit.get(session_id)

    def release(self, session_id: str) -> int:
        """Give a session's holding back (reset, close, or a dead replica)."""
        released = self._held.pop(session_id, 0)
        self._granted_versions.pop(session_id, None)
        self._grants.pop(session_id, None)
        self._usage.pop(session_id, None)
        self._overcommit.pop(session_id, None)
        return released

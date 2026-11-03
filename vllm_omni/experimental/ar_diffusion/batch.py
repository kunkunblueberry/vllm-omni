# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""AR-Diffusion execution envelopes.

This layer deliberately separates request/session lifecycle from strategy-specific
execution. Paged AR-Diffusion may carry several request records; the Noisy cell
strategy is a continuous request execution and therefore has exactly one record.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ARDiffusionExecutionMode(str, Enum):
    """Execution cardinality and scheduling contract for an AR batch."""

    PAGED_CONTINUOUS = "paged_continuous"
    NOISY_CONTINUOUS = "noisy_continuous"


@dataclass
class ARDiffusionRequestRecord:
    """One request's lifecycle envelope inside an AR-Diffusion batch."""

    request_id: str
    session_id: str | None = None
    chunk_start: int = 0
    strategy_payload: Any | None = None
    model_state: Any | None = None
    owned_request_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("ARDiffusionRequestRecord.request_id must be non-empty")
        if self.chunk_start < 0:
            raise ValueError(f"chunk_start must be non-negative, got {self.chunk_start}")
        if not self.owned_request_ids:
            self.owned_request_ids = (self.request_id,)


@dataclass
class ARDiffusionBatch:
    """Strategy-neutral AR execution envelope.

    The envelope unifies request identity, session/chunk metadata, ownership,
    cleanup and output attribution. It does not impose one tensor batching
    implementation on paged and Noisy strategies.
    """

    mode: ARDiffusionExecutionMode
    records: list[ARDiffusionRequestRecord]
    metadata: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode is ARDiffusionExecutionMode.NOISY_CONTINUOUS and len(self.records) != 1:
            raise ValueError("Noisy AR-Diffusion batches require exactly one request record")

    @classmethod
    def for_noisy(
        cls,
        request_id: str,
        *,
        session_id: str | None = None,
        chunk_start: int = 0,
        model_state: Any | None = None,
    ) -> ARDiffusionBatch:
        return cls(
            mode=ARDiffusionExecutionMode.NOISY_CONTINUOUS,
            records=[
                ARDiffusionRequestRecord(
                    request_id=request_id,
                    session_id=session_id,
                    chunk_start=chunk_start,
                    model_state=model_state,
                )
            ],
        )

    @classmethod
    def for_paged(
        cls,
        request_ids: Sequence[str],
        *,
        session_ids: Sequence[str | None] | None = None,
    ) -> ARDiffusionBatch:
        """Envelope for the paged strategy, which may carry several records.

        Paged execution is the only mode that admits more than one request per
        envelope; the strategy keeps its own storage and batching implementation
        behind ``strategy_payload``.
        """
        ids = tuple(request_ids)
        if not ids:
            raise ValueError("Paged AR-Diffusion batches require at least one request record")
        if len(set(ids)) != len(ids):
            raise ValueError("Paged AR-Diffusion batch request ids must be unique")
        sessions = tuple(session_ids) if session_ids is not None else (None,) * len(ids)
        if len(sessions) != len(ids):
            raise ValueError("session_ids must align with request_ids")
        return cls(
            mode=ARDiffusionExecutionMode.PAGED_CONTINUOUS,
            records=[
                ARDiffusionRequestRecord(request_id=request_id, session_id=session_id)
                for request_id, session_id in zip(ids, sessions, strict=True)
            ],
        )

    @property
    def request_ids(self) -> tuple[str, ...]:
        return tuple(record.request_id for record in self.records)

    @property
    def record(self) -> ARDiffusionRequestRecord:
        if len(self.records) != 1:
            raise ValueError("ARDiffusionBatch.record requires exactly one request record")
        return self.records[0]

    def set_strategy_payload(self, request_id: str, payload: Any) -> None:
        for record in self.records:
            if record.request_id == request_id:
                record.strategy_payload = payload
                return
        raise KeyError(f"request {request_id!r} is not part of the AR-Diffusion batch")

    def set_output(self, request_id: str, output: Any) -> None:
        if request_id not in self.request_ids:
            raise KeyError(f"request {request_id!r} is not part of the AR-Diffusion batch")
        self.outputs[request_id] = output

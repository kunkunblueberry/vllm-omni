# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""AR-Diffusion Session/Batch data-layer contract tests."""

from __future__ import annotations

import pytest

from vllm_omni.experimental.ar_diffusion.batch import (
    ARDiffusionBatch,
    ARDiffusionExecutionMode,
    ARDiffusionRequestRecord,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_noisy_batch_is_strictly_single_record() -> None:
    batch = ARDiffusionBatch.for_noisy("request-1", session_id="session-1", chunk_start=3)

    assert batch.mode is ARDiffusionExecutionMode.NOISY_CONTINUOUS
    assert batch.request_ids == ("request-1",)
    assert batch.record.session_id == "session-1"
    assert batch.record.chunk_start == 3
    assert batch.record.owned_request_ids == ("request-1",)

    with pytest.raises(ValueError, match="exactly one request record"):
        ARDiffusionBatch(
            mode=ARDiffusionExecutionMode.NOISY_CONTINUOUS,
            records=[
                ARDiffusionRequestRecord("request-1"),
                ARDiffusionRequestRecord("request-2"),
            ],
        )


def test_paged_batch_can_carry_multiple_request_records() -> None:
    batch = ARDiffusionBatch(
        mode=ARDiffusionExecutionMode.PAGED_CONTINUOUS,
        records=[
            ARDiffusionRequestRecord("request-1", session_id="session-1"),
            ARDiffusionRequestRecord("request-2"),
        ],
    )

    assert batch.request_ids == ("request-1", "request-2")
    batch.set_strategy_payload("request-2", {"paged": True})
    batch.set_output("request-1", "output-1")

    assert batch.records[1].strategy_payload == {"paged": True}
    assert batch.outputs == {"request-1": "output-1"}


def test_batch_rejects_unknown_request_payload_and_output() -> None:
    batch = ARDiffusionBatch.for_noisy("request-1")

    with pytest.raises(KeyError, match="request-2"):
        batch.set_strategy_payload("request-2", object())
    with pytest.raises(KeyError, match="request-2"):
        batch.set_output("request-2", object())


def test_request_record_rejects_invalid_identity_and_chunk() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ARDiffusionRequestRecord("")
    with pytest.raises(ValueError, match="non-negative"):
        ARDiffusionRequestRecord("request-1", chunk_start=-1)


def test_paged_batch_accepts_several_records_and_rejects_duplicates() -> None:
    batch = ARDiffusionBatch.for_paged(["request-1", "request-2"], session_ids=["s1", None])

    assert batch.mode is ARDiffusionExecutionMode.PAGED_CONTINUOUS
    assert batch.request_ids == ("request-1", "request-2")
    assert [record.session_id for record in batch.records] == ["s1", None]
    assert batch.records[1].owned_request_ids == ("request-2",)

    with pytest.raises(ValueError, match="unique"):
        ARDiffusionBatch.for_paged(["request-1", "request-1"])
    with pytest.raises(ValueError, match="at least one"):
        ARDiffusionBatch.for_paged([])
    with pytest.raises(ValueError, match="align"):
        ARDiffusionBatch.for_paged(["request-1", "request-2"], session_ids=["s1"])

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real Ming bridge failures must retire one request, not the orchestrator."""

from types import SimpleNamespace

import pytest
from vllm.sampling_params import SamplingParams

from tests.engine.test_orchestrator import FakeStageClient, _build_stage_pools
from tests.engine.test_orchestrator_error_handling import _build_bare_orchestrator
from tests.model_executor.models.ming_image.test_config_and_pipeline import _source_output
from vllm_omni.engine.messages import ErrorMessage
from vllm_omni.engine.orchestrator import OrchestratorRequestState
from vllm_omni.errors import OmniClientError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.stage_input_processors.ming_image import thinker2image

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _state(name, *, prompt=None, extra=None):
    return OrchestratorRequestState(
        request_id=name,
        prompt=prompt or {},
        sampling_params_list=[SamplingParams(max_tokens=1), OmniDiffusionSamplingParams(extra_args=extra or {})],
        final_stage_id=1,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid",
    [
        "negative_prompt",
        "multiple_references",
        "zero_layers",
        "string_layers",
        "null_layers",
        "list_layers",
        "dict_layers",
        "infinite_layers",
    ],
)
async def test_real_ming_bridge_rejection_retires_only_bad_request_and_allows_next(invalid):
    ar = FakeStageClient()
    dit = FakeStageClient(stage_type="diffusion", final_output=True, final_output_type="image")
    dit.custom_process_input_func = thinker2image
    orchestrator, queues = _build_bare_orchestrator(_build_stage_pools([[ar], [dit]]))
    prompt, extra = {}, {}
    if invalid == "negative_prompt":
        extra["negative_prompt"] = "avoid blur"
    elif invalid == "multiple_references":
        prompt["multi_modal_data"] = {"img2img": [object(), object()]}
    else:
        extra["num_layers"] = {
            "zero_layers": 0,
            "string_layers": "abc",
            "null_layers": None,
            "list_layers": [],
            "infinite_layers": float("inf"),
            "dict_layers": {},
        }[invalid]
    bad, peer = _state("bad", prompt=prompt, extra=extra), _state("peer")
    orchestrator.request_states.update(bad=bad, peer=peer)
    for pool in orchestrator.stage_pools:
        pool.select_replica_id("bad")
        pool.select_replica_id("peer")
    try:
        await orchestrator._forward_to_next_stage("bad", 0, _source_output(), bad)
        error = queues[1].async_q.get_nowait()
        assert isinstance(error, ErrorMessage)
        assert error.request_id == "bad" and error.stage_id == 1
        assert error.status_code == 400 and error.error_type == "BadRequestError" and not error.fatal
        assert queues[1].async_q.empty()
        assert orchestrator.request_states == {"peer": peer}
        assert dit.add_request_calls == []
        assert ar.shutdown_calls == dit.shutdown_calls == 0
        for pool in orchestrator.stage_pools:
            assert pool.live_replica_ids() == [0]
            assert pool.get_bound_replica_id("bad") is None
            assert pool.get_bound_replica_id("peer") == 0
        assert all("peer" not in ids for ids in ar.abort_calls + dit.abort_calls)
        next_state = _state("next")
        orchestrator.request_states["next"] = next_state
        await orchestrator._forward_to_next_stage("next", 0, _source_output(), next_state)
        assert len(dit.add_request_calls) == 1
        assert "next" in orchestrator.request_states and orchestrator.request_states["peer"] is peer
        assert queues[1].async_q.empty()
    finally:
        for q in queues:
            q.close()


def test_ming_input_rejection_has_client_error_metadata():
    with pytest.raises(OmniClientError) as excinfo:
        thinker2image([_source_output()], sampling_params=SimpleNamespace(extra_args={"negative_prompt": "avoid blur"}))
    assert excinfo.value.status_code == 400 and excinfo.value.error_type == "BadRequestError"


@pytest.mark.asyncio
async def test_internal_ming_output_error_is_not_mislabeled_as_client_error():
    dit = FakeStageClient(stage_type="diffusion", final_output=True, final_output_type="image")
    dit.custom_process_input_func = thinker2image
    orchestrator, queues = _build_bare_orchestrator(_build_stage_pools([[FakeStageClient()], [dit]]))
    state = _state("internal")
    orchestrator.request_states["internal"] = state
    source = _source_output()
    source.outputs = []
    try:
        with pytest.raises(ValueError, match="thinker produced no outputs") as excinfo:
            await orchestrator._forward_to_next_stage("internal", 0, source, state)
        assert not isinstance(excinfo.value, OmniClientError)
        assert queues[1].async_q.empty()
        assert dit.add_request_calls == []
    finally:
        for q in queues:
            q.close()

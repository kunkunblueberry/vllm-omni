# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ming producer/consumer boundaries, mapped from Qwen size/layers guards.

Ming requires positive multiples of 16, exactly 256 query states, nonempty
6144-wide direct states, and one reference for Layer. It must reject, not
silently clamp to Qwen's geometry convention. No checkpoint or GPU is loaded.
"""

import copy

import pytest
import torch

from tests.diffusion.models.ming_image.test_batch_step_contract import _owned_request, _wave_pipeline
from vllm_omni.diffusion.worker.utils import StepRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize("height,width", [(16, 16), (16, 32), (32, 16), (64, 80), (272, 256)])
@pytest.mark.parametrize("layered", [False, True])
def test_minimum_and_non_square_geometry_preserves_five_axis_latents(monkeypatch, height, width, layered):
    pipeline, seen = _wave_pipeline(monkeypatch, layered=layered)
    request = _owned_request("shape", layered=layered)
    request.sampling_params.height, request.sampling_params.width = height, width
    if layered:
        request.prompt["extra"]["reference_image"] = torch.ones(1, 4, height, width)
    state = StepRequestState(request.request_id, copy.deepcopy(request.sampling_params), request.prompt)
    pipeline.prepare_encode(state)
    assert state.latents.shape == (1, 4, 4 if layered else 1, height // 8, width // 8)
    assert state.scheduler is not pipeline.scheduler
    assert seen == []


@pytest.mark.parametrize("height,width", [(0, 16), (-16, 16), (1, 1), (15, 16), (17, 16), (31, 33), (32, 0)])
def test_invalid_geometry_is_rejected_before_dit_and_cleans_state(monkeypatch, height, width):
    pipeline, seen = _wave_pipeline(monkeypatch)
    request = _owned_request("invalid")
    request.sampling_params.height, request.sampling_params.width = height, width
    state = StepRequestState(request.request_id, copy.deepcopy(request.sampling_params), request.prompt)
    with pytest.raises(ValueError, match="positive multiples"):
        pipeline.prepare_encode(state)
    assert seen == []
    assert pipeline._num_frames_per_prompt == 1
    assert pipeline._pending_prompt_embeds is None


@pytest.mark.parametrize("layers", [1, 2, 6])
def test_layer_count_includes_one_composite_and_excludes_reference(monkeypatch, layers):
    pipeline, _ = _wave_pipeline(monkeypatch, layered=True)
    request = _owned_request("layers", layered=True)
    request.sampling_params.extra_args["num_layers"] = layers
    state = StepRequestState(request.request_id, copy.deepcopy(request.sampling_params), request.prompt)
    pipeline.prepare_encode(state)
    assert state.latents.shape[2] == layers + 1
    assert state.extra["ref_latent"].shape[2] == 1


@pytest.mark.parametrize("layers", [-1, 0, 2, 6])
def test_design_rejects_layer_count_that_changes_its_single_frame(monkeypatch, layers):
    pipeline, seen = _wave_pipeline(monkeypatch)
    request = _owned_request("design")
    request.sampling_params.extra_args["num_layers"] = layers
    state = StepRequestState(request.request_id, copy.deepcopy(request.sampling_params), request.prompt)
    with pytest.raises(ValueError, match="num_layers|exactly one"):
        pipeline.prepare_encode(state)
    assert seen == [] and pipeline._num_frames_per_prompt == 1


@pytest.mark.parametrize(
    "kind",
    [
        "missing_query",
        "query255",
        "query257",
        "query_batch",
        "missing_direct",
        "direct_empty",
        "direct_width",
        "direct_batch",
    ],
)
def test_invalid_producer_conditions_fail_before_dit(monkeypatch, kind):
    pipeline, seen = _wave_pipeline(monkeypatch)
    request = _owned_request("conditions")
    extra = request.prompt["extra"]
    if kind == "missing_query":
        extra.pop("query_hidden_states")
    elif kind.startswith("query"):
        shape = {"query255": (255, 2048), "query257": (257, 2048), "query_batch": (2, 256, 2048)}[kind]
        extra["query_hidden_states"] = torch.zeros(shape)
    elif kind == "missing_direct":
        extra.pop("direct_hidden_states")
    else:
        shape = {"direct_empty": (0, 6144), "direct_width": (4, 6143), "direct_batch": (2, 4, 6144)}[kind]
        extra["direct_hidden_states"] = torch.zeros(shape)
    state = StepRequestState(request.request_id, copy.deepcopy(request.sampling_params), request.prompt)
    with pytest.raises(ValueError, match="conditions|conditioning"):
        pipeline.prepare_encode(state)
    assert seen == [] and pipeline._pending_prompt_embeds is None


@pytest.mark.parametrize("seed", [0, 1, 2**31 - 1, 2**32 - 1])
def test_seed_boundaries_use_private_generator_and_do_not_mutate_global_rng(monkeypatch, seed):
    pipeline, _ = _wave_pipeline(monkeypatch)
    request = _owned_request("seed", seed=seed)
    state = StepRequestState(request.request_id, copy.deepcopy(request.sampling_params), request.prompt)
    global_rng = torch.random.get_rng_state().clone()
    pipeline.prepare_encode(state)
    expected = torch.randn(1, 4, 4, 6, generator=torch.Generator().manual_seed(seed)).unsqueeze(2)
    torch.testing.assert_close(state.latents, expected, rtol=0, atol=0)
    assert torch.equal(torch.random.get_rng_state(), global_rng)
    assert request.sampling_params.generator is None


@pytest.mark.parametrize("kind", ["missing", "empty_list", "two_images", "rgb", "two_rows", "wrong_rank"])
def test_reference_boundary_rejects_before_dit_and_next_prepare_succeeds(monkeypatch, kind):
    pipeline, seen = _wave_pipeline(monkeypatch, layered=True)
    request = _owned_request("reference", layered=True)
    reference = request.prompt["extra"]["reference_image"]
    values = {
        "missing": None,
        "empty_list": [],
        "two_images": [reference, reference],
        "rgb": torch.ones(1, 3, 32, 48),
        "two_rows": torch.ones(2, 4, 32, 48),
        "wrong_rank": torch.ones(4, 32, 48),
    }
    request.prompt["extra"]["reference_image"] = values[kind]
    state = StepRequestState(request.request_id, copy.deepcopy(request.sampling_params), request.prompt)
    with pytest.raises(ValueError, match="reference"):
        pipeline.prepare_encode(state)
    assert seen == [] and pipeline._num_frames_per_prompt == 1
    valid = _owned_request("next", layered=True)
    next_state = StepRequestState(valid.request_id, copy.deepcopy(valid.sampling_params), valid.prompt)
    pipeline.prepare_encode(next_state)
    assert next_state.latents.shape == (1, 4, 4, 4, 6)
    assert next_state.extra["ref_latent"].shape == (1, 4, 1, 4, 6)


@pytest.mark.parametrize("layers", [-3, -1, 0])
def test_layer_rejects_nonpositive_count_without_leaking_frame_state(monkeypatch, layers):
    pipeline, seen = _wave_pipeline(monkeypatch, layered=True)
    request = _owned_request("layers", layered=True)
    request.sampling_params.extra_args["num_layers"] = layers
    state = StepRequestState(request.request_id, copy.deepcopy(request.sampling_params), request.prompt)
    with pytest.raises(ValueError, match="at least 1"):
        pipeline.prepare_encode(state)
    assert seen == [] and pipeline._num_frames_per_prompt == 1

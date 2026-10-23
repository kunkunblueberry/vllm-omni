# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Independent B1/B2 oracles through the real bridge, batch and step framework.

The checkpoint-backed projector/DiT/VAE/Euler kernels are weight-free doubles.
Request ownership, context, CFG dispatch, admission, gather/scatter and runner
are production code. Expected predictions and Euler updates are explicit below.
"""

import copy
from types import SimpleNamespace

import pytest
import torch

from tests.diffusion.models.ming_image.test_a6_contract import _pipeline, _request
from vllm_omni.diffusion.forward_context import ForwardContext, override_forward_context
from vllm_omni.diffusion.models.interface import supports_step_execution
from vllm_omni.diffusion.models.ming_image.pipeline import (
    get_ming_image_post_process_func,
    get_ming_image_pre_process_func,
)
from vllm_omni.diffusion.models.z_image.z_image_transformer import ZImageTransformer2DModel
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched import StepScheduler
from vllm_omni.diffusion.sched.interface import DiffusionRequestStatus
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner, _normalize_pipeline_outputs
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _PrivateEuler:
    config = {}
    order = 1

    def __init__(self):
        self.cursor = 0

    def set_timesteps(self, num_inference_steps=None, device=None, mu=None, timesteps=None, sigmas=None):
        self.mu = mu
        if timesteps is not None:
            values = timesteps
        elif sigmas is not None:
            values = [s * 1000 for s in sigmas]
        else:
            values = [900, 500, 100][:num_inference_steps]
        self.timesteps = torch.as_tensor(values, dtype=torch.float32, device=device)
        self.cursor = 0

    def set_begin_index(self, index):
        self.cursor = index

    def step(self, noise, t, latents, return_dict=False):
        assert float(t) == float(self.timesteps[self.cursor]), "shared/stale scheduler cursor"
        assert noise.dtype == latents.dtype == torch.float32
        self.cursor += 1
        # Independent Euler delta sigma = 1/4, not the UUT's scheduler formula.
        return (latents + noise * 0.25,)


def _wave_pipeline(monkeypatch, *, layered=False):
    pipeline, events = _pipeline(monkeypatch, layered=layered)
    pipeline.scheduler = _PrivateEuler()
    seen = []

    def dit(self, x, t, cap_feats, *, ref_x, cap_feats_2, **kwargs):
        assert events[-1] == "marker"
        assert len(x) == len(t) == len(cap_feats) == len(cap_feats_2)
        predictions = []
        for row, (latent, query, direct) in enumerate(zip(x, cap_feats, cap_feats_2, strict=True)):
            ref = float(ref_x[row].mean()) if ref_x is not None else 0.0
            positive = bool(torch.count_nonzero(query))
            if not positive:
                assert torch.count_nonzero(direct) == 0, "negative must be zero after biased projection"
            value = float(query[0, 0]) + float(direct[0, 0]) + ref + float(t[row])
            seen.append((positive, value, ref, float(t[row]), len(x)))
            frames = latent.shape[1]
            out = torch.empty_like(latent)
            for frame in range(frames):
                out[:, frame] = value + frame
            if ref_x is not None:
                out = torch.cat((out, torch.full_like(out[:, :1], 999)), dim=1)
            predictions.append(out)
        events.append("dit")
        return predictions, None

    def decode(latents, return_dict=False):
        return (latents.clone(),)

    monkeypatch.setattr(ZImageTransformer2DModel, "forward", dit)
    monkeypatch.setattr(pipeline.vae, "decode", decode)
    return pipeline, seen


def _owned_request(name, *, layered=False, offset=0, seed=42, steps=1, truncation=1):
    # _request calls the actual AR bridge; alter valid captured hidden tensors.
    request = _request(layered=layered).requests[0]
    request.request_id = name
    extra = request.prompt["extra"]
    extra["query_hidden_states"] = extra["query_hidden_states"] + offset
    extra["direct_hidden_states"] = extra["direct_hidden_states"] + offset
    if layered:
        extra["reference_image"] = torch.full((1, 4, 32, 48), 1.0 + offset)
    request.sampling_params.seed = seed
    request.sampling_params.num_inference_steps = steps
    request.sampling_params.extra_args["cfg_truncation"] = truncation
    return request


def _initial(seed, frames):
    return torch.randn(frames, 4, 4, 6, generator=torch.Generator().manual_seed(seed)).permute(1, 0, 2, 3)[None]


def _prediction(offset, frames, t, *, layered=False, apply_cfg=True):
    # Producer q[0]=5+offset; biased direct d[0]=12+offset.
    # p=17+2*offset+ref+t+frame; n=ref+t+frame; u=p+2*(p-n).
    ref = 1 + offset if layered else 0
    base = (51 + 6 * offset if apply_cfg else 17 + 2 * offset) + ref + t
    return torch.tensor([base + f for f in range(frames)]).reshape(1, 1, frames, 1, 1).expand(1, 4, frames, 4, 6)


@pytest.mark.parametrize("layered", [False, True])
def test_request_wave_preserves_seed_condition_reference_frame_and_output(monkeypatch, layered):
    pipeline, seen = _wave_pipeline(monkeypatch, layered=layered)
    requests = [
        _owned_request("second", layered=layered, offset=2, seed=13),
        _owned_request("first", layered=layered, offset=0, seed=42),
    ]
    context = ForwardContext(direct_condition=torch.tensor([71]), ref_latent=torch.tensor([72]))
    direct, ref = context.direct_condition, context.ref_latent
    with override_forward_context(context):
        outputs = pipeline.forward(DiffusionRequestBatch(requests))
    assert context.direct_condition is direct and context.ref_latent is ref
    assert len(outputs) == 2
    frames = 4 if layered else 1
    for output, offset, seed in zip(outputs, [2, 0], [13, 42], strict=True):
        expected = _initial(seed, frames) - 0.25 * _prediction(offset, frames, 0.1, layered=layered)
        # Decode each request separately: frame index remains composite-first.
        torch.testing.assert_close(output.output, expected[0].permute(1, 0, 2, 3))
        images = get_ming_image_post_process_func(None)(output.output)
        assert len(images) == frames and all(image.mode == "RGBA" for image in images)
    assert any(item[-1] == 2 for item in seen), "compatible rows must share a real transformer invocation"
    assert pipeline.scheduler.cursor == 0, "pipeline template scheduler must stay untouched"


def test_step_unequal_progress_effective_cfg_and_private_rng(monkeypatch):
    pipeline, seen = _wave_pipeline(monkeypatch)
    assert supports_step_execution(pipeline)
    states = [
        StepRequestState(r.request_id, copy.deepcopy(r.sampling_params), r.prompt)
        for r in [
            _owned_request("a", steps=3, truncation=0.4),
            _owned_request("b", offset=2, seed=13, steps=3, truncation=0.4),
        ]
    ]
    for state in states:
        pipeline.prepare_encode(state)
    assert states[0].scheduler is not states[1].scheduler is not pipeline.scheduler
    assert states[0].sampling.generator is not states[1].sampling.generator
    # Advance a independently; next wave a is truncated, b is still guided.
    with override_forward_context(ForwardContext()):
        batch = InputBatch.make_batch(states[:1])
        pipeline.step_scheduler(states[0], pipeline.denoise_step(batch)[0:1])
        seen.clear()
        batch = InputBatch.make_batch(states)
        noise = pipeline.denoise_step(batch, states=states)
        expected_a = -_prediction(0, 1, 0.5, apply_cfg=False)
        expected_b = -_prediction(2, 1, 0.1)
        torch.testing.assert_close(noise, torch.cat([expected_a, expected_b]))
        for state, pred in zip(states, noise.split(1), strict=True):
            pipeline.step_scheduler(state, pred)
    assert [state.step_index for state in states] == [2, 1]
    assert [state.scheduler.cursor for state in states] == [2, 1]
    assert sum(not item[0] for item in seen) == 1
    torch.testing.assert_close(
        states[0].latents, _initial(42, 1) - 0.25 * (_prediction(0, 1, 0.1) + _prediction(0, 1, 0.5, apply_cfg=False))
    )
    torch.testing.assert_close(states[1].latents, _initial(13, 1) - 0.25 * _prediction(2, 1, 0.1))


def _runner(monkeypatch, pipeline):
    runner = object.__new__(DiffusionModelRunner)
    runner.pipeline = pipeline
    runner.od_config = SimpleNamespace(
        cache_backend=None, parallel_config=SimpleNamespace(use_hsdp=False), streaming_output=False
    )
    runner.vllm_config = None
    runner.device = torch.device("cpu")
    runner.state_cache = {}
    runner.input_batch = None
    runner.kv_transfer_manager = None
    # Transport/profiling boundaries do not replace scheduler/batch/runner work.
    monkeypatch.setattr(runner, "_maybe_recv_stage_payload", lambda req: None)
    monkeypatch.setattr(runner, "_maybe_send_stage_payload", lambda states, results: None)
    monkeypatch.setattr(runner, "_prepare_output_for_transport", lambda result, sampling: result)
    monkeypatch.setattr(runner, "_attach_stepwise_metadata", lambda state, result: None)
    return runner


@pytest.mark.parametrize("layered", [False, True])
def test_scheduler_input_batch_runner_late_admission_retirement_and_next_request(monkeypatch, layered):
    pipeline, _ = _wave_pipeline(monkeypatch, layered=layered)
    runner = _runner(monkeypatch, pipeline)
    scheduler = StepScheduler()
    scheduler.initialize(SimpleNamespace(max_num_seqs=2))
    preprocess = get_ming_image_pre_process_func(
        SimpleNamespace(step_execution=True, tf_model_config=SimpleNamespace(multi_frame_output=layered))
    )

    def admit(request):
        preprocess(request)
        scheduler.add_request(request)

    def wave():
        scheduled = scheduler.schedule()
        runner._cleanup_finished_step_requests(scheduled)
        output = runner._execute_stepwise_core(scheduled, record_output_peak_memory=False)
        finished = scheduler.update_from_output(scheduled, output)
        # Actual engine retires scheduler metadata when delivering results.
        for request_id in finished:
            scheduler.pop_request_state(request_id)
        return scheduled, output

    admit(_owned_request("a", layered=layered, steps=3, truncation=0.4))
    first, result = wave()
    assert first.scheduled_request_ids == ["a"]
    saved_a = runner.state_cache["a"]
    admit(_owned_request("b", layered=layered, offset=2, seed=13, steps=1, truncation=0.4))
    second, result = wave()
    assert set(second.scheduled_request_ids) == {"a", "b"}
    assert result.get_request_output("b").finished
    assert "b" not in runner.state_cache
    assert runner.input_batch is None or "b" not in runner.input_batch.request_ids
    assert runner.state_cache["a"] is saved_a and saved_a.step_index == 2
    admit(_owned_request("c", layered=layered, offset=3, seed=17, steps=1, truncation=0.4))
    _, result = wave()
    frames = 4 if layered else 1
    expected_a = _initial(42, frames) - 0.25 * sum(
        _prediction(0, frames, t, layered=layered, apply_cfg=i == 0) for i, t in enumerate([0.1, 0.5, 0.9])
    )
    torch.testing.assert_close(result.get_request_output("a").result.output, expected_a[0].permute(1, 0, 2, 3))
    expected_c = _initial(17, frames) - 0.25 * _prediction(3, frames, 0.1, layered=layered)
    torch.testing.assert_close(result.get_request_output("c").result.output, expected_c[0].permute(1, 0, 2, 3))
    assert runner.state_cache == {} and runner.input_batch is None
    assert scheduler._request_progress == {}


@pytest.mark.parametrize("layered", [False, True])
def test_selected_input_batch_has_no_capacity_padding_and_keeps_frame_axis(monkeypatch, layered):
    pipeline, _ = _wave_pipeline(monkeypatch, layered=layered)
    states = []
    for index in range(3):
        request = _owned_request(str(index), layered=layered, offset=index, seed=42 + index, steps=3)
        states.append(
            pipeline.prepare_encode(StepRequestState(request.request_id, request.sampling_params, request.prompt))
        )
    frames = 4 if layered else 1
    # Shared InputBatch is actual selected rows, not a max_num_seqs bucket.
    for mapping in ([2, 0, 1], [1], [1, 2], [2]):
        batch = InputBatch.make_batch(states, idx_mapping=torch.tensor(mapping))
        assert batch.num_reqs == batch.num_reqs_after_padding == len(mapping)
        assert batch.request_ids == [str(i) for i in mapping]
        assert batch.latents.shape == (len(mapping), 4, frames, 4, 6)
        assert batch.prompt_embeds.shape == (len(mapping), 256, 2)
        assert batch.prompt_embeds_mask is None
        for row, state_index in enumerate(mapping):
            torch.testing.assert_close(batch.latents[row : row + 1], _initial(42 + state_index, frames))
            assert float(batch.prompt_embeds[row, 0, 0]) == 5 + state_index
            assert batch.states[row] is states[state_index]
            assert batch.states[row].extra["direct_condition"].shape == (1, 4, 3)


def test_scheduler_admits_three_real_rows_and_separates_direct_shape(monkeypatch):
    pipeline, _ = _wave_pipeline(monkeypatch)
    runner = _runner(monkeypatch, pipeline)
    scheduler = StepScheduler()
    scheduler.initialize(SimpleNamespace(max_num_seqs=4))
    preprocess = get_ming_image_pre_process_func(
        SimpleNamespace(step_execution=True, tf_model_config=SimpleNamespace(multi_frame_output=False))
    )
    for index in range(4):
        request = _owned_request(str(index), steps=3)
        if index == 3:
            request.prompt["extra"]["direct_hidden_states"] = request.prompt["extra"]["direct_hidden_states"][:2]
        preprocess(request)
        scheduler.add_request(request)
    first = scheduler.schedule()
    assert first.scheduled_request_ids == ["0", "1", "2"]
    output = runner._execute_stepwise_core(first, record_output_peak_memory=False)
    scheduler.update_from_output(first, output)
    assert runner.input_batch.num_reqs == runner.input_batch.num_reqs_after_padding == 3
    assert runner.input_batch.latents.shape[0:3] == (3, 4, 1)
    assert set(runner.state_cache) == {"0", "1", "2"}
    for request_id in ("0", "1", "2"):
        scheduler.finish_requests(request_id, DiffusionRequestStatus.FINISHED_ABORTED)
        scheduler.pop_request_state(request_id)
    second = scheduler.schedule()
    assert second.scheduled_request_ids == ["3"]
    runner._cleanup_finished_step_requests(second)
    output = runner._execute_stepwise_core(second, record_output_peak_memory=False)
    assert runner.input_batch.num_reqs == runner.input_batch.num_reqs_after_padding == 1
    assert runner.input_batch.states[0].extra["direct_condition"].shape == (1, 2, 3)
    torch.testing.assert_close(runner.input_batch.latents, _initial(42, 1) - 0.25 * _prediction(0, 1, 0.1))
    scheduler.finish_requests("3", DiffusionRequestStatus.FINISHED_ABORTED)
    scheduler.pop_request_state("3")
    runner._cleanup_finished_step_requests(scheduler.schedule())
    assert runner.state_cache == {} and runner.input_batch is None
    assert scheduler._request_progress == {}


def test_empty_duplicate_and_invalid_inputs_fail_without_context_leak(monkeypatch):
    pipeline, _ = _wave_pipeline(monkeypatch, layered=True)
    with pytest.raises(ValueError, match="empty"):
        pipeline.forward(DiffusionRequestBatch([]))
    request = _owned_request("a", layered=True)
    with pytest.raises(ValueError, match="unique"):
        pipeline.forward(DiffusionRequestBatch([request, request]))
    request.prompt["extra"].pop("reference_image")
    with pytest.raises(ValueError, match="reference"):
        pipeline.forward(DiffusionRequestBatch([request]))
    assert pipeline._num_frames_per_prompt == 1


def test_supplied_latents_custom_schedule_and_generator_are_request_owned(monkeypatch):
    pipeline, _ = _wave_pipeline(monkeypatch)
    requests = [_owned_request("a"), _owned_request("b", offset=2)]
    generator = torch.Generator().manual_seed(9)
    original_rng = generator.get_state().clone()
    for index, request in enumerate(requests):
        request.sampling_params.seed = None
        request.sampling_params.generator = generator
        request.sampling_params.timesteps = torch.tensor([500.0])
        request.sampling_params.latents = torch.full((1, 4, 1, 4, 6), float(index))
    with override_forward_context(ForwardContext()):
        outputs = pipeline.forward(DiffusionRequestBatch(requests))
    for index, output in enumerate(outputs):
        torch.testing.assert_close(
            output.output,
            (torch.full((1, 4, 1, 4, 6), float(index)) - 0.25 * _prediction(index * 2, 1, 0.5))[0].permute(1, 0, 2, 3),
        )
    assert torch.equal(generator.get_state(), original_rng)
    assert all(torch.all(request.sampling_params.latents == index) for index, request in enumerate(requests))


def test_preprocessor_compatibility_tracks_frames_prefix_reference_but_not_progress():
    preprocess = get_ming_image_pre_process_func(
        SimpleNamespace(step_execution=True, tf_model_config=SimpleNamespace(multi_frame_output=True))
    )
    a = _owned_request("a", layered=True, steps=3)
    b = _owned_request("b", layered=True, offset=2, steps=1)
    preprocess(a)
    preprocess(b)
    assert a.batch_compatibility_key == b.batch_compatibility_key
    assert a.use_step_execution and a.sampling_params.num_frames == 4
    b.prompt["extra"]["direct_hidden_states"] = b.prompt["extra"]["direct_hidden_states"][:2]
    preprocess(b)
    assert a.batch_compatibility_key != b.batch_compatibility_key


def test_step_scheduler_isolates_distinct_resolutions_before_the_input_batch(monkeypatch):
    """Regression: different resolutions must not share one step wave.

    Input source: real pre_process plus StepScheduler admission for 16x16 and
    16x32 requests. Expected source: request-owned latents carry the spatial
    geometry and InputBatch rejects mixed trailing shapes, so the scheduler key
    has to keep the two requests apart.
    """
    pipeline, _ = _wave_pipeline(monkeypatch)
    runner = _runner(monkeypatch, pipeline)
    scheduler = StepScheduler()
    scheduler.initialize(SimpleNamespace(max_num_seqs=4))
    preprocess = get_ming_image_pre_process_func(
        SimpleNamespace(step_execution=True, tf_model_config=SimpleNamespace(multi_frame_output=False))
    )

    def admit(request_id, height, width):
        request = _owned_request(request_id, steps=2)
        request.sampling_params.height = height
        request.sampling_params.width = width
        preprocess(request)
        scheduler.add_request(request)
        return request

    square = admit("square", 16, 16)
    wide = admit("wide", 16, 32)
    assert square.batch_compatibility_key != wide.batch_compatibility_key

    def wave():
        scheduled = scheduler.schedule()
        runner._cleanup_finished_step_requests(scheduled)
        output = runner._execute_stepwise_core(scheduled, record_output_peak_memory=False)
        scheduler.update_from_output(scheduled, output)
        return scheduled

    # One resolution per wave: the geometry is part of the compatibility key,
    # and InputBatch rejects mixed trailing latent shapes.
    assert [wave().scheduled_request_ids for _ in range(2)] == [["square"], ["square"]]
    assert [wave().scheduled_request_ids for _ in range(2)] == [["wide"], ["wide"]]


def test_singleton_request_wave_obeys_runner_output_list_contract(monkeypatch):
    pipeline, _ = _wave_pipeline(monkeypatch)
    with override_forward_context(ForwardContext()):
        outputs = pipeline.forward(DiffusionRequestBatch([_owned_request("a")]))
    assert (
        _normalize_pipeline_outputs(
            outputs, expected_count=1, allow_single_output=False, pipeline_name=type(pipeline).__name__
        )
        == outputs
    )


def test_layer_default_cfg_survives_request_sentinel_and_preprocessing(monkeypatch):
    pipeline, _ = _wave_pipeline(monkeypatch, layered=True)
    request = _owned_request("a", layered=True)
    # Real request ingress remembers omission even after resolving scale to 1.
    request = OmniDiffusionRequest(
        request.prompt, OmniDiffusionSamplingParams(height=32, width=48, num_inference_steps=1, seed=42), "a"
    )
    assert request.sampling_params.guidance_scale == 1
    assert request.sampling_params.guidance_scale_provided is False
    preprocess = get_ming_image_pre_process_func(
        SimpleNamespace(step_execution=True, tf_model_config=SimpleNamespace(multi_frame_output=True))
    )
    preprocess(request)
    assert request.sampling_params.guidance_scale == 2
    assert request.sampling_params.do_classifier_free_guidance
    state = pipeline.prepare_encode(StepRequestState("a", request.sampling_params, request.prompt))
    assert state.do_true_cfg and state.extra["cfg"] == 2


def test_nonidentity_input_batch_and_per_sample_layer_normalization(monkeypatch):
    pipeline, _ = _wave_pipeline(monkeypatch, layered=True)
    states = []
    for index, offset in enumerate([0, 2]):
        request = _owned_request(str(index), layered=True, offset=offset)
        request.sampling_params.cfg_normalize = True
        states.append(
            pipeline.prepare_encode(StepRequestState(request.request_id, request.sampling_params, request.prompt))
        )
    batch = InputBatch.make_batch(states, idx_mapping=torch.tensor([1, 0]))
    with override_forward_context(ForwardContext()):
        noise = pipeline.denoise_step(batch, states=states)
    expected = []
    for offset in [2, 0]:
        p = _prediction(offset, 4, 0.1, layered=True, apply_cfg=False)
        u = _prediction(offset, 4, 0.1, layered=True)
        expected.append(-u * torch.linalg.vector_norm(p) / torch.linalg.vector_norm(u))
    torch.testing.assert_close(noise, torch.cat(expected))


@pytest.mark.parametrize("failure", ["prepare", "decode"])
def test_runner_request_error_retires_conditions_and_keeps_peer_then_next(monkeypatch, failure):
    pipeline, _ = _wave_pipeline(monkeypatch)
    runner = _runner(monkeypatch, pipeline)
    scheduler = StepScheduler()
    scheduler.initialize(SimpleNamespace(max_num_seqs=2))
    preprocess = get_ming_image_pre_process_func(
        SimpleNamespace(step_execution=True, tf_model_config=SimpleNamespace(multi_frame_output=False))
    )
    bad = _owned_request("bad")
    if failure == "prepare":
        bad.prompt["negative_prompt"] = "unsupported"
    else:
        decode = pipeline.post_decode

        def fail_decode(state, **kwargs):
            if state.request_id == "bad":
                raise RuntimeError("injected decode")
            return decode(state, **kwargs)

        monkeypatch.setattr(pipeline, "post_decode", fail_decode)
    for request in [bad, _owned_request("peer", offset=2, seed=13)]:
        preprocess(request)
        scheduler.add_request(request)
    scheduled = scheduler.schedule()
    output = runner._execute_stepwise_core(scheduled, record_output_peak_memory=False)
    assert output.get_request_output("bad").finished and output.get_request_output("bad").result.error
    peer = output.get_request_output("peer")
    assert peer.finished and peer.result.error is None
    torch.testing.assert_close(
        peer.result.output, (_initial(13, 1) - 0.25 * _prediction(2, 1, 0.1))[0].permute(1, 0, 2, 3)
    )
    assert runner.state_cache == {} and runner.input_batch is None
    for request_id in scheduler.update_from_output(scheduled, output):
        scheduler.pop_request_state(request_id)
    request = _owned_request("next", offset=3, seed=17)
    preprocess(request)
    scheduler.add_request(request)
    scheduled = scheduler.schedule()
    runner._cleanup_finished_step_requests(scheduled)
    output = runner._execute_stepwise_core(scheduled, record_output_peak_memory=False)
    torch.testing.assert_close(
        output.get_request_output("next").result.output,
        (_initial(17, 1) - 0.25 * _prediction(3, 1, 0.1))[0].permute(1, 0, 2, 3),
    )
    assert runner.state_cache == {} and runner.input_batch is None


def test_denoise_wave_failure_releases_runner_cache_and_restores_context(monkeypatch):
    pipeline, _ = _wave_pipeline(monkeypatch)
    runner = _runner(monkeypatch, pipeline)
    scheduler = StepScheduler()
    scheduler.initialize(SimpleNamespace(max_num_seqs=2))
    for name in ["a", "b"]:
        scheduler.add_request(_owned_request(name))

    def fail_dit(*args, **kwargs):
        raise RuntimeError("injected wave")

    monkeypatch.setattr(ZImageTransformer2DModel, "forward", fail_dit)
    outer = ForwardContext(direct_condition=torch.tensor([71]), ref_latent=torch.tensor([72]))
    direct, ref = outer.direct_condition, outer.ref_latent
    with override_forward_context(outer), pytest.raises(RuntimeError, match="injected wave"):
        runner._execute_stepwise_core(scheduler.schedule(), record_output_peak_memory=False)
    assert outer.direct_condition is direct and outer.ref_latent is ref
    assert runner.state_cache == {} and runner.input_batch is None


@pytest.mark.parametrize(
    "invalid", ["geometry", "zero_geometry", "latents", "reference_batch", "schedule", "zero_steps"]
)
def test_invalid_shape_and_empty_schedule_are_rejected_before_dit(monkeypatch, invalid):
    pipeline, seen = _wave_pipeline(monkeypatch, layered=True)
    request = _owned_request("a", layered=True)
    if invalid == "geometry":
        request.sampling_params.height = 33
    elif invalid == "zero_geometry":
        request.sampling_params.height = 0
    elif invalid == "zero_steps":
        request.sampling_params.num_inference_steps = 0
    elif invalid == "latents":
        request.sampling_params.latents = torch.zeros(4, 4, 4, 6)
    elif invalid == "reference_batch":
        request.prompt["extra"]["reference_image"] = torch.ones(2, 4, 32, 48)
    else:
        request.sampling_params.timesteps = torch.tensor([])
    with pytest.raises(ValueError):
        pipeline.forward(DiffusionRequestBatch([request]))
    assert seen == [] and pipeline._num_frames_per_prompt == 1


def test_scheduler_abort_releases_cached_batch_before_new_admission(monkeypatch):
    pipeline, _ = _wave_pipeline(monkeypatch)
    runner = _runner(monkeypatch, pipeline)
    scheduler = StepScheduler()
    scheduler.initialize(SimpleNamespace(max_num_seqs=2))
    scheduler.add_request(_owned_request("abort", steps=3))
    scheduled = scheduler.schedule()
    output = runner._execute_stepwise_core(scheduled, record_output_peak_memory=False)
    scheduler.update_from_output(scheduled, output)
    assert "abort" in runner.state_cache and runner.input_batch is not None
    scheduler.finish_requests("abort", DiffusionRequestStatus.FINISHED_ABORTED)
    scheduler.pop_request_state("abort")
    scheduler.add_request(_owned_request("next", seed=13, offset=2))
    scheduled = scheduler.schedule()
    runner._cleanup_finished_step_requests(scheduled)
    assert "abort" not in runner.state_cache and runner.input_batch is None
    output = runner._execute_stepwise_core(scheduled, record_output_peak_memory=False)
    torch.testing.assert_close(
        output.get_request_output("next").result.output,
        (_initial(13, 1) - 0.25 * _prediction(2, 1, 0.1))[0].permute(1, 0, 2, 3),
    )
    assert runner.state_cache == {} and runner.input_batch is None

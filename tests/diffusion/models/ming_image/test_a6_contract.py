# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Weight-free Ming contracts; provenance is in the Ming-Image recipe.

The real bridge, conditioning forward, pipeline, CFG dispatcher, transformer
adapter and image consumer run here. Only checkpoint-backed kernels are doubles.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.distributed import cfg_parallel
from vllm_omni.diffusion.forward_context import ForwardContext, override_forward_context
from vllm_omni.diffusion.models.ming_image.condition import MingImageConditioning
from vllm_omni.diffusion.models.ming_image.pipeline import (
    MingImageDiffusionPipeline,
    get_ming_image_post_process_func,
)
from vllm_omni.diffusion.models.ming_image.transformer import MingImageTransformer2DModel
from vllm_omni.diffusion.models.z_image.z_image_transformer import ZImageTransformer2DModel
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.stage_input_processors.ming_image import thinker2image

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Query(torch.nn.Module):
    def forward(self, hidden):
        return hidden[..., :2]


class _Direct(torch.nn.Module):
    def forward(self, hidden):
        # A bias makes zeroing hidden states before projection observably wrong.
        return hidden[..., [0, 2048, 4096]] + 7


class _Euler:
    config = {}
    order = 1

    def set_timesteps(self, num_inference_steps, device=None, mu=None):
        self.mu = mu
        self.timesteps = torch.tensor([500.0], device=device)

    def step(self, noise, t, latents, return_dict=False):
        assert noise.shape == latents.shape
        assert noise.dtype == latents.dtype == torch.float32
        self.noise = noise.clone()
        # Hand-chosen delta sigma = 1; verifies scheduler sign independently.
        return (latents + noise,)


class _RGBA:
    dtype = torch.float32
    config = SimpleNamespace(scaling_factor=1.0, shift_factor=0.0)

    def encode(self, image):
        latent = image[:, :, :, ::8, ::8]
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: latent))

    def decode(self, latents, return_dict=False):
        self.latents = latents.clone()
        # Distinct RGBA sentinels for composite, then layers.
        image = torch.zeros_like(latents)
        for frame in range(len(image)):
            image[frame, 0] = -1 + frame * 0.5
            image[frame, 3] = (-1, 0, 1, 0)[frame]
        return (image,)


def _request(*, layered=False, cfg=2.0, negative=None):
    ids = [11] * 4 + [157158] + [157157] * 256 + [157159]
    length = len(ids)
    final = torch.arange(length, dtype=torch.float32)[:, None].expand(-1, 2048)
    payload = {"final_hidden_states": final}
    payload.update({f"hidden_states_{i}": torch.full((length, 2048), float(i)) for i in (5, 12, 20)})
    source = SimpleNamespace(prompt_token_ids=ids, outputs=[SimpleNamespace(multimodal_output=payload)])
    reference = torch.ones(1, 4, 32, 48) if layered else None
    prompts = thinker2image(
        [source],
        prompt={"multi_modal_data": {"img2img": reference}},
        sampling_params=SimpleNamespace(extra_args={"num_layers": 3 if layered else 1}),
    )
    extra = prompts[0]["extra"]
    assert extra["query_hidden_states"][:, 0].tolist() == list(range(5, 261))
    assert extra["direct_hidden_states"].shape == (4, 6144)
    assert extra["direct_hidden_states"][0, [0, 2048, 4096]].tolist() == [5, 12, 20]
    if negative is not None:
        prompts[0]["negative_prompt"] = negative
    sampling = OmniDiffusionSamplingParams(height=32, width=48, num_inference_steps=1, guidance_scale=cfg, seed=42)
    return DiffusionRequestBatch(
        requests=[OmniDiffusionRequest(prompt=prompts[0], sampling_params=sampling, request_id="a6")]
    )


def _pipeline(monkeypatch, *, layered=False, fail_branch=None):
    pipeline = MingImageDiffusionPipeline.__new__(MingImageDiffusionPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = SimpleNamespace(dtype=torch.float32)
    pipeline.device = pipeline._execution_device = torch.device("cpu")
    pipeline.is_layer_decomposition = layered
    pipeline.default_num_inference_steps = 12
    pipeline.default_guidance_scale = 2.0 if layered else 1.0
    pipeline._num_frames_per_prompt = 1
    pipeline._pending_prompt_embeds = pipeline._pending_negative_prompt_embeds = None
    pipeline._uses_cudagraph_trees = True
    pipeline._interrupt = False
    pipeline.vae_scale_factor = 8
    pipeline.vae = _RGBA()
    pipeline.scheduler = _Euler()
    # The B1/B2 loop uses a private scheduler, keep a test-only observation
    # of the real dispatch instead of expecting mutation of the template.
    original_step = _Euler.step

    def observe_step(scheduler, *args, **kwargs):
        pipeline.observed_scheduler = scheduler
        return original_step(scheduler, *args, **kwargs)

    monkeypatch.setattr(_Euler, "step", observe_step)
    conditioning = MingImageConditioning.__new__(MingImageConditioning)
    torch.nn.Module.__init__(conditioning)
    conditioning.query_encoder = _Query()
    conditioning.direct_projector = _Direct()
    pipeline.conditioning = conditioning
    transformer = MingImageTransformer2DModel.__new__(MingImageTransformer2DModel)
    torch.nn.Module.__init__(transformer)
    transformer.in_channels = 4
    pipeline.transformer = transformer
    events = []

    def dit(self, x, t, cap_feats, *, ref_x, cap_feats_2, **kwargs):
        branch = "negative" if torch.count_nonzero(cap_feats[0]) == 0 else "positive"
        assert events and events[-1] == "marker", "missing marker before actual DiT branch"
        events.append(branch)
        assert len(x) == len(cap_feats) == len(cap_feats_2) == 1
        assert x[0].shape == (4, 4 if layered else 1, 4, 6)
        assert cap_feats_2[0].shape == (4, 3)
        expected_direct = [0, 0, 0] if branch == "negative" else [12, 19, 27]
        assert cap_feats_2[0][0].tolist() == expected_direct
        if layered:
            assert len(ref_x) == 1 and ref_x[0].shape == (4, 1, 4, 6)
            assert torch.equal(ref_x[0], torch.ones_like(ref_x[0]))
        else:
            assert ref_x is None
        if fail_branch == branch:
            raise RuntimeError(f"injected {branch}")
        frames = x[0].shape[1] + (1 if layered else 0)
        out = torch.full((4, frames, 4, 6), 2.0 if branch == "positive" else 1.0)
        if layered:
            out[:, -1] = 99  # Reference prediction must be cropped by the real adapter.
        return [out], None

    monkeypatch.setattr(ZImageTransformer2DModel, "forward", dit)
    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", lambda: events.append("marker"))
    monkeypatch.setattr(cfg_parallel, "is_cfg_group_initialized", lambda: False)
    return pipeline, events


@pytest.mark.parametrize(("layered", "cfg"), [(False, 0), (False, 1), (False, 2), (True, 2)])
def test_bridge_to_dit_to_rgba(monkeypatch, layered, cfg):
    pipeline, events = _pipeline(monkeypatch, layered=layered)
    frames = 4 if layered else 1
    expected_initial = torch.randn(frames, 4, 4, 6, generator=torch.Generator().manual_seed(42))
    context = ForwardContext(direct_condition=torch.tensor([71]), ref_latent=torch.tensor([72]))
    old_direct, old_ref = context.direct_condition, context.ref_latent
    with override_forward_context(context):
        output = pipeline.forward(_request(layered=layered, cfg=cfg))[0]
    assert context.direct_condition is old_direct and context.ref_latent is old_ref
    assert pipeline._pending_prompt_embeds is pipeline._pending_negative_prompt_embeds is None
    assert pipeline._num_frames_per_prompt == 1
    assert events == (["marker", "positive", "marker", "negative"] if cfg > 1 else ["marker", "positive"])
    assert pipeline.observed_scheduler.mu == pytest.approx(0.5 + (6 - 256) * (0.65 / 3840))
    assert torch.equal(pipeline.observed_scheduler.noise, torch.full((1, 4, frames, 4, 6), -4.0 if cfg > 1 else -2.0))
    assert torch.equal(pipeline.vae.latents.squeeze(2), expected_initial - (4 if cfg > 1 else 2))
    images = get_ming_image_post_process_func(None)(output.output)
    assert len(images) == frames and all(image.mode == "RGBA" for image in images)
    assert [image.getpixel((0, 0)) for image in images] == [
        (0, 128, 128, 0),
        (64, 128, 128, 128),
        (128, 128, 128, 255),
        (191, 128, 128, 128),
    ][:frames]


@pytest.mark.parametrize("branch", ["positive", "negative"])
def test_branch_error_restores_outer_context(monkeypatch, branch):
    pipeline, events = _pipeline(monkeypatch, layered=True, fail_branch=branch)
    context = ForwardContext(direct_condition=torch.tensor([71]), ref_latent=torch.tensor([72]))
    old_direct, old_ref = context.direct_condition, context.ref_latent
    with override_forward_context(context), pytest.raises(RuntimeError, match=f"injected {branch}"):
        pipeline.forward(_request(layered=True))
    assert context.direct_condition is old_direct and context.ref_latent is old_ref
    assert pipeline._pending_prompt_embeds is pipeline._pending_negative_prompt_embeds is None
    assert pipeline._num_frames_per_prompt == 1
    assert events[-2:] == ["marker", branch]


def test_ming_literal_cfg_and_whole_sample_norm():
    pipeline = MingImageDiffusionPipeline.__new__(MingImageDiffusionPipeline)
    p = torch.tensor([[[[[2.0]]]]])
    n = torch.tensor([[[[[1.0]]]]])
    assert pipeline.combine_cfg_noise((p,), (n,), 2).item() == 4
    # Different samples; full C/F/H/W norm, only shrink, zero denominator safe.
    p = torch.tensor([[3.0, 4.0], [0, 0], [3, 4]]).reshape(3, 1, 2, 1, 1)
    n = torch.tensor([[0.0, 0.0], [0, 0], [3, 4]]).reshape_as(p)
    expected = torch.tensor([[4.5, 6.0], [0, 0], [3, 4]]).reshape_as(p)
    torch.testing.assert_close(pipeline.combine_cfg_noise(p, n, 2, 1.5), expected)
    # Nonuniform residual across frames distinguishes a whole-sample clamp
    # from independently clamping each frame or the last coordinate.
    p = torch.tensor([3.0, 4.0]).reshape(1, 1, 2, 1, 1)
    n = torch.tensor([0.0, 4.0]).reshape_as(p)
    expected = torch.tensor([67.5 / (97**0.5), 30 / (97**0.5)]).reshape_as(p)
    torch.testing.assert_close(pipeline.combine_cfg_noise(p, n, 2, 1.5), expected)


def test_truncation_skips_negative_and_keeps_frame_axis(monkeypatch):
    pipeline, events = _pipeline(monkeypatch)
    pipeline._interrupt = False
    direct = torch.tensor([[[12, 19, 27]]], dtype=torch.float32).expand(1, 4, 3)
    with override_forward_context(ForwardContext(direct_condition=direct)):
        result = pipeline.diffuse(
            [torch.ones(256, 2)],
            [torch.zeros(256, 2)],
            torch.zeros(1, 4, 1, 4, 6),
            torch.tensor([500.0]),
            True,
            2,
            cfg_truncation=0.4,
        )
    assert result.shape == (1, 4, 1, 4, 6)
    assert torch.equal(result, torch.full_like(result, -2))
    assert events == ["marker", "positive"]


def test_pipeline_rejects_missing_and_negative_conditions(monkeypatch):
    pipeline, _ = _pipeline(monkeypatch)
    request = _request(negative="bad")
    with pytest.raises(ValueError, match="negative_prompt"):
        pipeline.forward(request)
    request = _request()
    request.requests[0].prompt["extra"].pop("direct_hidden_states")
    with pytest.raises(ValueError, match="require query and direct"):
        pipeline.forward(request)


def test_reference_encode_error_cleans_frame_state(monkeypatch):
    pipeline, _ = _pipeline(monkeypatch, layered=True)
    request = _request(layered=True)
    request.requests[0].prompt["extra"]["reference_image"] = torch.ones(1, 3, 32, 48)
    with pytest.raises(ValueError, match="batched RGBA"):
        pipeline.forward(request)
    assert pipeline._num_frames_per_prompt == 1
    assert pipeline._pending_prompt_embeds is pipeline._pending_negative_prompt_embeds is None


@pytest.mark.parametrize("rank", [0, 1])
def test_cfg_rank_dispatch_uses_branch_local_conditions(monkeypatch, rank):
    pipeline, events = _pipeline(monkeypatch, layered=True)
    monkeypatch.setattr(cfg_parallel, "is_cfg_group_initialized", lambda: True)
    monkeypatch.setattr(cfg_parallel, "get_classifier_free_guidance_world_size", lambda: 2)
    monkeypatch.setattr(cfg_parallel, "get_classifier_free_guidance_rank", lambda: rank)

    def gather(local, separate_tensors):
        assert separate_tensors
        assert torch.equal(local, torch.full((1, 4, 4, 4, 6), 2.0 if rank == 0 else 1.0))
        return [torch.full_like(local, 2), torch.full_like(local, 1)]

    monkeypatch.setattr(cfg_parallel, "get_cfg_group", lambda: SimpleNamespace(all_gather=gather))
    context = ForwardContext(direct_condition=torch.tensor([71]), ref_latent=torch.tensor([72]))
    old_direct, old_ref = context.direct_condition, context.ref_latent
    with override_forward_context(context):
        pipeline.forward(_request(layered=True))
    assert context.direct_condition is old_direct and context.ref_latent is old_ref
    assert events == ["marker", "positive" if rank == 0 else "negative"]
    assert torch.equal(pipeline.observed_scheduler.noise, torch.full((1, 4, 4, 4, 6), -4.0))


def test_transformer_rejects_fused_condition_on_single_branch(monkeypatch):
    pipeline, _ = _pipeline(monkeypatch)
    context = ForwardContext(direct_condition=torch.zeros(2, 4, 3))
    with override_forward_context(context), pytest.raises(ValueError, match="direct conditions must match"):
        pipeline.transformer([torch.zeros(4, 1, 4, 6)], torch.tensor([0.5]), [torch.ones(256, 2)])


def test_duplicate_request_id_guard(monkeypatch):
    pipeline, _ = _pipeline(monkeypatch)
    request = _request()
    request.requests.append(request.requests[0])
    with pytest.raises(ValueError, match="unique"):
        pipeline.forward(request)

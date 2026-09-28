# SPDX-License-Identifier: Apache-2.0
"""Helios request- and step-batching contract tests."""

from __future__ import annotations

from types import MethodType, SimpleNamespace

import torch

from vllm_omni.diffusion.models.helios.pipeline_helios import (
    HeliosPipeline,
    get_helios_pre_process_func,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform


class _CountingTransformer:
    dtype = torch.float32

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *, hidden_states: torch.Tensor, **kwargs):
        del kwargs
        self.calls += 1
        return (hidden_states + 1.0,)


class _TinyScheduler:
    def __init__(self) -> None:
        self.timesteps = torch.tensor([1.0])

    def set_timesteps(self, *args, **kwargs) -> None:
        del args, kwargs
        self.timesteps = torch.tensor([1.0])


def _sampling(**overrides) -> OmniDiffusionSamplingParams:
    params = OmniDiffusionSamplingParams(
        height=32,
        width=32,
        num_frames=1,
        num_inference_steps=1,
        guidance_scale=1.0,
        output_type="latent",
    )
    for name, value in overrides.items():
        setattr(params, name, value)
    return params


def _step_state(request_id: str, value: float) -> StepRequestState:
    state = StepRequestState(
        request_id=request_id,
        sampling=_sampling(seed=int(value)),
        prompt=f"prompt-{request_id}",
        prompt_embeds=torch.full((1, 2, 3), value),
        latents=torch.full((1, 1, 1, 2, 2), value),
        timesteps=torch.tensor([1.0]),
    )
    state.extra.update(
        {
            "batch_size": 1,
            "dtype": torch.float32,
            "attention_kwargs": {},
            "indices_hidden_states": torch.zeros(1, 2, dtype=torch.long),
            "indices_latents_history_short": torch.zeros(1, 2, dtype=torch.long),
            "indices_latents_history_mid": torch.zeros(1, 2, dtype=torch.long),
            "indices_latents_history_long": torch.zeros(1, 2, dtype=torch.long),
            "latents_history_short": torch.zeros(1, 1, 1, 2, 2),
            "latents_history_mid": torch.zeros(1, 1, 1, 2, 2),
            "latents_history_long": torch.zeros(1, 1, 1, 2, 2),
            "is_enable_stage2": False,
            "use_cfg_zero_star": False,
            "guidance_scale": 1.0,
        }
    )
    return state


def _step_pipeline() -> HeliosPipeline:
    pipeline = object.__new__(HeliosPipeline)
    pipeline.device = torch.device("cpu")
    pipeline.transformer = _CountingTransformer()
    pipeline._current_timestep = None
    pipeline._guidance_scale = 1.0
    return pipeline


def test_helios_step_batch_runs_one_transformer_call_and_preserves_rows() -> None:
    pipeline = _step_pipeline()
    states = [_step_state("request-a", 2.0), _step_state("request-b", 7.0)]

    batch = InputBatch.make_batch(states, idx_mapping=torch.tensor([1, 0]))
    selected_states = [states[1], states[0]]
    noise_pred = pipeline.denoise_step(batch, states=selected_states)

    assert pipeline.transformer.calls == 1
    assert noise_pred.shape == (2, 1, 1, 2, 2)
    assert torch.allclose(noise_pred[0], torch.full_like(noise_pred[0], 8.0))
    assert torch.allclose(noise_pred[1], torch.full_like(noise_pred[1], 3.0))
    assert batch.request_ids == ["request-b", "request-a"]


def test_helios_step_single_request_regression_uses_same_path() -> None:
    pipeline = _step_pipeline()
    state = _step_state("single", 4.0)
    batch = InputBatch.make_batch([state])

    noise_pred = pipeline.denoise_step(batch, states=[state])

    assert pipeline.transformer.calls == 1
    assert torch.allclose(noise_pred, torch.full_like(noise_pred, 5.0))


def test_helios_step_request_churn_keeps_state_and_output_identity() -> None:
    pipeline = _step_pipeline()
    request_a = _step_state("request-a", 2.0)
    request_b = _step_state("request-b", 7.0)
    request_c = _step_state("request-c", 11.0)

    first_batch = InputBatch.make_batch([request_a, request_b])
    first_pred = pipeline.denoise_step(first_batch, states=[request_a, request_b])

    request_c.timesteps = torch.tensor([0.5])
    second_batch = InputBatch.make_batch([request_b, request_c], cached_batch=first_batch)
    second_pred = pipeline.denoise_step(second_batch, states=[request_b, request_c])

    assert pipeline.transformer.calls == 2
    assert second_batch.request_ids == ["request-b", "request-c"]
    assert torch.allclose(first_pred[0], torch.full_like(first_pred[0], 3.0))
    assert torch.allclose(first_pred[1], torch.full_like(first_pred[1], 8.0))
    assert torch.allclose(second_pred[0], torch.full_like(second_pred[0], 8.0))
    assert torch.allclose(second_pred[1], torch.full_like(second_pred[1], 12.0))


def _request(request_id: str, *, extra_args: dict | None = None) -> OmniDiffusionRequest:
    return OmniDiffusionRequest(
        request_id=request_id,
        prompt=f"prompt-{request_id}",
        sampling_params=_sampling(seed=len(request_id), extra_args=extra_args or {}),
    )


def test_helios_preprocess_key_separates_structural_request_options() -> None:
    preprocess = get_helios_pre_process_func(SimpleNamespace())
    request_a = preprocess(_request("a", extra_args={"num_latent_frames_per_chunk": 9}))
    request_b = preprocess(_request("b", extra_args={"num_latent_frames_per_chunk": 5}))

    assert request_a.batch_compatibility_key != request_b.batch_compatibility_key


def _batch_pipeline() -> HeliosPipeline:
    pipeline = object.__new__(HeliosPipeline)
    pipeline.device = torch.device("cpu")
    pipeline._guidance_scale = None
    pipeline._current_timestep = None
    pipeline.is_distilled = False
    pipeline.vae_scale_factor_temporal = 1
    pipeline.vae_scale_factor_spatial = 1
    pipeline.transformer = SimpleNamespace(
        dtype=torch.float32,
        config=SimpleNamespace(in_channels=1, patch_size=(1, 1, 1)),
    )
    pipeline.vae = SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.float32,
        config=SimpleNamespace(latents_mean=[0.0], latents_std=[1.0], z_dim=1),
    )
    pipeline.scheduler = _TinyScheduler()

    def encode_prompt(self, prompt, **kwargs):
        del kwargs
        return torch.arange(len(prompt), dtype=torch.float32).view(len(prompt), 1, 1), None

    def prepare_latents(self, batch_size, *args, **kwargs):
        del args, kwargs
        return torch.zeros(batch_size, 1, 1, 32, 32)

    def stage1_sample(self, latents, prompt_embeds, **kwargs):
        del kwargs
        return latents + prompt_embeds.view(prompt_embeds.shape[0], 1, 1, 1, 1)

    def decode(latents, **kwargs):
        del kwargs
        return (latents,)

    pipeline.encode_prompt = MethodType(encode_prompt, pipeline)
    pipeline.prepare_latents = MethodType(prepare_latents, pipeline)
    pipeline._stage1_sample = MethodType(stage1_sample, pipeline)
    pipeline.vae.decode = decode
    return pipeline


def test_helios_request_batch_forward_is_fused_and_keeps_output_order(monkeypatch) -> None:
    monkeypatch.setattr(current_omni_platform, "empty_cache", lambda: None)
    pipeline = _batch_pipeline()
    request_batch = DiffusionRequestBatch([_request("a"), _request("b")])

    result = pipeline.forward(request_batch, output_type="latent")

    assert [output.output.shape[0] for output in result] == [1, 1]
    assert torch.allclose(result[0].output, torch.zeros_like(result[0].output))
    assert torch.allclose(result[1].output, torch.ones_like(result[1].output))


def test_helios_request_batch_matches_single_request_output(monkeypatch) -> None:
    monkeypatch.setattr(current_omni_platform, "empty_cache", lambda: None)
    single = _batch_pipeline().forward(DiffusionRequestBatch([_request("a")]), output_type="latent")[0]
    batched = _batch_pipeline().forward(
        DiffusionRequestBatch([_request("a"), _request("b")]), output_type="latent"
    )[0]

    assert torch.allclose(single.output, batched.output)


def test_helios_declares_both_batch_capabilities() -> None:
    assert HeliosPipeline.supports_step_execution is True
    assert HeliosPipeline.supports_request_batch is True

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Diffusion pipeline for Ming-Image Design and Design-Layer checkpoints."""

from __future__ import annotations

import copy
import logging
import os
from collections.abc import Iterable, Sequence
from typing import Any, ClassVar

import torch
import torch.nn as nn
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from PIL import Image
from vllm.model_executor.models.utils import AutoWeightsLoader

from vllm_omni.diffusion.compile import regionally_compile
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_qwenimage import (
    DistributedAutoencoderKLQwenImage,
)
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.forward_context import (
    get_forward_context,
    is_forward_context_available,
    set_forward_context_direct_condition,
    set_forward_context_ref_latent,
)
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import from_pretrained_with_prefetch
from vllm_omni.diffusion.models.ming_image.condition import MingImageConditioning
from vllm_omni.diffusion.models.ming_image.transformer import MingImageTransformer2DModel
from vllm_omni.diffusion.models.z_image.pipeline_z_image import ZImagePipeline, calculate_shift, retrieve_timesteps
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.utils.hf_utils import get_diffusion_model_index
from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

logger = logging.getLogger(__name__)

_DESIGN_PIPELINE = "MingImageDiffusionPipeline"
_LAYERED_PIPELINE = "MingImageLayeredDiffusionPipeline"
_VENDOR_TRANSFORMER_CLASS = "DiffusionTransformer"
_DIFFUSION_REQUIRED_PATTERNS = [
    "model_index.json",
    "scheduler/**",
    "transformer/**",
    "vae/**",
    "mlp/**",
    "connector/**",
]


def _validate_variant_config(
    model_index: dict[str, Any] | None,
    transformer_config: Any,
) -> bool:
    """Validate checkpoint-owned variant metadata and return layered mode."""
    declared_transformer = getattr(transformer_config, "_class_name", None)
    if declared_transformer != _VENDOR_TRANSFORMER_CLASS:
        raise ValueError(
            "Ming-Image transformer/config.json must preserve the vendor "
            f"_class_name={_VENDOR_TRANSFORMER_CLASS!r}, got {declared_transformer!r}. "
            "vLLM-Omni loads those weights through MingImageTransformer2DModel."
        )

    alignment_padding_mode = getattr(transformer_config, "alignment_padding_mode", None)
    multi_frame_output = getattr(transformer_config, "multi_frame_output", None)
    variant = (alignment_padding_mode, multi_frame_output)
    if variant == ("zero_masked", False):
        is_layer_decomposition = False
    elif variant == ("learned", True):
        is_layer_decomposition = True
    else:
        raise ValueError(
            "Ming-Image transformer/config.json must use either "
            "alignment_padding_mode='zero_masked' with multi_frame_output=False "
            "or alignment_padding_mode='learned' with multi_frame_output=True; "
            f"got alignment_padding_mode={alignment_padding_mode!r}, "
            f"multi_frame_output={multi_frame_output!r}."
        )

    declared_pipeline = (model_index or {}).get("_class_name")
    if declared_pipeline is not None:
        if declared_pipeline not in {_DESIGN_PIPELINE, _LAYERED_PIPELINE}:
            raise ValueError(
                "Ming-Image model_index.json must declare "
                f"{_DESIGN_PIPELINE!r} or {_LAYERED_PIPELINE!r}, got {declared_pipeline!r}."
            )
        expected_pipeline = _LAYERED_PIPELINE if is_layer_decomposition else _DESIGN_PIPELINE
        if declared_pipeline != expected_pipeline:
            raise ValueError(
                f"Ming-Image model_index.json declares {declared_pipeline!r}, but "
                f"transformer/config.json describes {expected_pipeline!r}."
            )
    return is_layer_decomposition


class MingImageDiffusionPipeline(ZImagePipeline):
    """Ming-Image component adapter around the canonical Z-Image loop."""

    supports_request_batch = True
    supports_step_execution = True

    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["conditioning"]
    _vae_modules: ClassVar[list[str]] = ["vae"]

    @property
    def do_classifier_free_guidance(self) -> bool:
        # The vendor Ming-Image pipeline only enables CFG above 1.0
        return self._guidance_scale > 1

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        del prefix
        nn.Module.__init__(self)

        model_path = od_config.model
        if not os.path.exists(model_path):
            model_path = download_weights_from_hf_specific(
                model_name_or_path=model_path,
                cache_dir=None,
                allow_patterns=_DIFFUSION_REQUIRED_PATTERNS,
                revision=od_config.revision,
                require_all=True,
            )
        local_files_only = os.path.isdir(model_path)
        dtype = od_config.dtype

        self.od_config = od_config
        self._execution_device = get_local_device()
        self.device = self._execution_device
        model_index = get_diffusion_model_index(model_path, revision=od_config.revision) or {}
        transformer_config = od_config.tf_model_config
        self.is_layer_decomposition = _validate_variant_config(
            model_index,
            transformer_config,
        )

        self.default_num_inference_steps = 12
        self.default_guidance_scale = 2.0 if self.is_layer_decomposition else 1.0
        self._num_frames_per_prompt = 1
        self._pending_prompt_embeds: list[torch.Tensor] | None = None
        self._pending_negative_prompt_embeds: list[torch.Tensor] | None = None
        self._uses_cudagraph_trees = False
        self._interrupt = False

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model_path,
                subfolder="transformer",
                revision=od_config.revision,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]
        subfolders = ["scheduler", "transformer", "vae"]

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_path,
            subfolder="scheduler",
            local_files_only=local_files_only,
        )

        self.vae = from_pretrained_with_prefetch(
            DistributedAutoencoderKLQwenImage.from_pretrained,
            model_path,
            subfolder="vae",
            prefetch_list=subfolders,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        ).to(self.device)
        self.vae.eval()
        vae_channels = int(getattr(self.vae.config, "input_channels", 3))
        if vae_channels != 4:
            raise ValueError(f"Ming-Image requires an RGBA VAE, got input_channels={vae_channels}.")

        transformer_kwargs = get_transformer_config_kwargs(
            od_config.tf_model_config,
            MingImageTransformer2DModel,
        )
        self.transformer = MingImageTransformer2DModel(
            quant_config=od_config.quantization_config,
            **transformer_kwargs,
        )
        self.conditioning = MingImageConditioning(
            model_path,
            device=self.device,
            dtype=dtype,
        )
        self.text_encoder = None
        self.tokenizer = None

        self.vae_scale_factor = 2 ** len(self.vae.config.temperal_downsample)
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2,
            do_convert_rgb=False,
            resample="bilinear",
        )
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=od_config.enable_diffusion_pipeline_profiler
        )

    def setup_compile(self) -> None:
        # Keep request preparation, scheduling, and VAE work eager,
        # while capturing repeated DiT blocks for CUDAGraph Trees replay.
        if self.od_config.diffusion_compile_granularity != "regional":
            logger.warning(
                "Ming-Image CUDA Graph uses regional DiT compilation; diffusion_compile_granularity=%r is ignored.",
                self.od_config.diffusion_compile_granularity,
            )
        self.transformer = regionally_compile(
            self.transformer,
            mode="reduce-overhead",
            fullgraph=True,
            dynamic=self.od_config.diffusion_compile_dynamic,
        )
        self._uses_cudagraph_trees = True

    def encode_prompt(self, *args, **kwargs):
        del args, kwargs
        if self._pending_prompt_embeds is None or self._pending_negative_prompt_embeds is None:
            raise RuntimeError("Ming-Image conditioning is only available during forward.")
        return self._pending_prompt_embeds, self._pending_negative_prompt_embeds

    def predict_noise_maybe_with_cfg(
        self,
        do_true_cfg: bool,
        true_cfg_scale: float,
        positive_kwargs: dict[str, Any],
        negative_kwargs: dict[str, Any] | None,
        cfg_normalize: bool | float = True,
        output_slice: int | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Attach projected conditions to each branch before shared dispatch.

        The mixin can execute either branch locally or on a dedicated CFG rank.
        Negative direct conditioning is zero after the biased projection; both
        branches use the same reference. Never expose a fused 2B context to a
        transformer invocation with only B samples.
        """
        context = get_forward_context()
        direct = context.direct_condition
        if direct is None:
            raise RuntimeError("Ming-Image denoising requires projected direct conditioning.")
        positive_kwargs = dict(positive_kwargs, direct_condition=direct, ref_latent=context.ref_latent)
        if negative_kwargs is not None:
            negative_kwargs = dict(
                negative_kwargs, direct_condition=torch.zeros_like(direct), ref_latent=context.ref_latent
            )
        return super().predict_noise_maybe_with_cfg(
            do_true_cfg,
            true_cfg_scale,
            positive_kwargs,
            negative_kwargs,
            cfg_normalize=cfg_normalize,
            output_slice=output_slice,
            kwargs=kwargs,
        )

    def predict_noise(self, *args: Any, direct_condition: torch.Tensor, ref_latent: torch.Tensor | None, **kwargs: Any):
        context = get_forward_context()
        previous_direct, previous_ref = context.direct_condition, context.ref_latent
        set_forward_context_direct_condition(direct_condition)
        set_forward_context_ref_latent(ref_latent)
        try:
            return super().predict_noise(*args, **kwargs)
        finally:
            set_forward_context_direct_condition(previous_direct)
            set_forward_context_ref_latent(previous_ref)

    def prepare_latents(self, batch_size, *args, **kwargs):
        frames = self._num_frames_per_prompt
        flat = super().prepare_latents(batch_size * frames, *args, **kwargs)
        return torch.stack(flat.chunk(frames, dim=0), dim=2)

    def _encode_reference(
        self,
        reference: Any | None,
        height: int,
        width: int,
    ) -> torch.Tensor | None:
        if reference is None:
            return None
        if isinstance(reference, list):
            if len(reference) != 1:
                raise ValueError("Ming-Image currently accepts exactly one reference image.")
            reference = reference[0]
        if isinstance(reference, Image.Image):
            reference = reference.convert("RGBA")
        if not isinstance(reference, torch.Tensor):
            reference = self.image_processor.preprocess(reference, height=height, width=width)
        if reference.ndim != 4 or reference.shape[:2] != (1, 4):
            raise ValueError("Ming-Image reference must be a batched RGBA tensor or an RGBA-compatible image.")
        reference = reference.to(device=self.device, dtype=self.vae.dtype).unsqueeze(2)
        latent = self.vae.encode(reference).latent_dist.mode()
        return (latent - self.vae.config.shift_factor) * self.vae.config.scaling_factor

    @staticmethod
    def _get_prompt_extra(req: DiffusionRequestBatch) -> dict[str, Any]:
        if not req.prompts:
            return {}
        prompt = req.prompts[0]
        if isinstance(prompt, dict):
            return dict(prompt.get("extra") or {})
        if hasattr(prompt, "_asdict"):
            return dict(prompt._asdict().get("extra") or {})
        return {}

    def _configure_output_frames(
        self,
        *,
        reference: Any | None,
        num_layers: int,
        is_dummy_run: bool,
    ) -> None:
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")
        if self.is_layer_decomposition:
            if reference is None and not is_dummy_run:
                raise ValueError("Ming-Image Design-Layer requires a reference image.")
            self._num_frames_per_prompt = num_layers + 1
            return
        if num_layers != 1:
            raise ValueError("Ming-Image Design supports exactly one output frame.")
        self._num_frames_per_prompt = 1

    @torch.inference_mode()
    def forward(self, req: DiffusionRequestBatch) -> list[DiffusionOutput]:
        # The runner owns the outer context. Restore its values on every exit,
        # including failures during reference encoding before denoising starts.
        context = get_forward_context() if is_forward_context_available() else None
        previous_direct = context.direct_condition if context is not None else None
        previous_ref = context.ref_latent if context is not None else None
        try:
            return self._forward(req)
        finally:
            set_forward_context_ref_latent(previous_ref)
            set_forward_context_direct_condition(previous_direct)
            self._pending_prompt_embeds = None
            self._pending_negative_prompt_embeds = None
            self._num_frames_per_prompt = 1

    def _prepare_state(self, state: StepRequestState) -> StepRequestState:
        sampling = state.sampling
        if sampling.num_outputs_per_prompt != 1:
            raise ValueError(
                f"Ming-Image currently supports num_outputs_per_prompt=1 only, got {sampling.num_outputs_per_prompt}."
            )

        prompt = state.prompt
        extra = dict(prompt.get("extra") or {}) if isinstance(prompt, dict) else {}
        extra_args = sampling.extra_args or {}
        negative_prompt = extra_args.get("negative_prompt")
        if isinstance(prompt, dict):
            negative_prompt = prompt.get("negative_prompt", negative_prompt)
        if negative_prompt is not None and negative_prompt != "":
            raise ValueError("Ming-Image uses zero negative conditioning and does not accept negative_prompt.")

        is_dummy_run = OmniDiffusionRequest.is_dummy_run_request_id(state.request_id)
        query_hidden = extra.get("query_hidden_states")
        direct_hidden = extra.get("direct_hidden_states")
        if query_hidden is None or direct_hidden is None:
            if not is_dummy_run:
                raise ValueError("Ming-Image requests require query and direct conditions.")
            logger.warning("Ming-Image conditions are absent during warmup; using zero tensors.")
            query_hidden = torch.zeros((256, 2048), device=self.device, dtype=self.od_config.dtype)
            direct_hidden = torch.zeros((1, 6144), device=self.device, dtype=self.od_config.dtype)
        if not isinstance(query_hidden, torch.Tensor) or not isinstance(direct_hidden, torch.Tensor):
            raise TypeError("Ming-Image query and direct conditions must be tensors.")
        if query_hidden.ndim == 2:
            query_hidden = query_hidden.unsqueeze(0)
        if direct_hidden.ndim == 2:
            direct_hidden = direct_hidden.unsqueeze(0)
        if query_hidden.ndim != 3 or query_hidden.shape[0] != 1 or query_hidden.shape[1:] != (256, 2048):
            raise ValueError("Ming-Image query conditions must have shape [1, 256, 2048].")
        if (
            direct_hidden.ndim != 3
            or direct_hidden.shape[0] != 1
            or direct_hidden.shape[1] < 1
            or direct_hidden.shape[-1] != 6144
        ):
            raise ValueError("Ming-Image direct conditions must have shape [1, prefix_length, 6144].")
        query_hidden = query_hidden.to(device=self.device, dtype=self.od_config.dtype)
        direct_hidden = direct_hidden.to(device=self.device, dtype=self.od_config.dtype)
        cap_feats, direct_condition = self.conditioning(query_hidden, direct_hidden)

        reference = extra.get("reference_image")
        num_layers = int(extra_args.get("num_layers", extra.get("num_layers", 1)))
        self._configure_output_frames(
            reference=reference,
            num_layers=num_layers,
            is_dummy_run=is_dummy_run,
        )

        height_value = extra_args.get("height", sampling.height)
        width_value = extra_args.get("width", sampling.width)
        height = int(1024 if height_value is None else height_value)
        width = int(1024 if width_value is None else width_value)
        steps = int(
            self.default_num_inference_steps if sampling.num_inference_steps is None else sampling.num_inference_steps
        )
        cfg = float(sampling.guidance_scale if sampling.guidance_scale_provided else self.default_guidance_scale)
        seed = extra_args.get("seed", sampling.seed)
        generator = torch.Generator(device="cpu").manual_seed(int(seed)) if seed is not None else sampling.generator
        if generator is None:
            generator = torch.Generator(device="cpu")
            generator.seed()
        if isinstance(generator, list) and len(generator) != 1:
            raise ValueError("Ming-Image requires one generator per request.")
        sampling.generator = generator
        scale = self.vae_scale_factor * 2
        if height <= 0 or width <= 0 or height % scale or width % scale:
            raise ValueError(f"Ming-Image height and width must be positive multiples of {scale}.")
        if steps <= 0:
            raise ValueError("Ming-Image requires positive num_inference_steps.")
        frames = self._num_frames_per_prompt
        ref_latent = self._encode_reference(reference, height, width)
        channels = self.transformer.in_channels
        shape = (1, channels, frames, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if sampling.latents is not None:
            if tuple(sampling.latents.shape) != shape:
                raise ValueError(
                    f"Unexpected Ming-Image latents shape {tuple(sampling.latents.shape)}; expected {shape}."
                )
            state.latents = sampling.latents.to(device=self.device, dtype=torch.float32).clone()
        else:
            # Preserve the foundation's frame-major RNG draw within ONE request.
            state.latents = self.prepare_latents(
                1,
                channels,
                height,
                width,
                torch.float32,
                self.device,
                generator * frames if isinstance(generator, list) else generator,
            )
        if ref_latent is not None and tuple(ref_latent.shape) != (1, channels, 1, shape[-2], shape[-1]):
            raise ValueError("Ming-Image reference latent must match the request's channels and spatial geometry.")
        state.scheduler = copy.deepcopy(self.scheduler)
        mu = calculate_shift(
            (shape[-2] // 2) * (shape[-1] // 2),
            state.scheduler.config.get("base_image_seq_len", 256),
            state.scheduler.config.get("max_image_seq_len", 4096),
            state.scheduler.config.get("base_shift", 0.5),
            state.scheduler.config.get("max_shift", 1.15),
        )
        state.scheduler.sigma_min = 0.0
        state.timesteps, _ = retrieve_timesteps(
            state.scheduler,
            steps,
            self.device,
            timesteps=sampling.timesteps,
            sigmas=sampling.sigmas,
            mu=mu,
        )
        state.step_index = int(sampling.step_index or 0)
        if state.total_steps < 1 or not 0 <= state.step_index < state.total_steps:
            raise ValueError("Ming-Image requires a nonempty schedule and valid initial step_index.")
        if state.step_index and sampling.latents is None:
            raise ValueError("Ming-Image resumed step_index requires request-owned latents.")
        if hasattr(state.scheduler, "set_begin_index"):
            state.scheduler.set_begin_index(state.step_index)
        state.prompt_embeds = cap_feats
        state.negative_prompt_embeds = torch.zeros_like(cap_feats)
        state.do_true_cfg = cfg > 1
        state.extra.update(
            direct_condition=direct_condition,
            ref_latent=ref_latent,
            frames=frames,
            cfg=cfg,
            cfg_normalize=sampling.cfg_normalize,
            cfg_truncation=extra_args.get("cfg_truncation", 1.0),
        )
        return state

    @torch.inference_mode()
    def prepare_encode(self, state: StepRequestState, **kwargs: Any) -> StepRequestState:
        del kwargs
        # Clone RNG/latents even if callers share sampling objects. No pipeline
        # condition or scheduler cursor persists between admitted requests.
        state.sampling = copy.deepcopy(state.sampling)
        try:
            return self._prepare_state(state)
        finally:
            self._num_frames_per_prompt = 1
            self._pending_prompt_embeds = self._pending_negative_prompt_embeds = None

    @staticmethod
    def _effective_cfg(state: StepRequestState) -> bool:
        truncation = state.extra["cfg_truncation"]
        t_norm = (1000 - float(state.current_timestep)) / 1000
        return state.do_true_cfg and not (
            truncation is not None and float(truncation) <= 1 and t_norm > float(truncation)
        )

    def _denoise_states(self, states: Sequence[StepRequestState], latents: torch.Tensor) -> torch.Tensor:
        if not states:
            raise ValueError("Ming-Image cannot denoise an empty wave.")
        if len(states) != latents.shape[0] or any(state.latents.shape[0] != 1 for state in states):
            raise ValueError("Ming-Image denoise rows must match request ownership.")
        # Unequal progress can change actual CFG control flow in a homogeneous
        # scheduler batch. Group locally, then scatter predictions to caller order.
        groups: dict[tuple, list[int]] = {}
        for index, state in enumerate(states):
            if state.current_timestep is None:
                raise ValueError("Ming-Image cannot denoise a completed request.")
            ref = state.extra["ref_latent"]
            key = (
                self._effective_cfg(state),
                state.extra["cfg"],
                state.extra["cfg_normalize"],
                tuple(state.extra["direct_condition"].shape[1:]),
                None if ref is None else tuple(ref.shape[1:]),
            )
            groups.setdefault(key, []).append(index)
        result = torch.empty_like(latents, dtype=torch.float32)
        context = get_forward_context()
        previous_direct, previous_ref = context.direct_condition, context.ref_latent
        try:
            for (apply_cfg, cfg, normalize, _, ref_shape), indices in groups.items():
                selected = [states[index] for index in indices]
                direct = torch.cat([state.extra["direct_condition"] for state in selected], dim=0)
                ref = torch.cat([state.extra["ref_latent"] for state in selected], dim=0) if ref_shape else None
                set_forward_context_direct_condition(direct)
                set_forward_context_ref_latent(ref)
                x = list(latents[indices].to(self.od_config.dtype).unbind(0))
                t = torch.stack([state.current_timestep for state in selected]).to(latents.device, torch.float32)
                t = (1000 - t) / 1000
                positive = {"x": x, "t": t, "cap_feats": [state.prompt_embeds[0] for state in selected]}
                negative = (
                    {"x": x, "t": t, "cap_feats": [state.negative_prompt_embeds[0] for state in selected]}
                    if apply_cfg
                    else None
                )
                prediction = self.predict_noise_maybe_with_cfg(
                    apply_cfg,
                    cfg,
                    positive,
                    negative,
                    cfg_normalize=normalize,
                )
                result[indices] = -prediction.float()
        finally:
            set_forward_context_direct_condition(previous_direct)
            set_forward_context_ref_latent(previous_ref)
        return result

    @torch.inference_mode()
    def denoise_step(
        self,
        input_batch: InputBatch,
        *,
        states: Sequence[StepRequestState] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | None:
        del kwargs
        if self.interrupt:
            return None
        # InputBatch.states follows its idx_mapping; an optional runner state
        # sequence is not a substitute for that gather order.
        return self._denoise_states(input_batch.states, input_batch.latents)

    @torch.inference_mode()
    def step_scheduler(self, state: StepRequestState, noise_pred: torch.Tensor, **kwargs: Any) -> None:
        del kwargs
        if noise_pred.shape != state.latents.shape:
            raise ValueError("Ming-Image scheduler prediction must match the request latent envelope.")
        state.latents = self.scheduler_step_maybe_with_cfg(
            noise_pred.float(),
            state.current_timestep,
            state.latents,
            self._effective_cfg(state),
            per_request_scheduler=state.scheduler,
        )
        state.step_index += 1

    @torch.inference_mode()
    def post_decode(self, state: StepRequestState, **kwargs: Any) -> DiffusionOutput:
        del kwargs
        if not state.denoise_completed:
            raise ValueError("Ming-Image cannot decode unfinished request latents.")
        return DiffusionOutput(output=self._decode_latent_frames(state.latents))

    def _forward(self, req: DiffusionRequestBatch) -> list[DiffusionOutput]:
        if not req.requests:
            raise ValueError("Ming-Image cannot execute an empty request batch.")
        if len(set(req.request_ids)) != req.num_reqs:
            raise ValueError("Ming-Image request IDs must be unique within a batch.")
        self._interrupt = False
        states = [
            self.prepare_encode(StepRequestState(r.request_id, r.sampling_params, r.prompt)) for r in req.requests
        ]
        while active := [state for state in states if not state.denoise_completed]:
            # Request-mode callers normally arrive pre-grouped by the scheduler.
            # Keep independent geometry/frame envelopes even for direct callers.
            groups: dict[tuple, list[StepRequestState]] = {}
            for state in active:
                groups.setdefault(tuple(state.latents.shape[1:]), []).append(state)
            for selected in groups.values():
                predictions = self._denoise_states(selected, torch.cat([state.latents for state in selected]))
                for state, noise in zip(selected, predictions.split(1), strict=True):
                    self.step_scheduler(state, noise)
        outputs = [self.post_decode(state) for state in states]
        return outputs

    @staticmethod
    def _flatten_latent_frames(latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim == 4:
            return latents
        if latents.ndim != 5:
            raise ValueError(f"Expected 4D or 5D Ming-Image latents, got {tuple(latents.shape)}")
        batch, channels, frames, height, width = latents.shape
        return latents.permute(2, 0, 1, 3, 4).reshape(
            frames * batch,
            channels,
            height,
            width,
        )

    def _decode_latent_frames(self, latents: torch.Tensor) -> torch.Tensor:
        latents = self._flatten_latent_frames(latents).to(self.vae.dtype).unsqueeze(2)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        if image.ndim == 5:
            if image.shape[2] != 1:
                raise ValueError(
                    f"Ming-Image VAE returned multiple decoded frames for a single latent frame: {tuple(image.shape)}"
                )
            image = image.squeeze(2)
        return image

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = AutoWeightsLoader(self).load_weights(weights)
        loaded |= {f"vae.{name}" for name, _ in self.vae.named_parameters()}
        loaded |= {f"conditioning.{name}" for name, _ in self.conditioning.named_parameters()}
        return loaded


def get_ming_image_pre_process_func(od_config: OmniDiffusionConfig):
    """Resolve real Ming defaults before the shared scheduler builds its key.

    Steps/seeds/timesteps/truncation are private to each request. Shape and
    projected direct/reference structure must agree in a gathered InputBatch.
    """
    layered = bool(getattr(od_config.tf_model_config, "multi_frame_output", False))

    def pre_process(request: OmniDiffusionRequest):
        sampling = request.sampling_params
        extra_args = sampling.extra_args or {}
        extra = request.prompt.get("extra", {}) if isinstance(request.prompt, dict) else {}
        height_value = extra_args.get("height", sampling.height)
        width_value = extra_args.get("width", sampling.width)
        sampling.height = int(1024 if height_value is None else height_value)
        sampling.width = int(1024 if width_value is None else width_value)
        if sampling.num_inference_steps is None:
            sampling.num_inference_steps = 12
        if not sampling.guidance_scale_provided:
            sampling.guidance_scale = 2.0 if layered else 1.0
        sampling.do_classifier_free_guidance = sampling.guidance_scale > 1
        sampling.num_frames = int(extra_args.get("num_layers", extra.get("num_layers", 1))) + 1 if layered else 1
        reference = extra.get("reference_image")
        direct = extra.get("direct_hidden_states")
        direct_shape = tuple(direct.shape[-2:]) if isinstance(direct, torch.Tensor) else None
        request.batch_compatibility_key = (
            "ming_image",
            layered,
            sampling.num_frames,
            # Request-owned latents carry the spatial geometry, so a gathered
            # InputBatch can only hold one resolution at a time.
            sampling.height,
            sampling.width,
            direct_shape,
            reference is not None,
        )
        request.use_step_execution = bool(getattr(od_config, "step_execution", False))
        return request

    return pre_process


def get_ming_image_post_process_func(od_config: OmniDiffusionConfig):
    del od_config
    image_processor = VaeImageProcessor(vae_scale_factor=16, do_convert_rgb=False)

    def post_process(images: torch.Tensor):
        if images.ndim == 5:
            images = images.permute(0, 2, 1, 3, 4).flatten(0, 1)
        return image_processor.postprocess(images.float())

    return post_process


__all__ = ["MingImageDiffusionPipeline", "get_ming_image_pre_process_func", "get_ming_image_post_process_func"]

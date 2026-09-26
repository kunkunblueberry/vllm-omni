# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.

"""Ming-flash-omni-2.0 imagegen (text-to-image / img2img) diffusion pipeline.

Cross-stage data flow:

    Stage 0 (thinker, llm)           Stage 1 (imagegen, diffusion)
    ────────────────────             ──────────────────────────────
    forward returns                  thinker2imagegen hook slices
    multimodal_output[               final_hidden_states at
      "final_hidden_states"]         <imagePatch> positions,
       ↓                             returns list[dict] with
    shared_memory_connector          {"extra": {"thinker_hidden_states"}}
       ↓                             ──── via OmniMsgpackEncoder ────>
                                     MingImagePipeline.forward(req):
                                       hidden = req.prompts[0]["extra"][...]
                                       cond = condition_encoder(hidden)
                                       img = ZImagePipeline-style loop
                                       return DiffusionOutput(output=img)
"""

from __future__ import annotations

import logging
import os
from copy import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl import DistributedAutoencoderKL
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.forward_context import set_forward_context_ref_latent
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import prefetch_subfolders
from vllm_omni.diffusion.models.ming_flash_omni.byte5_encoder import (
    MingByT5Encoder,
)
from vllm_omni.diffusion.models.ming_flash_omni.condition_encoder import (
    MingConditionEncoder,
)
from vllm_omni.diffusion.models.ming_flash_omni.ming_zimage_transformer import (
    MingZImageTransformer2DModel,
)
from vllm_omni.diffusion.models.z_image.pipeline_z_image import ZImagePipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import (
    DiffusionRequestBatch,
    split_diffusion_output_by_request,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.model_loader.weight_utils import (
    download_weights_from_hf_specific,
)
from vllm_omni.transformers_utils.configs.ming_flash_omni import MingImageGenConfig

logger = logging.getLogger(__name__)


class MingImagePipeline(ZImagePipeline):
    """Ming-flash-omni-2.0 text-to-image diffusion pipeline.

    Ming-specific components added on top of the inherited contract:
      * ``condition_encoder`` — Qwen2 connector + proj_in/out + F.normalize×1000
      * ``byte5``             — Optional ByT5 glyph encoder (loaded if checkpoint
                                ships ``byt5/``)
    """

    supports_request_batch = True

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",  # noqa: ARG002
    ) -> None:
        # Skip ZImagePipeline.__init__ (it would eagerly load the Z-Image text
        # encoder/tokenizer that Ming replaces with its own condition_encoder).
        nn.Module.__init__(self)

        model_path = od_config.model
        if not os.path.exists(model_path):
            model_path = download_weights_from_hf_specific(model_path, od_config.revision, ["*"])

        dtype = getattr(od_config, "dtype", torch.bfloat16)
        local_files_only = os.path.exists(model_path)

        self.od_config = od_config
        self._execution_device = get_local_device()
        self.device = self._execution_device  # Ming convention alias
        self._dtype = dtype

        # Request-scoped conditioning handed to the inherited encode_prompt override
        self._pending_prompt_embeds: list[torch.Tensor] | None = None
        self._pending_negative_prompt_embeds: list[torch.Tensor] | None = None

        # Ming's per-checkpoint image-gen configuration. We cannot rely on
        # ``od_config.hf_config.image_gen_config`` because the diffusion
        # stage is started with ``hf_config_name: thinker_config`` (the
        # BailingMM2Config), which does not carry a MingImageGenConfig.
        # Fall back to defaults that match the released checkpoint.
        self.image_gen_config = MingImageGenConfig()
        logger.info(
            "[MingImagePipeline] init: model=%s dtype=%s image_gen_config=%s",
            model_path,
            dtype,
            self.image_gen_config,
        )

        # ----- weights_sources: DiT transformer + VAE.
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model_path,
                subfolder=self.image_gen_config.transformer_subfolder,
                revision=od_config.revision,
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model_path,
                subfolder=self.image_gen_config.vae_subfolder,
                revision=od_config.revision,
                prefix="vae.",
            ),
        ]

        prefetch_subfolders(
            model_path,
            [self.image_gen_config.scheduler_subfolder, self.image_gen_config.vae_subfolder],
            local_files_only=local_files_only,
        )

        # ----- Scheduler: load config-only from disk + Ming-specific override.
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_path,
            subfolder=self.image_gen_config.scheduler_subfolder,
            local_files_only=local_files_only,
        )
        # Ming forces use_dynamic_shifting=True at runtime regardless of what
        # the checkpoint scheduler_config.json ships.
        self.scheduler.config["use_dynamic_shifting"] = True
        logger.info(
            "[MingImagePipeline] scheduler: %s (use_dynamic_shifting=True)",
            type(self.scheduler).__name__,
        )

        # ----- VAE: DistributedAutoencoderKL.
        vae_config = DistributedAutoencoderKL.load_config(
            model_path, subfolder=self.image_gen_config.vae_subfolder, local_files_only=local_files_only
        )
        self.vae = DistributedAutoencoderKL.from_config(vae_config).to(self._execution_device, dtype=dtype)
        self.vae.eval()

        # ----- DiT transformer.
        self.transformer = MingZImageTransformer2DModel(quant_config=None)

        # Ming brings its own conditioning path — no Z-Image text_encoder /
        # tokenizer.
        self.text_encoder = None
        self.tokenizer = None

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2, do_convert_rgb=True)
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=getattr(od_config, "enable_diffusion_pipeline_profiler", False)
        )

        # ----- Condition encoder (Qwen2 connector + proj_in/out + norm×1000).
        self.condition_encoder = MingConditionEncoder(
            self.image_gen_config,
            thinker_hidden_size=self.image_gen_config.thinker_hidden_size,
            device=self.device,
            dtype=dtype,
        )
        self.condition_encoder.load_from_checkpoint(model_path)

        # Optional ByT5 glyph/text encoder. Only loaded when the checkpoint
        # ships a byt5/ subfolder; otherwise byte5_text requests are ignored
        byte5_dir = Path(model_path) / "byt5"
        if byte5_dir.exists():
            self.byte5 = MingByT5Encoder.from_checkpoint(byte5_dir, device=self.device, dtype=dtype)
        else:
            self.byte5 = None
            logger.info("[MingImagePipeline] no byt5/ subfolder at %s; ByT5 enhancement disabled", byte5_dir)

        logger.info("[MingImagePipeline] ready — vae_scale_factor=%d", self.vae_scale_factor)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_byte5_texts(extra: dict, sampling_params) -> list[str]:
        """Resolve byte5 glyph texts.

        Two sources, in order of priority:
        1. `extra["byte5_text"]`: auto-extracted from the user prompt's quoted spans
            by thinker2imagegen (already wrapped as `'Text "<glyph>". '`
            by Ming's get_text_from_prompt).
        2. `sampling_params.extra_args["byte5_text"]`:
            raw strings without the `Text "..."` wrapper are auto-wrapped here
            to match the distribution ByT5 was trained on.
        """
        # Source 1: auto-extracted, already wrapped. Return as-is if non-empty.
        raw = extra.get("byte5_text")
        if isinstance(raw, str):
            raw = [raw]
        if isinstance(raw, list):
            cleaned = [t for t in raw if isinstance(t, str) and t.strip()]
            if cleaned:
                return cleaned

        # Source 2: explicit override — wrap raw strings so the byte5 encoder
        # sees the same ``Text "<glyph>". `` format Ming used during training.
        raw = (getattr(sampling_params, "extra_args", None) or {}).get("byte5_text")
        if isinstance(raw, str):
            raw = [raw]
        if isinstance(raw, list):
            out: list[str] = []
            for t in raw:
                if not isinstance(t, str):
                    continue
                s = t.strip()
                if not s:
                    continue
                # Don't double-wrap if the caller already supplied ``Text "...". ``.
                out.append(s if s.startswith('Text "') else f'Text "{s}". ')
            if out:
                return out
        return []

    @torch.inference_mode()
    def _encode_reference_image(self, ref, height: int, width: int) -> torch.Tensor | None:
        """Turn a PIL/tensor reference image into a VAE latent for ``ref_x``.

        Applies the same shift/scale Ming uses (``(z - shift_factor) * scaling_factor``)
        so the concatenated frame lives in the DiT's latent space.
        """
        if ref is None:
            return None
        if not isinstance(ref, torch.Tensor):
            ref = self.image_processor.preprocess(ref, height, width)
        ref = ref.to(device=self.device, dtype=self.vae.dtype)
        latent = self.vae.encode(ref).latent_dist.mode()
        return (latent - self.vae.config.shift_factor) * self.vae.config.scaling_factor

    def encode_prompt(self, *args, **kwargs):  # noqa: ARG002
        """Return Ming's precomputed conditioning instead of encoding text.

        NOTE: Ming has no Z-Image text_encoder; its conditioning (cap_feats, optionally ByT5-augmented)
        is computed in forward and stashed on `self._pending_*` immediately before
        the `super().forward` call, so we simply hand it back here.
        """
        if self._pending_prompt_embeds is None:
            raise RuntimeError(
                "MingImagePipeline.encode_prompt called without pending "
                "conditioning; it must run within MingImagePipeline.forward."
            )

        return self._pending_prompt_embeds, self._pending_negative_prompt_embeds

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def forward(self, req: DiffusionRequestBatch) -> list[DiffusionOutput]:
        """Run a compatible wave of independent image-generation requests.

        Args:
            req: Independent requests in scheduler order. Each request's
                thinker hidden states must be present in its own prompt
                ``extra`` mapping.

        Returns:
            One ``DiffusionOutput`` per input request, in the same order.
        """
        if req.num_reqs == 0:
            return []
        target_device = next(self.parameters()).device
        target_dtype = next(self.parameters()).dtype
        cfg = self.image_gen_config

        def _prompt_extra(prompt: Any) -> dict[str, Any]:
            if isinstance(prompt, dict):
                return prompt.get("extra") or {}
            if prompt is not None and hasattr(prompt, "_asdict"):
                return (_prompt_extra(prompt._asdict()))
            if prompt is not None and hasattr(prompt, "__dict__"):
                return (_prompt_extra(vars(prompt)))
            return {}

        def _as_hidden(value: Any, request_id: str, field: str) -> torch.Tensor:
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Ming request {request_id!r} {field} must be a Tensor, got {type(value).__name__}")
            value = value.to(device=target_device, dtype=target_dtype)
            if value.dim() == 2:
                value = value.unsqueeze(0)
            if value.dim() != 3:
                raise ValueError(
                    f"Ming request {request_id!r} {field} must have shape [B,N,H], got {tuple(value.shape)}"
                )
            if value.shape[0] != 1:
                raise ValueError(
                    f"Ming request {request_id!r} {field} must contain one request, got {tuple(value.shape)}"
                )
            return value

        extras = [_prompt_extra(prompt) for prompt in req.prompts]
        hidden_items: list[torch.Tensor] = []
        negative_items: list[torch.Tensor | None] = []
        hidden_metadata: list[tuple[tuple[int, ...], torch.dtype, torch.device] | None] = []
        negative_metadata: list[tuple[tuple[int, ...], torch.dtype, torch.device] | None] = []
        for request, extra in zip(req.requests, extras, strict=True):
            hidden = extra.get("thinker_hidden_states")
            if hidden is None:
                hidden = (request.sampling_params.extra_args or {}).get("thinker_hidden_states")
            if hidden is None:
                scale = cfg.img_gen_scales[-1]
                hidden = torch.zeros(
                    (scale * scale, cfg.thinker_hidden_size), dtype=target_dtype, device=target_device
                )
                logger.warning(
                    "[MingImagePipeline.forward] request %s has no thinker hidden states; using zeros",
                    request.request_id,
                )
                hidden_metadata.append(None)
            else:
                if not isinstance(hidden, torch.Tensor):
                    raise TypeError(
                        f"Ming request {request.request_id!r} thinker_hidden_states must be a Tensor, "
                        f"got {type(hidden).__name__}"
                    )
                normalized_shape = tuple(hidden.shape) if hidden.dim() == 3 else (1, *tuple(hidden.shape))
                hidden_metadata.append((normalized_shape, hidden.dtype, hidden.device))
            hidden_items.append(_as_hidden(hidden, request.request_id, "thinker_hidden_states"))
            negative = extra.get("negative_thinker_hidden_states")
            if negative is None:
                negative_metadata.append(None)
                negative_items.append(None)
            else:
                if not isinstance(negative, torch.Tensor):
                    raise TypeError(
                        f"Ming request {request.request_id!r} negative_thinker_hidden_states must be a Tensor, "
                        f"got {type(negative).__name__}"
                    )
                normalized_shape = tuple(negative.shape) if negative.dim() == 3 else (1, *tuple(negative.shape))
                negative_metadata.append((normalized_shape, negative.dtype, negative.device))
                negative_items.append(_as_hidden(negative, request.request_id, "negative_thinker_hidden_states"))

        provided_hidden_metadata = {item for item in hidden_metadata if item is not None}
        if len(provided_hidden_metadata) > 1:
            raise ValueError(
                "Ming request batch thinker hidden states have incompatible shape/dtype/device: "
                f"request_ids={req.request_ids}, metadata={hidden_metadata}"
            )
        provided_negative_metadata = {item for item in negative_metadata if item is not None}
        if len(provided_negative_metadata) > 1:
            raise ValueError(
                "Ming request batch negative hidden states have incompatible shape/dtype/device: "
                f"request_ids={req.request_ids}, metadata={negative_metadata}"
            )
        for request, positive, negative in zip(req.requests, hidden_items, negative_items, strict=True):
            if negative is not None and negative.shape != positive.shape:
                raise ValueError(
                    f"Ming request {request.request_id!r} negative hidden shape {tuple(negative.shape)} "
                    f"does not match positive shape {tuple(positive.shape)}"
                )

        first_shape = tuple(hidden_items[0].shape)
        if any(tuple(item.shape) != first_shape for item in hidden_items[1:]):
            details = [(rid, tuple(item.shape)) for rid, item in zip(req.request_ids, hidden_items, strict=True)]
            raise ValueError(f"Ming request batch hidden-state shapes are incompatible: {details}")
        hidden_batch = torch.cat(hidden_items, dim=0)
        cap_batch = self.condition_encoder(hidden_batch)
        cap_feats = [cap_batch[i] for i in range(req.num_reqs)]

        negative_cap_feats: list[torch.Tensor] = []
        if any(item is not None for item in negative_items):
            negative_batch = torch.cat(
                [
                    item if item is not None else torch.zeros_like(hidden)
                    for item, hidden in zip(negative_items, hidden_items, strict=True)
                ],
                dim=0,
            )
            negative_batch_feats = self.condition_encoder(negative_batch)
            negative_cap_feats = [
                negative_batch_feats[i] if item is not None else self.condition_encoder.zero_negative(cap_feats[i])
                for i, item in enumerate(negative_items)
            ]
        else:
            negative_cap_feats = [self.condition_encoder.zero_negative(item) for item in cap_feats]

        byte5_features: list[torch.Tensor | None] = [None] * req.num_reqs
        if self.byte5 is not None:
            for i, (request, extra) in enumerate(zip(req.requests, extras, strict=True)):
                texts = self._resolve_byte5_texts(extra, request.sampling_params)
                if texts:
                    encoded = self.byte5(texts).to(device=target_device, dtype=target_dtype)
                    byte5_features[i] = encoded.reshape(1, -1, encoded.shape[-1])[0]
        for i, byte5 in enumerate(byte5_features):
            if byte5 is not None:
                cap_feats[i] = torch.cat((cap_feats[i], byte5), dim=0)
                negative_cap_feats[i] = torch.cat((negative_cap_feats[i], torch.zeros_like(byte5)), dim=0)

        # Sampling knobs are resolved per request. The scheduler's compatibility
        # key keeps shape/control-flow fields homogeneous within this wave.
        resolved_params: list[tuple[OmniDiffusionSamplingParams, int, int, int, float, int | None]] = []
        for request in req.requests:
            sp = request.sampling_params
            ea = sp.extra_args or {}
            values: dict[str, Any] = {}
            for ea_key, sp_attr, default in (
                ("height", "height", cfg.default_height),
                ("width", "width", cfg.default_width),
                ("steps", "num_inference_steps", cfg.num_inference_steps),
                ("cfg", "guidance_scale", cfg.guidance_scale),
                ("seed", "seed", None),
            ):
                for value in (ea.get(ea_key), getattr(sp, sp_attr), default):
                    if value is not None:
                        values[ea_key] = value
                        break
            explicit_seed = ea.get("seed")
            seed = values.get("seed")
            if sp.generator is not None and explicit_seed is None:
                generator = sp.generator
            elif seed is not None:
                generator = torch.Generator(device=target_device).manual_seed(int(seed))
            else:
                generator = sp.generator
            z_sp = copy(sp)
            z_sp.height = int(values["height"])
            z_sp.width = int(values["width"])
            z_sp.num_inference_steps = int(values["steps"])
            z_sp.guidance_scale = float(values["cfg"])
            z_sp.generator = generator
            z_sp.output_type = "pt"
            resolved_params.append((z_sp, z_sp.height, z_sp.width, z_sp.num_inference_steps, z_sp.guidance_scale, seed))

        heights = {item[1] for item in resolved_params}
        widths = {item[2] for item in resolved_params}
        steps = {item[3] for item in resolved_params}
        guidance = {item[4] for item in resolved_params}
        if len(heights) != 1 or len(widths) != 1 or len(steps) != 1 or len(guidance) != 1:
            raise ValueError(f"Ming request batch has incompatible sampling fields for request_ids={req.request_ids}")
        height, width, num_inference_steps, guidance_scale = (
            next(iter(heights)), next(iter(widths)), next(iter(steps)), next(iter(guidance))
        )
        num_images_per_prompt = resolved_params[0][0].num_outputs_per_prompt or 1
        if any(item[0].num_outputs_per_prompt != num_images_per_prompt for item in resolved_params):
            raise ValueError(
                "Ming request batch has incompatible num_outputs_per_prompt "
                f"for request_ids={req.request_ids}"
            )

        prompt_embeds = cap_feats
        negative_prompt_embeds = negative_cap_feats
        self._pending_prompt_embeds = prompt_embeds
        self._pending_negative_prompt_embeds = negative_prompt_embeds

        z_req = DiffusionRequestBatch(
            requests=[
                OmniDiffusionRequest(prompt={"prompt": ""}, sampling_params=z_sp, request_id=request.request_id)
                for request, (z_sp, *_rest) in zip(req.requests, resolved_params, strict=True)
            ]
        )

        ref_latents = []
        has_reference = [extra.get("reference_image") is not None for extra in extras]
        if any(has_reference):
            if not all(has_reference):
                raise ValueError(f"Ming request batch mixes reference and non-reference requests: {req.request_ids}")
            ref_latents = [
                self._encode_reference_image(extra.get("reference_image"), height, width) for extra in extras
            ]
            ref_latent = torch.cat(ref_latents, dim=0)
            ref_latent = ref_latent.repeat_interleave(num_images_per_prompt, dim=0)
        else:
            ref_latent = None
        set_forward_context_ref_latent(ref_latent)

        logger.debug(
            "[MingImagePipeline.forward] running z_pipeline hw=(%d,%d) steps=%d cfg=%.2f seed=%s overrides=%s ref=%s",
            height,
            width,
            num_inference_steps,
            guidance_scale,
            [item[5] for item in resolved_params],
            [item[0].extra_args for item in resolved_params],
            None if ref_latent is None else tuple(ref_latent.shape),
        )
        try:
            output: DiffusionOutput = super().forward(z_req)
        finally:
            set_forward_context_ref_latent(None)
            # Drop request-scoped conditioning so we don't retain GPU tensors.
            self._pending_prompt_embeds = None
            self._pending_negative_prompt_embeds = None

        raw = output.output
        if not isinstance(raw, torch.Tensor):
            raise RuntimeError(f"ZImagePipeline returned non-tensor output: {type(raw).__name__}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[MingImagePipeline.forward] produced image tensor shape=%s range=[%.3f,%.3f]",
                tuple(raw.shape),
                raw.float().min().item(),
                raw.float().max().item(),
            )
        return split_diffusion_output_by_request(output, req, num_outputs_per_prompt=num_images_per_prompt)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def get_ming_image_post_process_func(od_config: OmniDiffusionConfig):
    """Return a post-process callable that converts the raw VAE tensor to PIL.

    The diffusion engine calls ``post_process_func(output_data)`` where
    ``output_data`` is the ``DiffusionOutput.output`` tensor returned by
    ``MingImagePipeline.forward``. It has shape ``[B, 3, H, W]`` in ``[-1, 1]``
    (Z-image VAE convention). We run the standard ``VaeImageProcessor``
    postprocess to convert it to ``list[PIL.Image]`` which vllm-omni's
    ``OmniRequestOutput.from_diffusion`` then bubbles up as
    ``omni_outputs.images`` for serving_chat to base64-encode.

    Registered via ``_DIFFUSION_POST_PROCESS_FUNCS["MingImagePipeline"]``
    in vllm_omni/diffusion/registry.py.
    """
    import json

    model_path = od_config.model
    vae_config_path = os.path.join(model_path, "vae", "config.json")
    try:
        with open(vae_config_path) as f:
            vae_cfg = json.load(f)
        block_out_channels = vae_cfg.get("block_out_channels", [128, 256, 512, 512])
        vae_scale_factor = 2 ** (len(block_out_channels) - 1)
    except Exception:
        vae_scale_factor = 8  # Ming's Flux-format VAE default

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor * 2, do_convert_rgb=True)

    def post_process_func(images: torch.Tensor):
        # VaeImageProcessor.postprocess with default output_type="pil"
        # returns ``list[PIL.Image]``.
        return image_processor.postprocess(images.float())

    return post_process_func


def get_ming_image_pre_process_func(od_config: OmniDiffusionConfig):
    """Annotate Ming requests with the fields that must be homogeneous in a wave."""

    del od_config
    defaults = MingImageGenConfig()

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        extra = request.prompt.get("extra") if isinstance(request.prompt, dict) else {}
        extra = extra or {}
        sampling = request.sampling_params
        sampling_extra = sampling.extra_args or {}

        hidden = extra.get("thinker_hidden_states")
        if hidden is None:
            hidden = sampling_extra.get("thinker_hidden_states")
        if isinstance(hidden, torch.Tensor):
            hidden_shape = tuple(hidden.shape[-2:]) if hidden.dim() >= 2 else tuple(hidden.shape)
            hidden_dtype = str(hidden.dtype)
        else:
            scale = defaults.img_gen_scales[-1]
            hidden_shape = (scale * scale, defaults.thinker_hidden_size)
            hidden_dtype = str(sampling_extra.get("thinker_hidden_states_dtype", "default"))

        def resolve(name: str, attr: str, default: Any) -> Any:
            return next(
                (
                    value
                    for value in (sampling_extra.get(name), getattr(sampling, attr), default)
                    if value is not None
                ),
                default,
            )

        byte5_count = len(MingImagePipeline._resolve_byte5_texts(extra, sampling))
        request.batch_compatibility_key = (
            "ming_image",
            int(extra.get("reference_image") is not None),
            byte5_count,
            hidden_shape,
            hidden_dtype,
            int(resolve("height", "height", defaults.default_height)),
            int(resolve("width", "width", defaults.default_width)),
            int(resolve("steps", "num_inference_steps", defaults.num_inference_steps)),
            float(resolve("cfg", "guidance_scale", defaults.guidance_scale)),
            int(sampling.num_outputs_per_prompt or 1),
            sampling.output_type or "pil",
            sampling.strength,
        )
        return request

    return pre_process_func


__all__ = [
    "MingImagePipeline",
    "get_ming_image_pre_process_func",
    "get_ming_image_post_process_func",
]

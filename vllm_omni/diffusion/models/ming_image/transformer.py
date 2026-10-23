# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ming-Image adapter for the shared Z-Image transformer."""

from __future__ import annotations

import torch
from cache_dit import ForwardPattern

from vllm_omni.diffusion.cache.cachedit import CacheDiTAdapterConfig
from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available
from vllm_omni.diffusion.layers.rope import RotaryEmbedding
from vllm_omni.diffusion.models.z_image.z_image_transformer import ZImageTransformer2DModel


class MingImageRotaryEmbedding(RotaryEmbedding):
    """Rotate Q/K with the checkpoint's FP32 coefficients, rounded once.

    ``RotaryEmbedding`` rounds the coefficients to the Q/K dtype first. The
    Ming-Image reference rotates with FP32 coefficients and rounds only the
    result, which keeps low-precision rows stable, and the same math applies one
    cos/sin row per request so a batched wave keeps its own rotation.
    """

    # The shared attention must hand over the unrounded coefficients, and the
    # fused QK-norm/RoPE kernel would replace this rotation with its own.
    owns_rotation_dtype = True
    supports_fused_qk_norm_rope = False

    def forward(self, x, cos, sin):
        if not self.half_head_dim:
            cos = cos[..., ::2]
            sin = sin[..., ::2]
        rotary_dim = cos.shape[-1] * 2
        pairs = torch.view_as_complex(x[..., :rotary_dim].float().reshape(*x.shape[:-1], -1, 2))
        frequencies = torch.complex(cos.float(), sin.float()).unsqueeze(-2)
        rotated = torch.view_as_real(pairs * frequencies).flatten(-2)
        return torch.cat((rotated.to(x.dtype), x[..., rotary_dim:]), dim=-1)

class MingImageTransformer2DModel(ZImageTransformer2DModel):
    """Runtime adapter for vendor DiffusionTransformer checkpoints.

    This adapter reuses the shared Z-Image DiT and injects Ming-specific request conditions.
    """

    _cache_dit_adapter_config = CacheDiTAdapterConfig(
        block_forward_patterns={"layers": ForwardPattern.Pattern_3},
        has_separate_cfg=False,
        check_forward_pattern=False,
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Ming-Image owns the Q/K rotation policy for every block it reuses.
        for block in self.layers:
            block.attention.rope = MingImageRotaryEmbedding(is_neox_style=False)

    def forward(
        self,
        x: list[torch.Tensor],
        t,
        cap_feats: list[torch.Tensor],
        patch_size=2,
        f_patch_size=1,
    ):
        if not x or any(item.ndim != 4 for item in x):
            raise ValueError("Ming-Image DiT samples must have shape [C, F, H, W].")
        if len(cap_feats) != len(x):
            raise ValueError("Ming-Image query conditions must match the DiT branch batch.")
        generated_frames = [item.shape[1] for item in x]
        ref_x = None
        cap_feats_2 = None
        if is_forward_context_available():
            context = get_forward_context()
            batch_size = len(x)
            start = batch_size if context.cfg_branch == "negative" else 0
            reference = context.ref_latent
            direct = context.direct_condition
            if context.cfg_branch is not None:
                if reference is not None and reference.shape[0] == 2 * batch_size:
                    reference = reference[start : start + batch_size]
                if direct is not None and direct.shape[0] == 2 * batch_size:
                    direct = direct[start : start + batch_size]
            if reference is not None:
                ref_x = [item.to(device=x[0].device, dtype=x[0].dtype) for item in reference.unbind(dim=0)]
            if direct is not None:
                cap_feats_2 = [item.to(device=x[0].device, dtype=x[0].dtype) for item in direct.unbind(dim=0)]

        if cap_feats_2 is None or len(cap_feats_2) != len(x):
            raise ValueError("Ming-Image projected direct conditions must match the DiT branch batch.")
        if ref_x is not None and len(ref_x) != len(x):
            raise ValueError("Ming-Image reference latents must match the DiT branch batch.")

        output, metadata = super().forward(
            x,
            t,
            cap_feats,
            patch_size=patch_size,
            f_patch_size=f_patch_size,
            ref_x=ref_x,
            cap_feats_2=cap_feats_2,
        )
        output = [item[:, :frames] for item, frames in zip(output, generated_frames)]
        return output, metadata


__all__ = ["MingImageRotaryEmbedding", "MingImageTransformer2DModel"]


__all__ = ["MingImageTransformer2DModel"]

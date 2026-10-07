# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real parent/attention/SDPA padding contracts with independent mean oracles.

Only checkpoint projections and platform dispatch are replaced. Alignment,
mask metadata, SDPA math, unification, unpatchify and Ming frame cropping run.
Q/K are zero: softmax is uniform over valid keys, so literal means are exact
oracles; singleton parity alone would miss incorrectly masked learned tokens.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend, SDPAImpl
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.attention.parallel.base import NoParallelAttention
from vllm_omni.diffusion.forward_context import ForwardContext, override_forward_context
from vllm_omni.diffusion.layers.rope import RotaryEmbedding
from vllm_omni.diffusion.models.ming_image.transformer import MingImageTransformer2DModel
from vllm_omni.diffusion.models.z_image.z_image_transformer import (
    LEARNED_PADDING,
    ZERO_MASKED_PADDING,
    RopeEmbedder,
    UnifiedPrepare,
    ZImageAttention,
    ZImageTransformer2DModel,
)
from vllm_omni.model_executor.stage_input_processors.ming_image import thinker2image

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _CPUSDPA(SDPAImpl):
    def forward(self, query, key, value, attn_metadata=None):
        self.last_metadata = attn_metadata
        return self._forward_impl(query, key, value, attn_metadata)


class _ZeroQK(nn.Module):
    num_heads = num_kv_heads = 1

    def forward(self, hidden):
        return torch.cat((torch.zeros_like(hidden), torch.zeros_like(hidden), hidden), dim=-1), None


def _attention():
    # Avoid TP/native model construction; keep shared dispatch/capability guard.
    shared = Attention.__new__(Attention)
    nn.Module.__init__(shared)
    shared.skip_sequence_parallel = True
    shared._no_parallel_strategy = NoParallelAttention()
    shared.parallel_strategy = shared._no_parallel_strategy
    shared._scheduler_paged_kv = shared._has_custom_attention = False
    shared.paged_kv_cache_role = None
    shared._disable_kv_quant = False
    shared._kv_cache_dtype = shared._kv_cache_skip_steps = shared._kv_cache_skip_layers = None
    shared.allow_fp32_fallback = shared.use_ring = False
    shared.backend_explicit = True
    shared.attn_backend = SDPABackend
    shared.attn_spec = None
    shared.attention = _CPUSDPA(1, 4, 0.5)
    attention = ZImageAttention.__new__(ZImageAttention)
    nn.Module.__init__(attention)
    attention.to_qkv = _ZeroQK()
    attention.head_dim = 4
    attention.norm_q = attention.norm_k = nn.Identity()
    attention.rope = RotaryEmbedding(is_neox_style=False)
    attention.attn = shared
    attention.to_out = nn.ModuleList([nn.Identity()])
    return attention


def test_key_mask_excludes_padding_values_and_keeps_rows_independent():
    attention = _attention()
    values = torch.tensor([[1, 3, 1000], [10, 20, 30]], dtype=torch.float32)[..., None].expand(-1, -1, 4)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    cos, sin = torch.ones(2, 3, 2), torch.zeros(2, 3, 2)
    for sentinel in (1000, -9000):
        x = values.clone()
        x[0, 2] = sentinel
        output = attention(x, mask, cos, sin)
        torch.testing.assert_close(output[0, :2], torch.full((2, 4), 2.0), rtol=0, atol=1e-6)
        torch.testing.assert_close(output[1], torch.full((3, 4), 20.0), rtol=0, atol=1e-6)
        metadata = attention.attn.attention.last_metadata
        assert metadata.attn_mask.ndim == 2
        torch.testing.assert_close(metadata.attn_mask, mask)


def test_none_mask_stays_none_and_mask_free_backend_rejects_mask():
    attention = _attention()
    x = torch.tensor([1, 3, 5], dtype=torch.float32)[None, :, None].expand(1, 3, 4)
    cos, sin = torch.ones(1, 3, 2), torch.zeros(1, 3, 2)
    torch.testing.assert_close(attention(x, None, cos, sin), torch.full_like(x, 3))
    assert attention.attn.attention.last_metadata is None

    class MaskFree(SDPABackend):
        @classmethod
        def supports_attention_mask(cls, attention_spec=None):
            return False

    attention.attn.attn_backend = MaskFree
    with pytest.raises(ValueError, match="does not support.*attn_mask"):
        attention(x, torch.tensor([[True, True, False]]), cos, sin)


class _MeanBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = _attention()
        self.calls = []

    def forward(self, x, mask, cos, sin, adaln_input=None):
        self.calls.append((x.detach().clone(), mask.clone()))
        return self.attention(x, mask, cos, sin)


class _Final(nn.Module):
    def forward(self, x, adaln_input):
        return x


def _transformer(mode, *, mean_blocks=True):
    model = MingImageTransformer2DModel.__new__(MingImageTransformer2DModel)
    nn.Module.__init__(model)
    model.all_patch_size, model.all_f_patch_size = (2,), (1,)
    model.out_channels, model.t_scale = 1, 1000
    model.multi_frame_output = True
    model.alignment_padding_mode = mode
    model.t_embedder = nn.Identity()
    model.all_x_embedder = nn.ModuleDict({"2-1": nn.Identity()})
    model.cap_embedder = nn.Identity()
    model.all_final_layer = nn.ModuleDict({"2-1": _Final()})
    model.x_pad_token = nn.Parameter(torch.full((1, 4), 101.0)) if mode == LEARNED_PADDING else None
    model.cap_pad_token = nn.Parameter(torch.full((1, 4), 203.0)) if mode == LEARNED_PADDING else None
    model.rope_embedder = RopeEmbedder(axes_dims=[2, 2, 0], axes_lens=[1024, 32, 32])
    model.unified_prepare = UnifiedPrepare()
    for name in ("noise_refiner", "context_refiner", "layers"):
        setattr(model, name, nn.ModuleList([_MeanBlock()] if mean_blocks else []))
    return model


def _bridge_conditions(prefix):
    ids = [11] * prefix + [157158] + [157157] * 256 + [157159]
    payload = {"final_hidden_states": torch.full((len(ids), 2048), 2.0)}
    payload.update({f"hidden_states_{i}": torch.full((len(ids), 2048), 3.0) for i in (5, 12, 20)})
    source = SimpleNamespace(prompt_token_ids=ids, outputs=[SimpleNamespace(multimodal_output=payload)])
    extra = thinker2image([source])[0]["extra"]
    assert extra["query_hidden_states"].shape == (256, 2048)
    assert extra["direct_hidden_states"].shape == (prefix, 6144)
    # Weight-free checkpoint projection; preserve real producer token lengths.
    return extra["query_hidden_states"][:, :4], extra["direct_hidden_states"][:, :4]


@pytest.mark.parametrize("mode", [ZERO_MASKED_PADDING, LEARNED_PADDING])
@pytest.mark.parametrize("prefix", [1, 31, 32, 33])
def test_parent_attention_alignment_batch_and_singleton_match_independent_mean(mode, prefix):
    model = _transformer(mode)
    query, direct = _bridge_conditions(prefix)
    # Public 32x48 / 64x80 (VAE scale8); F is not the request batch axis.
    learned = mode == LEARNED_PADDING
    frames = 4 if learned else 1
    images = [torch.full((1, frames, 4, 6), 7.0), torch.full((1, frames, 8 if learned else 16, 10), 11.0)]
    context = ForwardContext(direct_condition=torch.stack([direct, direct]))
    with override_forward_context(context):
        outputs, _ = model(images, torch.zeros(2), [query, query])
    x_call, x_mask = model.noise_refiner[0].calls[0]
    cap_call, cap_mask = model.context_refiner[0].calls[0]
    unified_mask = model.layers[0].calls[0][1]
    cap_len = 256 + prefix
    cap_aligned = ((cap_len + 31) // 32) * 32
    x_lengths = [24, 80] if learned else [6, 40]
    x_aligned = [32, 96] if learned else [32, 64]
    max_x_aligned = x_aligned[1]
    for row, (length, aligned, value) in enumerate(zip(x_lengths, x_aligned, (7, 11), strict=True)):
        x_valid = aligned if learned else length
        cap_valid = cap_aligned if learned else cap_len
        torch.testing.assert_close(x_mask[row], torch.arange(max_x_aligned) < x_valid)
        torch.testing.assert_close(cap_mask[row], torch.arange(cap_aligned) < cap_valid)
        expected_mask = torch.cat((torch.arange(aligned) < x_valid, torch.arange(cap_aligned) < cap_valid))
        expected_mask = torch.cat((expected_mask, torch.zeros(max_x_aligned - aligned, dtype=torch.bool)))
        torch.testing.assert_close(unified_mask[row], expected_mask)
        assert torch.all(x_call[row, length:aligned] == (101 if learned else 0))
        assert torch.all(cap_call[row, cap_len:cap_aligned] == (203 if learned else 0))
        total = value * length + 2 * 256 + 3 * prefix
        if learned:
            total += 101 * (aligned - length) + 203 * (cap_aligned - cap_len)
        expected = total / (x_valid + cap_valid)
        torch.testing.assert_close(outputs[row], torch.full_like(images[row], expected), rtol=1e-5, atol=1e-5)
        with override_forward_context(ForwardContext(direct_condition=direct[None])):
            singleton, _ = model([images[row]], torch.zeros(1), [query])
        torch.testing.assert_close(outputs[row], singleton[0], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mode", [ZERO_MASKED_PADDING, LEARNED_PADDING])
def test_patch_positions_varying_direct_lengths_and_reference_crop(mode):
    model = _transformer(mode, mean_blocks=False)
    pairs = [_bridge_conditions(prefix) for prefix in (1, 31, 32, 33)]
    frames = 4 if mode == LEARNED_PADDING else 1
    patches = 6 * (frames + 1)
    images = [torch.arange(frames * 4 * 6, dtype=torch.float32).reshape(1, frames, 4, 6) + 100 * i for i in range(4)]
    refs = [torch.full((1, 1, 4, 6), 999.0)] * 4
    patched = model.patchify_and_embed(images, [p[0] for p in pairs], 2, 1, refs, [p[1] for p in pairs])
    for row, prefix in enumerate((1, 31, 32, 33)):
        cap_len = 256 + prefix
        aligned = ((cap_len + 31) // 32) * 32
        assert patched[2][row] == (frames + 1, 4, 6)
        assert patched[0][row].shape == (32, 4)
        torch.testing.assert_close(patched[5][row], torch.arange(32) >= patches)
        torch.testing.assert_close(patched[6][row], torch.arange(aligned) >= cap_len)
        torch.testing.assert_close(patched[4][row][:, 0], torch.arange(1, aligned + 1, dtype=torch.int32))
        expected_positions = torch.tensor(
            [[aligned + 1 + frame, y, x] for frame in range(frames + 1) for y in range(2) for x in range(3)],
            dtype=torch.int32,
        )
        torch.testing.assert_close(patched[3][row][:patches], expected_positions)
        assert torch.count_nonzero(patched[3][row][patches:]) == 0
        # Adapter accepts a rectangular direct context; ragged prefixes use the
        # real parent's list API above, not fictional padded request rows.
        with override_forward_context(ForwardContext(direct_condition=pairs[row][1][None], ref_latent=refs[row][None])):
            output, _ = model([images[row]], torch.zeros(1), [pairs[row][0]])
        assert output[0].shape == (1, frames, 4, 6)
        torch.testing.assert_close(output[0], images[row], rtol=0, atol=0)


@pytest.mark.parametrize("mode", [ZERO_MASKED_PADDING, LEARNED_PADDING])
def test_ragged_parent_direct_conditions_and_reference_attention_have_literal_means(mode):
    model = _transformer(mode)
    pairs = [_bridge_conditions(prefix) for prefix in (1, 33)]
    learned = mode == LEARNED_PADDING
    frames = 4 if learned else 1
    patches = 6 * frames
    images = [torch.full((1, frames, 4, 6), 7.0), torch.full((1, frames, 4, 6), 11.0)]
    refs = [torch.full((1, 1, 4, 6), 19.0)] * 2
    # Parent list API supports token padding. Pipeline admission keeps different
    # direct shapes in separate waves; the Ming context itself is rectangular.
    outputs, _ = ZImageTransformer2DModel.forward(
        model, images, torch.zeros(2), [p[0] for p in pairs], ref_x=refs, cap_feats_2=[p[1] for p in pairs]
    )
    cap_mask = model.context_refiner[0].calls[0][1]
    for row, (prefix, value) in enumerate(zip((1, 33), (7, 11), strict=True)):
        cap_len = 256 + prefix
        cap_aligned = ((cap_len + 31) // 32) * 32
        x_len = patches + 6
        valid_x = 32 if learned else x_len
        valid_cap = cap_aligned if learned else cap_len
        torch.testing.assert_close(cap_mask[row], torch.arange(320) < valid_cap)
        total = patches * value + 6 * 19 + 256 * 2 + prefix * 3
        if learned:
            total += (32 - x_len) * 101 + (cap_aligned - cap_len) * 203
        expected = total / (valid_x + valid_cap)
        assert outputs[row].shape == (1, images[row].shape[1] + 1, 4, 6)
        torch.testing.assert_close(outputs[row], torch.full_like(outputs[row], expected), rtol=1e-5, atol=1e-5)
        with override_forward_context(ForwardContext(direct_condition=pairs[row][1][None], ref_latent=refs[row][None])):
            cropped, _ = model([images[row]], torch.zeros(1), [pairs[row][0]])
        assert cropped[0].shape == images[row].shape
        torch.testing.assert_close(cropped[0], torch.full_like(images[row], expected), rtol=1e-5, atol=1e-5)

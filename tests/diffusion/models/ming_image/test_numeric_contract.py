# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Nonzero attention oracles; positions may differ between real padded rows.

Qwen's packed-QKV/dtype/short-long RoPE cases motivate these shapes. Expected
rotation is the explicit complex-pair equation, not another production helper.
"""

import pytest
import torch

from tests.diffusion.models.ming_image.test_padding_contract import _attention

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _NonzeroQKV(torch.nn.Module):
    num_heads = num_kv_heads = 1

    def forward(self, hidden):
        return torch.cat((hidden * 0.3, hidden * -0.2, hidden), dim=-1), None


def _rotate_pairs(x, cos, sin):
    result = torch.empty_like(x)
    result[..., ::2] = x[..., ::2] * cos - x[..., 1::2] * sin
    result[..., 1::2] = x[..., ::2] * sin + x[..., 1::2] * cos
    return result


@pytest.mark.parametrize("length", [1, 7, 31, 32, 33, 257])
@pytest.mark.parametrize("packed_view", [False, True])
def test_nonzero_attention_uses_each_rows_position_and_mask(length, packed_view):
    attention = _attention()
    attention.to_qkv = _NonzeroQKV()
    values = torch.arange(2 * length * 8, dtype=torch.float32).reshape(2, length, 8) / 100
    x = values[..., ::2] if packed_view else values[..., :4].contiguous()
    angles = torch.arange(2 * length * 2, dtype=torch.float32).reshape(2, length, 2) / 17
    angles[1] *= 2
    cos, sin = angles.cos(), angles.sin()
    mask = torch.ones(2, length, dtype=torch.bool)
    if length > 1:
        mask[0, -1] = False
    q = _rotate_pairs(x * 0.3, cos, sin)
    k = _rotate_pairs(x * -0.2, cos, sin)
    scores = q.double() @ k.double().transpose(-1, -2) * 0.5
    scores = scores.masked_fill(~mask[:, None, :], -torch.inf)
    expected = (scores.softmax(-1) @ x.double()).float()
    rotations = []
    handle = attention.rope.register_forward_hook(lambda _module, _args, result: rotations.append(result))
    actual = attention(x, mask, cos, sin)
    handle.remove()
    torch.testing.assert_close(rotations[0].reshape_as(q), q, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(rotations[1].reshape_as(k), k, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    for row in range(2):
        independent = attention(x[row : row + 1], mask[row : row + 1], cos[row : row + 1], sin[row : row + 1])
        torch.testing.assert_close(actual[row : row + 1], independent, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("length", [1, 7, 31, 32, 33, 257, 1024])
@pytest.mark.parametrize("packed_view", [False, True])
@pytest.mark.cuda
def test_cuda_batched_rope_quarter_turns_preserve_rows_and_unrotated_tail(dtype, length, packed_view):
    from vllm_omni.diffusion.models.z_image.z_image_transformer import ZImageRotaryEmbedding as RotaryEmbedding
    from vllm_omni.diffusion.models.z_image.z_image_transformer import _apply_batched_rope

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    values = torch.arange(3 * length * 2 * 24, device="cuda", dtype=torch.float32).reshape(3, length, 2, 24)
    packed = (values % 17 - 8).to(dtype)
    q = packed[..., :8] if packed_view else packed[..., :8].contiguous()
    k = packed[..., 8:16] if packed_view else packed[..., 8:16].contiguous()
    # Three request rows: identity, +90 degrees, 180 degrees. These rotations
    # use exact 0/1 coefficients, so all dtypes have an exact independent oracle.
    cos = torch.tensor([1, 0, -1], device="cuda", dtype=dtype)[:, None, None].expand(3, length, 3)
    sin = torch.tensor([0, 1, 0], device="cuda", dtype=dtype)[:, None, None].expand(3, length, 3)
    expected_q, expected_k = q.clone(), k.clone()
    for expected, source in ((expected_q, q), (expected_k, k)):
        expected[..., :6:2] = source[..., :6:2] * cos[:, :, None] - source[..., 1:6:2] * sin[:, :, None]
        expected[..., 1:6:2] = source[..., :6:2] * sin[:, :, None] + source[..., 1:6:2] * cos[:, :, None]
    actual_q, actual_k = _apply_batched_rope(RotaryEmbedding(is_neox_style=False), q, k, cos, sin)
    torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=0)
    torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=0)


@pytest.mark.cuda
def test_batched_rope_compile_fullgraph_repeated_and_changed_shape():
    from vllm_omni.diffusion.models.z_image.z_image_transformer import ZImageRotaryEmbedding as RotaryEmbedding
    from vllm_omni.diffusion.models.z_image.z_image_transformer import _apply_batched_rope

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    rope = RotaryEmbedding(is_neox_style=False)

    def fn(q, k, cos, sin):
        return _apply_batched_rope(rope, q, k, cos, sin)

    compiled = torch.compile(fn, dynamic=True, fullgraph=True)
    for length in (7, 33, 7):
        q = torch.arange(2 * length * 8, device="cuda", dtype=torch.float32).reshape(2, length, 1, 8) % 11
        cos = torch.tensor([1, 0], device="cuda", dtype=torch.float32)[:, None, None].expand(2, length, 4)
        sin = torch.tensor([0, 1], device="cuda", dtype=torch.float32)[:, None, None].expand(2, length, 4)
        expected = _rotate_pairs(q.squeeze(2), cos, sin).unsqueeze(2)
        eager = fn(q, q, cos, sin)
        actual = compiled(q, q, cos, sin)
        for output in (*eager, *actual):
            torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("packed_view", [False, True])
def test_low_precision_qk_keeps_float_rotary_coefficients_until_final_rounding(dtype, packed_view):
    from vllm_omni.diffusion.models.z_image.z_image_transformer import ZImageRotaryEmbedding as RotaryEmbedding
    from vllm_omni.diffusion.models.z_image.z_image_transformer import _apply_batched_rope

    values = torch.arange(2 * 7 * 2 * 24, dtype=torch.float32).reshape(2, 7, 2, 24) / 11 - 5
    packed = values.to(dtype)
    query = packed[..., :8] if packed_view else packed[..., :8].contiguous()
    key = packed[..., 8:16] if packed_view else packed[..., 8:16].contiguous()
    angles = torch.arange(2 * 7 * 4, dtype=torch.float32).reshape(2, 7, 4) / 13
    cos, sin = angles.cos(), angles.sin()
    # Vendor Ming rotates in complex64 outside autocast, then rounds once to
    # the Q/K dtype. The coefficients must not first be rounded to BF16/FP16.
    expected = []
    for source in (query, key):
        pairs = torch.view_as_complex(source.float().reshape(2, 7, 2, 4, 2))
        frequencies = torch.complex(cos, sin)[:, :, None, :]
        expected.append(torch.view_as_real(pairs * frequencies).flatten(-2).to(dtype))
    rope = RotaryEmbedding(is_neox_style=False)
    rope._forward_method = rope.forward_native
    actual = _apply_batched_rope(rope, query, key, cos, sin)
    for output, oracle in zip(actual, expected, strict=True):
        torch.testing.assert_close(output, oracle, rtol=0, atol=0)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Packed AR tokens must inject all image-generation query suffixes, not just the last."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.ming_image.model import MingImageForConditionalGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _model():
    model = MingImageForConditionalGeneration.__new__(MingImageForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(image_patch_token=157157, image_end_token=157159, image_start_token=157158)
    model.query_tokens_dict = torch.nn.ParameterDict(
        {"16x16": torch.nn.Parameter(torch.arange(256 * 4).reshape(256, 4).float(), requires_grad=False)}
    )
    return model


@pytest.mark.parametrize("prefixes", [[1, 7], [31, 32, 33], [1024, 1]])
def test_packed_query_suffix_injection_preserves_each_request_and_prefix(prefixes):
    model = _model()
    ids, ranges = [], []
    for prefix in prefixes:
        ids += [11] * prefix + [157158]
        ranges.append((len(ids), len(ids) + 256))
        ids += [157157] * 256 + [157159]
    inputs = torch.tensor(ids)
    embeds = torch.full((len(ids), 4), -71.0)
    model._maybe_inject_image_gen_query_embeds(inputs, embeds)
    mask = torch.zeros(len(ids), dtype=torch.bool)
    for start, end in ranges:
        torch.testing.assert_close(embeds[start:end], model.query_tokens_dict["16x16"], rtol=0, atol=0)
        mask[start:end] = True
    assert torch.all(embeds[~mask] == -71)


@pytest.mark.parametrize(
    "ids", [[157157], [11, 157159], [157158] + [157157] * 255 + [157159], [157158] + [157157] * 257 + [157159]]
)
def test_decode_or_invalid_suffix_does_not_inject_partial_query(ids):
    model = _model()
    embeds = torch.full((len(ids), 4), -71.0)
    model._maybe_inject_image_gen_query_embeds(torch.tensor(ids), embeds)
    assert torch.all(embeds == -71)


def test_query_injection_preserves_reference_features_with_same_patch_count():
    model = _model()
    block = [157158] + [157157] * 256 + [157159]
    tokens = torch.tensor(block + [11] + block)
    embeds = torch.full((len(tokens), 4), 37.0)
    reference_mask = torch.zeros(len(tokens), dtype=torch.bool)
    reference_mask[1:257] = True
    model._maybe_inject_image_gen_query_embeds(tokens, embeds, is_multimodal=reference_mask)
    assert torch.all(embeds[:259] == 37)
    torch.testing.assert_close(embeds[260:516], model.query_tokens_dict["16x16"], rtol=0, atol=0)


@pytest.mark.cuda
def test_cpu_input_ids_and_reference_mask_inject_into_cuda_embedding():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    model = _model().to("cuda")
    block = [157158] + [157157] * 256 + [157159]
    tokens = torch.tensor(block + [11] + block, device="cpu")
    embeds = torch.full((len(tokens), 4), 37.0, device="cuda")
    reference_mask = torch.zeros(len(tokens), dtype=torch.bool, device="cpu")
    reference_mask[1:257] = True
    model._maybe_inject_image_gen_query_embeds(tokens, embeds, is_multimodal=reference_mask)
    assert torch.all(embeds[:259] == 37)
    torch.testing.assert_close(embeds[260:516], model.query_tokens_dict["16x16"], rtol=0, atol=0)


def test_embed_input_ids_merges_reference_and_injects_every_packed_query():
    model = _model()
    block = [157158] + [157157] * 256 + [157159]
    tokens = torch.tensor(block + [11] + block + [12] + block)
    reference_mask = torch.zeros(len(tokens), dtype=torch.bool)
    reference_mask[1:257] = True
    model.language_model = SimpleNamespace(
        model=SimpleNamespace(word_embeddings=lambda ids: torch.full((len(ids), 4), -19.0))
    )
    reference = torch.arange(256 * 4).reshape(256, 4).float() + 10000
    output = model.embed_input_ids(tokens, (reference,), is_multimodal=reference_mask)
    torch.testing.assert_close(output[1:257], reference, rtol=0, atol=0)
    for start in (260, 519):
        torch.testing.assert_close(output[start : start + 256], model.query_tokens_dict["16x16"], rtol=0, atol=0)
    assert torch.all(output[[0, 257, 258, 259, 516, 517, 518, 775]] == -19)


def test_chunked_prefill_is_rejected_before_model_or_checkpoint_construction():
    config = SimpleNamespace(scheduler_config=SimpleNamespace(enable_chunked_prefill=True))
    with pytest.raises(ValueError, match="Ming-Image.*enable_chunked_prefill=False"):
        MingImageForConditionalGeneration(vllm_config=config)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Producer boundaries: 258 suffix tokens, one reference and captured layers."""

import pytest
import torch

from tests.model_executor.models.ming_image.test_config_and_pipeline import _source_output
from vllm_omni.model_executor.stage_input_processors.ming_image import thinker2image

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("count", [0, 1, 2])
def test_bridge_reference_cardinality_is_not_silently_truncated(count):
    refs = [object() for _ in range(count)]
    prompt = {"multi_modal_data": {"img2img": refs}}
    if count > 1:
        with pytest.raises(ValueError, match="exactly one reference"):
            thinker2image([_source_output()], prompt=prompt)
    else:
        extra = thinker2image([_source_output()], prompt=prompt)[0]["extra"]
        assert extra.get("reference_image") is (refs[0] if refs else None)


@pytest.mark.parametrize(
    "kind",
    [
        "no_outputs",
        "short_suffix",
        "wrong_patch",
        "wrong_end",
        "missing_final",
        "missing_layer",
        "length_mismatch",
        "batched_hidden",
    ],
)
def test_bridge_rejects_invalid_ar_output_before_publishing_conditions(kind):
    source = _source_output()
    data = source.outputs[0].multimodal_output
    if kind == "no_outputs":
        source.outputs = []
    elif kind == "short_suffix":
        source.prompt_token_ids = source.prompt_token_ids[-257:]
    elif kind == "wrong_patch":
        source.prompt_token_ids[-2] = 11
    elif kind == "wrong_end":
        source.prompt_token_ids[-1] = 11
    elif kind == "missing_final":
        data.pop("final_hidden_states")
    elif kind == "missing_layer":
        data.pop("hidden_states_12")
    elif kind == "length_mismatch":
        data["final_hidden_states"] = data["final_hidden_states"][:-1]
    else:
        data["final_hidden_states"] = data["final_hidden_states"][None].expand(2, -1, -1)
    with pytest.raises(ValueError):
        thinker2image([source])


@pytest.mark.parametrize("prefix", [1, 31, 32, 33, 1024])
def test_bridge_excludes_complete_query_suffix_and_preserves_prefix(prefix):
    source = _source_output(prefix_len=prefix)
    extra = thinker2image([source])[0]["extra"]
    assert extra["query_hidden_states"].shape == (256, 2048)
    assert extra["direct_hidden_states"].shape == (prefix, 6144)
    assert extra["query_hidden_states"][0, 0] == (prefix + 1) * 2048
    for index, layer in enumerate((5, 12, 20)):
        assert torch.all(extra["direct_hidden_states"][:, index * 2048 : (index + 1) * 2048] == layer)
    assert extra["query_hidden_states"].is_contiguous()
    assert extra["direct_hidden_states"].is_contiguous()

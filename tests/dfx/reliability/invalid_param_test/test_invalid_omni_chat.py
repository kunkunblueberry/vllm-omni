# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Invalid inputs on omni chat: ``POST /v1/chat/completions``, ``WS /v1/video/chat/stream``, ``WS /v1/realtime``."""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path
from typing import Any

import pytest
import yaml
from PIL import Image, ImageDraw

from tests.helpers.assertions import assert_diffusion_images_equal, assert_http_error_payload
from tests.helpers.fixtures.runtime import omni_fixture_lock
from tests.helpers.mark import hardware_marks, hardware_test
from tests.helpers.runtime import (
    OmniServer,
    OmniServerParams,
    OnlineOmniClient,
    dummy_messages_from_mix_data,
    iter_omni_server,
)
from tests.helpers.stage_config import get_deploy_config_path

pytestmark = [pytest.mark.slow]


def _minimal_chat_json(omni_server: OmniServer) -> dict[str, object]:
    """Minimal valid chat body; individual tests override one offending field."""
    return {
        "model": omni_server.model,
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }


_QWEN3_OMNI_SERVER = [
    pytest.param(
        OmniServerParams(
            model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
            stage_config_path=get_deploy_config_path("qwen3_omni_moe.yaml"),
            server_args=["--async-chunk"],
        ),
        id="qwen3_omni",
    ),
]


def _chat_completions_request_without_expectations(omni_server: OmniServer, case_id: str) -> dict[str, Any]:
    """Build ``send_chat_completions_http_request`` kwargs excluding ``err_code`` / ``err_message``."""
    if case_id == "malformed_json":
        return {"raw_body": "{not-json", "timeout": 120}
    if case_id == "missing_messages":
        return {"json": {"model": omni_server.model, "stream": False}, "timeout": 120}
    if case_id == "model_mismatch":
        return {
            "json": {
                "model": "this-model-is-not-served-on-this-instance",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
            "timeout": 120,
        }
    body = _minimal_chat_json(omni_server)
    if case_id == "stream_wrong_type":
        body["stream"] = "not-a-boolean"  # type: ignore[assignment]
    elif case_id == "temperature_wrong_type":
        body["temperature"] = "hot"  # type: ignore[assignment]
    elif case_id == "max_tokens_wrong_type":
        body["max_tokens"] = "not-an-int"  # type: ignore[assignment] # type: ignore[assignment]
    elif case_id == "modalities_list_bad":
        body["modalities"] = [123]  # type: ignore[assignment]
    elif case_id == "response_format_json_schema_incomplete":
        body["response_format"] = {"type": "json_schema"}
    elif case_id == "logprobs_wrong_type":
        body["logprobs"] = "yes"  # type: ignore[assignment]
    elif case_id == "logprobs_top_without_enabled":
        body["logprobs"] = False
        body["top_logprobs"] = 5
    elif case_id == "speaker_unknown":
        body["speaker"] = "zz_invalid_qwen3_omni_chat_speaker_xyz"
    elif case_id == "audio_format_unsupported":
        body["modalities"] = ["text", "audio"]
        body["audio"] = {"voice": "alloy", "format": "aac"}
    elif case_id == "audio_data_url_corrupt":
        body["messages"] = dummy_messages_from_mix_data(
            audio_data_url="data:audio/invalid;base64,AAAA",
            content_text="What is the capital of China? Answer in 20 words.",
        )
        body["stream"] = True
    else:
        raise AssertionError(f"unknown chat completions invalid case_id {case_id!r}")
    return {"json": body, "timeout": 120}


# ─────────────────────────────────────────────────────────────────────────────
# POST /v1/chat/completions
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize(
    "case_id, err_code, err_message",
    [
        pytest.param(
            "malformed_json",
            400,
            ("json_invalid", "decode error"),
            id="malformed_json",
        ),
        pytest.param("missing_messages", 400, ("messages", "missing", "Field required"), id="missing_messages"),
        pytest.param(
            "model_mismatch",
            404,
            ("model", "does not exist", "NotFoundError"),
            id="model_mismatch",
        ),
        pytest.param("stream_wrong_type", 400, ("stream", "bool_parsing", "not-a-boolean"), id="invalid_stream_type"),
        pytest.param(
            "temperature_wrong_type",
            400,
            ("temperature", "float_parsing", "a valid number"),
            id="invalid_temperature_type",
        ),
        pytest.param(
            "max_tokens_wrong_type", 400, ("max_tokens", "int_parsing", "integer"), id="invalid_max_tokens_type"
        ),
        pytest.param(
            "modalities_list_bad",
            400,
            ("modalities", "list of strings"),
            id="modalities_list_bad_element",
        ),
        pytest.param(
            "response_format_json_schema_incomplete",
            400,
            ("response_format", "BadRequestError", "json_schema"),
            id="invalid_response_format_json_schema",
        ),
        pytest.param("logprobs_wrong_type", 400, ("logprobs", "boolean"), id="logprobs_wrong_type"),
        pytest.param(
            "logprobs_top_without_enabled",
            400,
            ("top_logprobs", "logprobs", "True"),
            id="logprobs_top_without_logprobs_true",
        ),
        pytest.param(
            "speaker_unknown",
            400,
            ("Invalid speaker", "Supported"),
            id="speaker_unknown_preset",
        ),
        pytest.param(
            "audio_format_unsupported",
            400,
            ("Invalid audio format", "aac", "Supported formats"),
            id="unsupported_audio_format",
        ),
        pytest.param(
            "audio_data_url_corrupt",
            400,
            "Invalid or corrupted audio data",
            id="invalid_audio_data_url",
        ),
    ],
)
@pytest.mark.parametrize("omni_server", _QWEN3_OMNI_SERVER, indirect=True)
def test_chat_completions_invalid_requests(
    omni_server: OmniServer,
    online_client: OnlineOmniClient,
    case_id: str,
    err_code: int | tuple[int, ...],
    err_message: str | tuple[str, ...],
) -> None:
    cfg = _chat_completions_request_without_expectations(omni_server, case_id)
    cfg["err_code"] = err_code
    cfg["err_message"] = err_message
    online_client.send_chat_completions_http_request(cfg)[0]


# ─────────────────────────────────────────────────────────────────────────────
# WS /v1/video/chat/stream
# ─────────────────────────────────────────────────────────────────────────────

# First-frame JSON with ``omni_server.model`` for ``test_video_chat_stream_invalid_requests``.
_VIDEO_CHAT_WS_SESSION_MODALITIES_SCALAR = object()


@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize(
    "send_frames_spec, err_message",
    [
        pytest.param("{not-json", "Invalid JSON", id="invalid_json"),
        pytest.param(
            _VIDEO_CHAT_WS_SESSION_MODALITIES_SCALAR,
            "Invalid session config",
            id="session_modalities_scalar",
        ),
    ],
)
@pytest.mark.parametrize("omni_server", _QWEN3_OMNI_SERVER, indirect=True)
def test_video_chat_stream_invalid_requests(
    omni_server: OmniServer,
    online_client: OnlineOmniClient,
    send_frames_spec: Any,
    err_message: str,
) -> None:
    if send_frames_spec is _VIDEO_CHAT_WS_SESSION_MODALITIES_SCALAR:
        send_frames = json.dumps({"type": "session.config", "model": omni_server.model, "modalities": "text"})
    else:
        send_frames = send_frames_spec
    assert isinstance(send_frames, str)
    online_client.send_video_chat_stream_ws_request(
        {
            "send_frames": send_frames,
            "timeout": 120,
            "ws_max_size": None,
            "ws_json_type": "error",
            "err_message": err_message,
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# WS /v1/realtime
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize(
    "send_frames_spec, ws_error_code, err_message",
    [
        pytest.param("{not-json", "invalid_event", "Invalid or unrecognized client event", id="invalid_json"),
        pytest.param(
            json.dumps({"type": "session.update"}),
            "invalid_event",
            "Invalid or unrecognized client event",
            id="session_update_missing_session",
        ),
        pytest.param(
            json.dumps({"type": "session.update", "session": []}),
            "invalid_event",
            "Invalid or unrecognized client event",
            id="session_update_invalid_session",
        ),
        pytest.param(
            json.dumps({"type": "input_audio_buffer.commit"}),
            "invalid_request_error",
            "Input audio buffer is empty",
            id="commit_without_audio",
        ),
        pytest.param(
            json.dumps({"type": "unknown.event"}),
            "invalid_event",
            "Invalid or unrecognized client event",
            id="unknown_event_type",
        ),
        pytest.param(
            json.dumps({"type": "input_audio_buffer.append", "audio": "not-valid-base64!!!"}),
            "invalid_request_error",
            "Invalid base64 audio data",
            id="invalid_audio_append",
        ),
    ],
)
@pytest.mark.parametrize("omni_server", _QWEN3_OMNI_SERVER, indirect=True)
def test_realtime_invalid_requests(
    omni_server: OmniServer,
    online_client: OnlineOmniClient,
    send_frames_spec: Any,
    ws_error_code: str,
    err_message: str | tuple[str, ...],
) -> None:
    online_client.send_realtime_ws_request(
        {
            "send_frames": send_frames_spec,
            "timeout": 120,
            "ws_max_size": None,
            "ws_skip_types": ["session.created", "conversation.created"],
            "ws_json_type": "error",
            "ws_error_code": ws_error_code,
            "err_message": err_message,
        }
    )


# ─── POST /v1/chat/completions: local Ming image diffusion ────────────────────
# Linux prerequisites and invocation (no HF downloads; three CUDA cards):
# model_prefix= MING_IMAGE_CHECKPOINT_ROOT=/models CUDA_VISIBLE_DEVICES=0,1,2 \
#   pytest -sv tests/dfx/reliability/invalid_param_test/test_invalid_omni_chat.py \
#   -k ming -m 'slow and diffusion and H100 and cards_3' --run-level full_model
_MING_IMAGE_SERVERS = [
    pytest.param(checkpoint, id=case_id, marks=hardware_marks(res={"cuda": "H100"}, num_cards=3))
    for checkpoint, case_id in (
        ("Ming-Image-0.1-Design", "design"),
        ("Ming-Image-0.1-Design-Layer", "layer"),
    )
]


@pytest.fixture(scope="module")
def omni_server(request, run_level, tmp_path_factory):
    """Keep existing servers unchanged; prepare the Ming local three-card deploy."""
    if isinstance(request.param, OmniServerParams):
        yield from iter_omni_server(request, run_level, omni_fixture_lock)
        return
    root = os.environ.get("MING_IMAGE_CHECKPOINT_ROOT")
    if not root:
        pytest.fail("Set MING_IMAGE_CHECKPOINT_ROOT to the prepared local checkpoint parent")
    if os.environ.get("model_prefix"):
        pytest.fail("Unset model_prefix: Ming checkpoints are absolute local paths")
    model = Path(root).resolve() / request.param
    for component in ("mllm", "transformer"):
        if not (model / component / "config.json").is_file():
            pytest.fail(f"Missing prepared checkpoint: {model / component / 'config.json'}")
    deploy = yaml.safe_load(Path(get_deploy_config_path("ming_image_throughput.yaml")).read_text())
    ar, dit = deploy["stages"]
    ar.update(
        tensor_parallel_size=2,
        devices="0,1",
        gpu_memory_utilization=0.8,
        max_model_len=2048,
        max_num_batched_tokens=2048,
    )
    dit.update(devices="2", enforce_eager=True)
    dit.pop("diffusion_compile_granularity", None)
    dit.pop("diffusion_compile_dynamic", None)
    dit["engine_args"]["diffusion_attention_config"] = {"default": {"backend": "TORCH_SDPA"}}
    config = tmp_path_factory.mktemp("ming-invalid-chat") / "ming_tp2_step.yaml"
    config.write_text(yaml.safe_dump(deploy, sort_keys=False))
    request.param = OmniServerParams(
        model=str(model),
        stage_config_path=str(config),
        env_dict={
            "VLLM_BATCH_INVARIANT": "1",
            "VLLM_NO_USAGE_STATS": "1",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "OMP_NUM_THREADS": "4",
        },
        startup_timeout=360,
    )
    # Weekly reliability defaults to core_model; never substitute tiny weights here.
    yield from iter_omni_server(request, "full_model", omni_fixture_lock)


@pytest.fixture(scope="module")
def ming_reference_url():
    image = Image.new("RGBA", (64, 64), (240, 240, 240, 255))
    draw = ImageDraw.Draw(image)
    draw.ellipse((8, 8, 40, 40), fill=(230, 30, 30, 255))
    draw.rectangle((20, 46, 55, 57), fill=(25, 80, 220, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _minimal_ming_chat_json(omni_server, reference):
    layered = Path(omni_server.model).name.endswith("-Layer")
    text = "a cup of coffee on a table"
    if layered:
        text = "Decompose into two layers: red circle and blue rectangle. " + text
    content = (
        [{"type": "image_url", "image_url": {"url": reference}}, {"type": "text", "text": text}] if layered else text
    )
    extra = {"height": 64, "width": 64, "seed": 0, "num_inference_steps": 3}
    if layered:
        extra["num_layers"] = 2
    return {
        "model": omni_server.model,
        "messages": [{"role": "user", "content": content}],
        "modalities": ["image"],
        "extra_body": extra,
    }


@pytest.mark.diffusion
@pytest.mark.local_model
@pytest.mark.parametrize("omni_server", _MING_IMAGE_SERVERS, indirect=True)
@pytest.mark.parametrize(
    "override",
    [
        pytest.param({"height": 17}, id="unaligned_height"),
        pytest.param({"num_inference_steps": 0}, id="zero_steps"),
        pytest.param({"negative_prompt": "avoid blur"}, id="unsupported_negative_prompt"),
    ],
)
def test_ming_chat_invalid_diffusion_parameters(omni_server, online_client, ming_reference_url, override):
    body = _minimal_ming_chat_json(omni_server, ming_reference_url)
    body["extra_body"].update(override)
    responses = online_client.send_chat_completions_http_request(
        {"json": body, "timeout": 30, "err_code": 400 if "negative_prompt" in override else tuple(range(400, 600))}
    )
    assert_http_error_payload(responses[0])


@pytest.mark.diffusion
@pytest.mark.local_model
@pytest.mark.parametrize("omni_server", [_MING_IMAGE_SERVERS[1]], indirect=True)
def test_ming_layer_chat_requires_reference_image(omni_server, online_client, ming_reference_url):
    body = _minimal_ming_chat_json(omni_server, ming_reference_url)
    body["messages"][0]["content"] = "a cup of coffee on a table"
    responses = online_client.send_chat_completions_http_request(
        {"json": body, "timeout": 30, "err_code": tuple(range(400, 600))}
    )
    assert_http_error_payload(responses[0])


@pytest.mark.diffusion
@pytest.mark.local_model
@pytest.mark.parametrize("omni_server", _MING_IMAGE_SERVERS, indirect=True)
def test_ming_chat_error_retirement_preserves_exact_seed(omni_server, online_client, ming_reference_url):
    """The original invalid sequence must not poison exact replay of seed zero."""
    body = _minimal_ming_chat_json(omni_server, ming_reference_url)
    layered = Path(omni_server.model).name.endswith("-Layer")
    success_config = {
        **body,
        "expected_image_count": 3 if layered else 1,
        "expected_image_mode": "RGBA",
    }
    baseline = online_client.send_diffusion_request(success_config)
    for override in ({"height": 17}, {"num_inference_steps": 0}, {"negative_prompt": "avoid blur"}):
        invalid = {**body, "extra_body": {**body["extra_body"], **override}}
        responses = online_client.send_chat_completions_http_request(
            {
                "json": invalid,
                "timeout": 30,
                "err_code": 400 if "negative_prompt" in override else tuple(range(400, 600)),
            }
        )
        assert_http_error_payload(responses[0])
    if layered:
        invalid = {**body, "messages": [{"role": "user", "content": "a cup of coffee on a table"}]}
        responses = online_client.send_chat_completions_http_request(
            {"json": invalid, "timeout": 30, "err_code": tuple(range(400, 600))}
        )
        assert_http_error_payload(responses[0])
    repeated = online_client.send_diffusion_request(success_config)
    assert_diffusion_images_equal(repeated, baseline)

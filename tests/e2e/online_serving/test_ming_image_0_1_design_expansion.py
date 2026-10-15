# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ming-Image-0.1-Design family: exact three-card online numerics (L4).

Prerequisites: Linux, three CUDA GPUs with capacity for the original BF16
checkpoints, installed test dependencies, and MING_IMAGE_CHECKPOINT_ROOT containing
Ming-Image-0.1-Design and Ming-Image-0.1-Design-Layer (including mllm/transformer).
No HF downloads, dtype changes, approximate comparisons, or serial fallback.
Use --run-level full_model; an incorrect run level or missing local root fails.

Local exact parity (both checkpoints), from the repository root::

    CUDA_VISIBLE_DEVICES=0,1,2 model_prefix= MING_IMAGE_CHECKPOINT_ROOT=/models \
        pytest -sv tests/e2e/online_serving/test_ming_image_0_1_design_expansion.py \
        -k eight_prompt --run-level full_model

CI-like full success module::

    CUDA_VISIBLE_DEVICES=0,1,2 model_prefix= MING_IMAGE_CHECKPOINT_ROOT=/models \
        pytest -sv tests/e2e/online_serving/test_ming_image_0_1_design_expansion.py \
        -m 'full_model and diffusion and H100 and cards_3' --run-level full_model
"""

import base64
import io
import os
from pathlib import Path

import pytest
import yaml
from PIL import Image, ImageDraw

from tests.helpers.assertions import assert_diffusion_images_equal
from tests.helpers.fixtures.runtime import omni_fixture_lock
from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServerParams, iter_omni_server
from tests.helpers.stage_config import get_deploy_config_path

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion, pytest.mark.local_model]

PROMPTS = [
    "a cup of coffee on a table",
    "a toy dinosaur on a sandy beach",
    "a futuristic city skyline at sunset",
    "a bowl of fresh strawberries",
    "a medieval knight standing in the rain",
    "a cat wearing sunglasses lounging in a garden",
    "a spaceship flying above a volcano",
    "a watercolor painting of a mountain lake",
]
SERVER_CASES = [
    pytest.param(checkpoint, id=case_id, marks=hardware_marks(res={"cuda": "H100"}, num_cards=3))
    for checkpoint, case_id in (
        ("Ming-Image-0.1-Design", "design"),
        ("Ming-Image-0.1-Design-Layer", "layer"),
    )
]


@pytest.fixture(scope="module")
def omni_server(request, run_level, tmp_path_factory):
    """Prepare local paths/deploy, then reuse the standard module server harness."""
    if run_level != "full_model":
        pytest.fail("Ming exact numerics requires --run-level full_model (real local weights)")
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
    config = tmp_path_factory.mktemp("ming-numerics") / "ming_tp2_step.yaml"
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
    yield from iter_omni_server(request, run_level, omni_fixture_lock)


@pytest.fixture(scope="module")
def reference_url():
    image = Image.new("RGBA", (64, 64), (240, 240, 240, 255))
    draw = ImageDraw.Draw(image)
    draw.ellipse((8, 8, 40, 40), fill=(230, 30, 30, 255))
    draw.rectangle((20, 46, 55, 57), fill=(25, 80, 220, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _build_request(server, reference, index, *, layers=2, height=64, width=64, steps=None, guidance=None):
    layered = Path(server.model).name.endswith("-Layer")
    text = "Decompose into two layers: red circle and blue rectangle. " + PROMPTS[index] if layered else PROMPTS[index]
    content = (
        [{"type": "image_url", "image_url": {"url": reference}}, {"type": "text", "text": text}] if layered else text
    )
    extra = {
        "height": height,
        "width": width,
        "seed": index,
        "num_inference_steps": 3 + index % 3 if steps is None else steps,
    }
    if layered:
        extra["num_layers"] = layers
    if guidance is not None:
        extra["guidance_scale"] = guidance
    return {
        "model": server.model,
        "messages": [{"role": "user", "content": content}],
        "modalities": ["image"],
        "extra_body": extra,
        "expected_image_count": layers + 1 if layered else 1,
        "expected_image_mode": "RGBA",
    }


@pytest.mark.parametrize("omni_server", SERVER_CASES, indirect=True)
def test_eight_prompt_concurrent_matches_each_singleton_exactly(omni_server, online_client, reference_url):
    """Eight distinct prompts/seeds 0..7, steps 3/4/5; every RGBA frame is exact."""
    bodies = [_build_request(omni_server, reference_url, i) for i in range(8)]
    baselines = [online_client.send_diffusion_request(body)[0] for body in bodies]
    concurrent = online_client.send_diffusion_request(bodies)
    assert_diffusion_images_equal(concurrent, baselines)


@pytest.mark.parametrize("omni_server", SERVER_CASES, indirect=True)
def test_explicit_default_guidance_matches_implicit_exactly(omni_server, online_client, reference_url):
    layered = Path(omni_server.model).name.endswith("-Layer")
    baseline = online_client.send_diffusion_request(_build_request(omni_server, reference_url, 0))
    explicit = online_client.send_diffusion_request(
        _build_request(omni_server, reference_url, 0, guidance=2.0 if layered else 1.0)
    )
    assert_diffusion_images_equal(explicit, baseline)


@pytest.mark.parametrize("omni_server", SERVER_CASES, indirect=True)
def test_nondefault_guidance_rgba(omni_server, online_client, reference_url):
    layered = Path(omni_server.model).name.endswith("-Layer")
    online_client.send_diffusion_request(
        _build_request(omni_server, reference_url, 0, guidance=1.0 if layered else 2.0, steps=2)
    )


@pytest.mark.parametrize("omni_server", SERVER_CASES, indirect=True)
def test_nonsquare_80_by_64_rgba(omni_server, online_client, reference_url):
    online_client.send_diffusion_request(_build_request(omni_server, reference_url, 1, height=80, width=64, steps=2))


@pytest.mark.parametrize("omni_server", [SERVER_CASES[1]], indirect=True)
def test_layer_composite_plus_one_and_six_rgba_layers(omni_server, online_client, reference_url):
    for layers in (1, 6):
        online_client.send_diffusion_request(_build_request(omni_server, reference_url, 0, layers=layers, steps=1))


@pytest.mark.parametrize("omni_server", SERVER_CASES, indirect=True)
def test_shape_and_frame_state_retirement_preserves_exact_seed(omni_server, online_client, reference_url):
    """Changing shape/frame count must not poison replay of the original request."""
    body = _build_request(omni_server, reference_url, 0)
    baseline = online_client.send_diffusion_request(body)
    online_client.send_diffusion_request(_build_request(omni_server, reference_url, 1, height=80, width=64, steps=2))
    if Path(omni_server.model).name.endswith("-Layer"):
        for layers in (1, 6):
            online_client.send_diffusion_request(_build_request(omni_server, reference_url, 0, layers=layers, steps=1))
    repeated = online_client.send_diffusion_request(body)
    assert_diffusion_images_equal(repeated, baseline)

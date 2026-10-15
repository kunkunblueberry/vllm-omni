# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU regressions for ordered concurrent diffusion requests and exact media checks."""

import base64
import concurrent.futures
import io
import threading
import time
from types import SimpleNamespace

import pytest
from PIL import Image

from tests.helpers import client as client_module
from tests.helpers.assertions import assert_diffusion_images_equal, assert_diffusion_response
from tests.helpers.client import DiffusionResponse, OnlineOmniClient

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def diffusion_client(monkeypatch):
    # Replace only the SDK boundary; leave the real ThreadPool, decoder and assertions intact.
    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=None)))
    monkeypatch.setattr(client_module, "OpenAI", lambda **kwargs: sdk)
    return OnlineOmniClient(port=12345, run_level="full_model", log_stats=False)


def _completion(content, *, error=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, audio=None))],
        error=error,
    )


def _png_completion(color):
    image = Image.new("RGBA", (2, 3), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    return _completion([{"type": "image_url", "image_url": {"url": url}}])


def _request(seed):
    return {
        "model": "local-test-model",
        "messages": [{"role": "user", "content": f"independent prompt {seed}"}],
        "modalities": ["image"],
        "extra_body": {"seed": seed, "width": 2, "height": 3},
        "expected_image_count": 1,
        "expected_image_mode": "RGBA",
    }


def test_diffusion_list_real_threadpool_returns_input_order_after_reverse_completion(
    diffusion_client, monkeypatch, mocker
):
    """All eight workers must enter before any returns; decode in forced order 7..0."""
    barrier = threading.Barrier(8, timeout=10)
    release = [threading.Event() for _ in range(8)]
    release[7].set()
    worker_ids = set()
    lock = threading.Lock()
    processed = []
    completions = {i: _png_completion((i, 20, 30, 255)) for i in range(8)}

    def create(**kwargs):
        index = kwargs["extra_body"]["seed"]
        with lock:
            worker_ids.add(threading.get_ident())
        barrier.wait()
        if not release[index].wait(timeout=10):
            raise AssertionError(f"Worker {index} was not released")
        return completions[index]

    real_decode = diffusion_client._process_diffusion_response

    def decode(completion, *, wall_start):
        response = real_decode(completion, wall_start=wall_start)
        index = response.images[0].getpixel((0, 0))[0]
        processed.append(index)
        # Release the next worker only after this future was consumed by the caller.
        if index:
            release[index - 1].set()
        return response

    monkeypatch.setattr(diffusion_client.client.chat.completions, "create", create)
    monkeypatch.setattr(diffusion_client, "_process_diffusion_response", decode)
    executor_spy = mocker.spy(concurrent.futures, "ThreadPoolExecutor")
    try:
        responses = diffusion_client.send_diffusion_request([_request(i) for i in range(8)])
    finally:
        for event in release:
            event.set()
        barrier.abort()

    executor_spy.assert_called_once_with(max_workers=8)
    assert len(worker_ids) == 8
    assert processed == [7, 6, 5, 4, 3, 2, 1, 0]
    assert len(responses) == 8
    assert [response.images[0].getpixel((0, 0)) for response in responses] == [
        (0, 20, 30, 255),
        (1, 20, 30, 255),
        (2, 20, 30, 255),
        (3, 20, 30, 255),
        (4, 20, 30, 255),
        (5, 20, 30, 255),
        (6, 20, 30, 255),
        (7, 20, 30, 255),
    ]


def test_diffusion_list_propagates_worker_exception(diffusion_client, monkeypatch):
    barrier = threading.Barrier(2, timeout=10)
    failure = RuntimeError("worker transport failure")

    def create(**kwargs):
        barrier.wait()
        if kwargs["extra_body"]["seed"] == 1:
            raise failure
        return _png_completion((10, 20, 30, 255))

    monkeypatch.setattr(diffusion_client.client.chat.completions, "create", create)
    with pytest.raises(RuntimeError, match="worker transport failure") as excinfo:
        diffusion_client.send_diffusion_request([_request(0), _request(1)])
    assert excinfo.value is failure


def test_diffusion_list_propagates_response_assertion_failure(diffusion_client, monkeypatch):
    monkeypatch.setattr(
        diffusion_client.client.chat.completions,
        "create",
        lambda **kwargs: _png_completion((10, 20, 30, 40)),
    )
    wrong_size = {**_request(0), "extra_body": {"seed": 0, "width": 3, "height": 2}}
    with pytest.raises(AssertionError, match="Expected width=3"):
        diffusion_client.send_diffusion_request([wrong_size])


@pytest.mark.parametrize(
    "content, error",
    [
        pytest.param(
            [{"type": "image_url", "image_url": {"url": "data:image/png;base64,%%%invalid%%%"}}],
            None,
            id="invalid_base64",
        ),
        pytest.param([], None, id="empty_images"),
        pytest.param([], {"message": "explicit server error"}, id="error_object"),
    ],
)
def test_diffusion_decoder_marks_invalid_image_response_failed(diffusion_client, content, error):
    response = diffusion_client._process_diffusion_response(
        _completion(content, error=error), wall_start=time.perf_counter()
    )
    assert response.success is False
    assert response.images is None


@pytest.mark.parametrize(
    "content, error",
    [
        pytest.param(
            [{"type": "image_url", "image_url": {"url": "data:image/png;base64,%%%invalid%%%"}}],
            None,
            id="invalid_base64",
        ),
        pytest.param([], None, id="empty_images"),
        pytest.param([], {"message": "explicit server error"}, id="error_object"),
    ],
)
def test_diffusion_list_does_not_return_invalid_response(diffusion_client, monkeypatch, content, error):
    monkeypatch.setattr(
        diffusion_client.client.chat.completions,
        "create",
        lambda **kwargs: _completion(content, error=error),
    )
    with pytest.raises(AssertionError):
        diffusion_client.send_diffusion_request([_request(0)])


def test_diffusion_decoder_valid_rgba_pixels_pass(diffusion_client):
    response = diffusion_client._process_diffusion_response(
        _png_completion((10, 20, 30, 40)), wall_start=time.perf_counter()
    )
    assert_diffusion_response(response, _request(0), run_level="full_model")
    assert response.success is True
    assert response.images[0].mode == "RGBA"
    assert response.images[0].size == (2, 3)
    assert [response.images[0].getpixel((x, y)) for y in range(3) for x in range(2)] == [(10, 20, 30, 40)] * 6


def test_diffusion_images_equal_accepts_independently_constructed_rgba():
    actual = [DiffusionResponse(images=[Image.new("RGBA", (2, 3), (10, 20, 30, 40))], success=True)]
    expected = [DiffusionResponse(images=[Image.new("RGBA", (2, 3), (10, 20, 30, 40))], success=True)]
    assert_diffusion_images_equal(actual, expected)


def test_diffusion_images_equal_rejects_swapped_requests():
    expected = [
        DiffusionResponse(images=[Image.new("RGBA", (2, 3), (10, 20, 30, 40))]),
        DiffusionResponse(images=[Image.new("RGBA", (2, 3), (50, 60, 70, 80))]),
    ]
    actual = [
        DiffusionResponse(images=[Image.new("RGBA", (2, 3), (50, 60, 70, 80))]),
        DiffusionResponse(images=[Image.new("RGBA", (2, 3), (10, 20, 30, 40))]),
    ]
    with pytest.raises(AssertionError):
        assert_diffusion_images_equal(actual, expected)


def test_diffusion_images_equal_rejects_changed_pixel():
    image = Image.new("RGBA", (2, 3), (10, 20, 30, 40))
    image.putpixel((1, 2), (10, 20, 30, 41))
    actual = [DiffusionResponse(images=[image])]
    expected = [DiffusionResponse(images=[Image.new("RGBA", (2, 3), (10, 20, 30, 40))])]
    with pytest.raises(AssertionError):
        assert_diffusion_images_equal(actual, expected)


@pytest.mark.parametrize(
    "actual_specs",
    [
        pytest.param([], id="missing_frame"),
        pytest.param([("RGBA", (2, 3)), ("RGBA", (2, 3))], id="extra_frame"),
        pytest.param([("RGB", (2, 3))], id="rgb_mode"),
        # RGBX and RGBA have identical array shape/values; the mode is a separate contract.
        pytest.param([("RGBX", (2, 3))], id="same_array_different_mode"),
        pytest.param([("RGBA", (3, 2))], id="dimensions_swapped"),
    ],
)
def test_diffusion_images_equal_rejects_frame_mode_and_size_mismatch(actual_specs):
    actual = [DiffusionResponse(images=[Image.new(mode, size, (0, 0, 0)) for mode, size in actual_specs])]
    expected = [DiffusionResponse(images=[Image.new("RGBA", (2, 3), (0, 0, 0))])]
    with pytest.raises(AssertionError):
        assert_diffusion_images_equal(actual, expected)


def test_diffusion_images_equal_rejects_missing_request():
    actual = []
    expected = [DiffusionResponse(images=[Image.new("RGBA", (2, 3), (10, 20, 30, 40))])]
    with pytest.raises(AssertionError, match="Different response counts"):
        assert_diffusion_images_equal(actual, expected)

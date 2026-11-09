# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One derivation of a Noisy PP request's pool key and lifecycle."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.experimental.ar_diffusion.noisy_lifecycle import noisy_request_lifecycle

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_an_explicit_session_id_is_the_persistent_key() -> None:
    lifecycle = noisy_request_lifecycle("req-1", SimpleNamespace(extra_args={"session_id": "s1"}))

    assert lifecycle.kv_id == "s1"
    assert lifecycle.persist is True
    assert lifecycle.reset is False
    assert lifecycle.close_session is False


def test_without_a_session_the_request_id_is_ephemeral() -> None:
    lifecycle = noisy_request_lifecycle("req-1", SimpleNamespace(extra_args={}))

    assert lifecycle.kv_id == "req-1"
    assert lifecycle.persist is False


def test_an_empty_session_id_is_not_explicit() -> None:
    lifecycle = noisy_request_lifecycle("req-1", SimpleNamespace(extra_args={"session_id": ""}))

    assert lifecycle.kv_id == "req-1"
    assert lifecycle.persist is False


def test_reset_and_close_are_carried_through() -> None:
    sampling = SimpleNamespace(extra_args={"session_id": "s1", "reset": True, "close_session": True})

    lifecycle = noisy_request_lifecycle("req-1", sampling)

    assert lifecycle.reset is True
    assert lifecycle.close_session is True
    assert lifecycle.persist is False, "a closing request does not keep its history"


def test_the_streaming_tick_wins_over_extra_args() -> None:
    """The tick is the authoritative session state; extra_args must not override it."""
    tick = SimpleNamespace(session_id="tick-session", reset=False, close_session=False)
    sampling = SimpleNamespace(extra_args={"session_id": "extra-session"})

    lifecycle = noisy_request_lifecycle("req-1", sampling, tick=tick)

    assert lifecycle.kv_id == "tick-session"
    assert lifecycle.persist is True


def test_a_tick_close_is_ephemeral() -> None:
    tick = SimpleNamespace(session_id="tick-session", reset=False, close_session=True)

    lifecycle = noisy_request_lifecycle("req-1", SimpleNamespace(extra_args={}), tick=tick)

    assert lifecycle.persist is False
    assert lifecycle.close_session is True


def test_the_fallback_key_covers_a_request_without_an_id() -> None:
    lifecycle = noisy_request_lifecycle("", SimpleNamespace(extra_args={}), fallback="default")

    assert lifecycle.kv_id == "default"
    assert lifecycle.persist is False


def test_the_runner_and_the_pipeline_derive_the_same_key() -> None:
    """The regression this guards: two helpers could disagree on one request."""
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.experimental.ar_diffusion.runner import ARDiffusionModelRunner
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    req = OmniDiffusionRequest(
        prompt="x",
        request_id="req-1",
        sampling_params=OmniDiffusionSamplingParams(extra_args={"session_id": "s1", "reset": True}),
    )

    from_runner = ARDiffusionModelRunner._noisy_kv_id(req)
    direct = noisy_request_lifecycle("req-1", req.sampling_params)

    assert from_runner == (direct.kv_id, direct.persist, direct.reset, direct.close_session)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared allocation-retention contract used by paged and Noisy KV backends."""

from __future__ import annotations

import pytest

from vllm_omni.experimental.ar_diffusion.kv_cache.lifecycle import release_unretained

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_legacy_noisy_imports_are_the_same_implementation_objects() -> None:
    """Existing models keep their imports; moving code must not wrap or fork classes."""
    from vllm_omni.experimental.ar_diffusion.kv_cache import noisy
    from vllm_omni.experimental.ar_diffusion.kv_cache.noisy_context import NoisyLayerContext
    from vllm_omni.experimental.ar_diffusion.kv_cache.noisy_manager import NoisyKVCache
    from vllm_omni.experimental.ar_diffusion.kv_cache.noisy_pool import VersionPool
    from vllm_omni.experimental.ar_diffusion.kv_cache.noisy_retention import clean_history_cells, prior_history_in_pool
    from vllm_omni.experimental.ar_diffusion.kv_cache.noisy_spec import ARDiffusionNoisyKVSpec
    from vllm_omni.experimental.ar_diffusion.kv_cache.noisy_state import NoisyKVState
    from vllm_omni.experimental.ar_diffusion.kv_cache.noisy_transport import NoisyKVTransport

    assert noisy.ARDiffusionNoisyKVSpec is ARDiffusionNoisyKVSpec
    assert noisy.VersionPool is VersionPool
    assert noisy.NoisyLayerContext is NoisyLayerContext
    assert noisy.NoisyKVTransport is NoisyKVTransport
    assert noisy.NoisyKVCache is NoisyKVCache
    assert noisy.NoisyKVState is NoisyKVState
    assert noisy.clean_history_cells is clean_history_cells
    assert noisy.prior_history_in_pool is prior_history_in_pool


def test_release_unretained_preserves_retained_and_releases_only_owned() -> None:
    """A backend can retire its old allocations without touching another owner."""
    allocations = {"keep-a": object(), "drop-a": object(), "drop-b": object()}
    release_calls: list[str] = []

    def release(key: str) -> None:
        release_calls.append(key)
        allocations.pop(key)

    release_unretained(allocations, ("keep-a", "not-owned"), release)

    assert set(allocations) == {"keep-a"}
    assert release_calls == ["drop-a", "drop-b"]


def test_release_unretained_runs_every_prepare_before_any_release() -> None:
    """Noisy can retire all pending transfers before returning any version slot."""
    allocations = {"drop-a": object(), "keep": object(), "drop-b": object()}
    events: list[tuple[str, str]] = []

    def prepare(key: str) -> None:
        events.append(("prepare", key))

    def release(key: str) -> None:
        events.append(("release", key))
        allocations.pop(key)

    release_unretained(allocations, ("keep",), release, before_release=prepare)

    assert events == [
        ("prepare", "drop-a"),
        ("prepare", "drop-b"),
        ("release", "drop-a"),
        ("release", "drop-b"),
    ]
    assert set(allocations) == {"keep"}


def test_release_unretained_does_not_release_when_prepare_fails() -> None:
    """A failed visibility/communication fence must leave all slots allocated."""
    allocations = {"drop": object(), "keep": object()}
    released: list[str] = []

    def prepare(_key: str) -> None:
        raise RuntimeError("pending transfer is not complete")

    with pytest.raises(RuntimeError, match="pending transfer"):
        release_unretained(allocations, ("keep",), released.append, before_release=prepare)

    assert set(allocations) == {"drop", "keep"}
    assert released == []

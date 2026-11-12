# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Session capacity: the scheduler's grant and the worker's report are one loop."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.experimental.ar_diffusion.kv_cache.noisy import ARDiffusionNoisyKVSpec, NoisyKVCache
from vllm_omni.experimental.ar_diffusion.reservation import (
    CapacityDemand,
    CapacityUsage,
    ReservationDeniedError,
    ReservationOvercommitError,
    SessionCapacityLedger,
)
from vllm_omni.experimental.ar_diffusion.runner import ARDiffusionModelRunner
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_ledger_reconcile_accepts_usage_within_the_grant() -> None:
    ledger = SessionCapacityLedger(budget_bytes=1000)
    ledger.reserve("s1", CapacityDemand(versions=6, bytes=400))

    reported = ledger.reconcile("s1", CapacityUsage(versions=4, bytes=300))

    assert reported == CapacityUsage(versions=4, bytes=300)
    assert ledger.usage("s1") == CapacityUsage(versions=4, bytes=300)
    assert ledger.overcommit("s1") is None
    assert ledger.held_bytes("s1") == 400


def test_ledger_reconcile_fails_closed_on_overcommit() -> None:
    ledger = SessionCapacityLedger(budget_bytes=1000)
    ledger.reserve("s1", CapacityDemand(versions=6, bytes=400))

    with pytest.raises(ReservationOvercommitError):
        ledger.reconcile("s1", CapacityUsage(versions=7, bytes=500))

    assert ledger.overcommit("s1") == CapacityUsage(versions=7, bytes=500)
    assert ledger.usage("s1") is None


def test_release_forgets_usage_and_overcommit() -> None:
    ledger = SessionCapacityLedger(budget_bytes=1000)
    ledger.reserve("s1", CapacityDemand(versions=6, bytes=400))
    with pytest.raises(ReservationOvercommitError):
        ledger.reconcile("s1", CapacityUsage(versions=7, bytes=500))

    ledger.release("s1")

    assert ledger.usage("s1") is None
    assert ledger.overcommit("s1") is None
    assert ledger.free_bytes() == 1000


def test_a_demand_beyond_the_budget_is_denied() -> None:
    ledger = SessionCapacityLedger(budget_bytes=100)

    with pytest.raises(ReservationDeniedError):
        ledger.reserve("s1", CapacityDemand(versions=6, bytes=400))


def test_a_demand_is_absolute_and_charges_only_the_growth() -> None:
    """The caller passes the holding it needs, not a delta on top of it."""
    ledger = SessionCapacityLedger(budget_bytes=1000)
    first = ledger.reserve("s1", CapacityDemand(versions=6, bytes=400))
    assert first.incremental is True
    assert ledger.held_bytes("s1") == 400

    # The same absolute holding again costs nothing.
    again = ledger.reserve("s1", CapacityDemand(versions=6, bytes=400))
    assert again.incremental is False
    assert ledger.held_bytes("s1") == 400

    # A wider window charges the difference only.
    grown = ledger.reserve("s1", CapacityDemand(versions=8, bytes=700))
    assert grown.incremental is True
    assert ledger.held_bytes("s1") == 700
    assert ledger.free_bytes() == 300

    with pytest.raises(ReservationDeniedError, match="needs 700 more bytes"):
        ledger.reserve("s1", CapacityDemand(versions=10, bytes=1400))


def test_reconcile_fails_closed_when_versions_exceed_the_grant() -> None:
    """Versions are part of the grant, so bytes-only checking is not enough."""
    ledger = SessionCapacityLedger(budget_bytes=1000)
    ledger.reserve("s1", CapacityDemand(versions=2, bytes=400))

    with pytest.raises(ReservationOvercommitError, match="versions"):
        ledger.reconcile("s1", CapacityUsage(versions=3, bytes=100))

    assert ledger.overcommit("s1") == CapacityUsage(versions=3, bytes=100)
    assert ledger.usage("s1") is None


def test_reconcile_fails_closed_when_bytes_exceed_the_grant() -> None:
    ledger = SessionCapacityLedger(budget_bytes=1000)
    ledger.reserve("s1", CapacityDemand(versions=6, bytes=400))

    with pytest.raises(ReservationOvercommitError, match="bytes"):
        ledger.reconcile("s1", CapacityUsage(versions=2, bytes=401))


def _spec(num_layers: int = 2) -> ARDiffusionNoisyKVSpec:
    return ARDiffusionNoisyKVSpec(
        num_layers=num_layers,
        num_kv_heads=1,
        head_size=4,
        block_size=4,
        max_chunk_tokens=4,
        max_history_chunks=1,
    )


def _cache(*, num_layers: int = 2) -> NoisyKVCache:
    return NoisyKVCache(
        _spec(num_layers), dtype=torch.float32, device=torch.device("cpu"), layer_groups=1, max_batch_size=1
    )


def _runner(ledger: SessionCapacityLedger | None, usage: tuple[int, int] | None = None) -> ARDiffusionModelRunner:
    runner = object.__new__(ARDiffusionModelRunner)
    runner.noisy_kv_cache = _cache() if usage is None else SimpleNamespace(session_usage=lambda _kv_id: usage)
    runner._capacity_ledger = ledger
    runner._perf_e2e_times = []
    return runner


def _runner_with_pool(*, max_batch_size: int = 1) -> ARDiffusionModelRunner:
    """A runner whose Noisy pool was built the way preallocation builds it.

    Nothing is injected: the ledger under test is the one the worker installed
    itself, so the admission, reporting, and release paths are the production ones.
    """
    runner = object.__new__(ARDiffusionModelRunner)
    runner.od_config = SimpleNamespace(
        dtype=torch.float32,
        ar_diffusion_stage_config={"max_batch_size": max_batch_size},
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
    )
    runner.device = torch.device("cpu")
    runner.pipeline = None
    runner.ar_diffusion_kv_config = SimpleNamespace(superpage_tokens=0)
    runner._capacity_ledger = None
    runner._perf_e2e_times = []
    runner._preallocate_noisy_kv(SimpleNamespace(ar_diffusion_noisy_kv_spec=_spec), available_bytes=1 << 30)
    return runner


class _RecordingLedger(SessionCapacityLedger):
    """Keeps every reconciliation, so a report made before a release stays visible."""

    def __init__(self, *, budget_bytes: int) -> None:
        super().__init__(budget_bytes=budget_bytes)
        self.reconciled: list[tuple[str, CapacityUsage]] = []

    def reconcile(self, session_id: str, usage: CapacityUsage) -> CapacityUsage:
        self.reconciled.append((session_id, usage))
        return super().reconcile(session_id, usage)


def test_session_usage_counts_each_pool_key_once() -> None:
    """A key holds one block's layer; a full pool reports exactly its reservation."""
    cache = _cache(num_layers=2)
    for layer in range(cache.spec.num_layers):
        for slot in range(cache.capacity):
            cache.pool.alloc(("s1", slot, 1, layer))

    versions, nbytes = cache.session_usage("s1")

    assert versions == cache.capacity * cache.spec.num_layers
    assert nbytes == cache.reserved_bytes
    assert cache.session_usage("other") == (0, 0)


def test_preallocation_installs_a_ledger_over_the_pool_budget() -> None:
    """The pool is the budget: grants are charged against the bytes it reserved."""
    runner = _runner_with_pool()
    ledger = runner._capacity_ledger
    cache = runner.noisy_kv_cache

    assert ledger is not None, "the pool was built without a ledger guarding it"
    assert ledger.budget_bytes == cache.reserved_bytes
    # One session is granted the window the pool sized for it, so the budget holds
    # exactly the sessions the pool was sized for.
    assert cache.session_grant_bytes * cache.max_batch_size == cache.reserved_bytes
    assert cache.session_grant_versions * cache.bytes_per_version == cache.session_grant_bytes


def test_admission_grants_a_persistent_session_once() -> None:
    runner = _runner_with_pool(max_batch_size=2)
    ledger = runner._capacity_ledger
    cache = runner.noisy_kv_cache

    runner._prepare_noisy_kv_lifecycles([("s1", True, False, False)])

    assert ledger.grant("s1") is not None, "a persistent session was not admitted"
    assert ledger.grant("s1").versions == cache.session_grant_versions
    assert ledger.held_bytes("s1") == cache.session_grant_bytes

    # A continuation re-admits the same absolute holding: it costs nothing.
    runner._prepare_noisy_kv_lifecycles([("s1", True, False, False)])

    assert ledger.held_bytes("s1") == cache.session_grant_bytes
    assert ledger.free_bytes() == cache.reserved_bytes - cache.session_grant_bytes


def test_admission_leaves_ephemeral_and_closing_keys_ungranted() -> None:
    """Only a key that keeps history past the call holds capacity."""
    runner = _runner_with_pool(max_batch_size=2)
    ledger = runner._capacity_ledger

    runner._prepare_noisy_kv_lifecycles(
        [("", True, False, False), ("ephemeral", False, False, False), ("last", True, False, True)]
    )

    assert ledger._grants == {}
    assert ledger.used_bytes() == 0


def test_admission_refuses_a_session_the_pool_cannot_hold() -> None:
    """Refusing here is the pool's own boundary, reached before any version is written."""
    runner = _runner_with_pool(max_batch_size=1)
    ledger = runner._capacity_ledger

    with pytest.raises(ReservationDeniedError):
        runner._prepare_noisy_kv_lifecycles([("s1", True, False, False), ("s2", True, False, False)])

    assert ledger.grant("s2") is None
    assert ledger.grant("s1") is None, "a refused wave left an earlier key holding a window"
    assert ledger.used_bytes() == 0


def test_reset_returns_the_grant_before_re_admitting_it() -> None:
    runner = _runner_with_pool(max_batch_size=1)
    ledger = runner._capacity_ledger
    cache = runner.noisy_kv_cache
    cache.pool.alloc(("s1", 0, 1, 0))
    runner._prepare_noisy_kv_lifecycles([("s1", True, False, False)])

    runner._prepare_noisy_kv_lifecycles([("s1", True, True, False)])

    assert cache.session_usage("s1") == (0, 0), "reset kept the session's history"
    assert ledger.grant("s1") is not None, "reset left the session unadmitted"
    assert ledger.used_bytes() == cache.session_grant_bytes, "reset charged the window twice"


def test_a_failed_forward_returns_the_session_grant(monkeypatch) -> None:
    from vllm_omni.diffusion.worker import diffusion_model_runner as dmr

    runner = _runner_with_pool(max_batch_size=1)
    ledger = runner._capacity_ledger
    runner._ar_diffusion_noisy_kv_capability = SimpleNamespace(bind_ar_diffusion_noisy_kv=lambda _state: nullcontext())

    def _forward(self, req, kv_prefetch_job=None, diffusion_kv_metadata=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(dmr.DiffusionModelRunner, "execute_model", _forward)
    req = OmniDiffusionRequest(
        prompt="x", request_id="r1", sampling_params=OmniDiffusionSamplingParams(extra_args={"session_id": "s1"})
    )

    with pytest.raises(RuntimeError, match="boom"):
        runner.execute_model(req)

    assert ledger.grant("s1") is None, "a failed forward stranded its grant"
    assert ledger.used_bytes() == 0


def test_the_runner_reports_worker_usage_after_a_batch() -> None:
    ledger = SessionCapacityLedger(budget_bytes=1000)
    ledger.reserve("s1", CapacityDemand(versions=6, bytes=400))
    runner = _runner(ledger, (3, 200))

    runner._report_session_capacity("s1")

    assert ledger.usage("s1") == CapacityUsage(versions=3, bytes=200)
    assert ledger.overcommit("s1") is None


def test_a_worker_overcommit_raises_and_leaves_teardown_to_the_caller() -> None:
    """The report fails the call; dropping history and the grant is teardown's job."""
    runner = _runner(SessionCapacityLedger(budget_bytes=10**9))
    cache = runner.noisy_kv_cache
    cache.pool.alloc(("s1", 0, 1, 0))
    cache.pool.alloc(("s1", 1, 1, 0))
    cache.pool.alloc(("s2", 0, 1, 0))
    runner._capacity_ledger.reserve("s1", CapacityDemand(versions=1, bytes=cache.bytes_per_version))

    with pytest.raises(ReservationOvercommitError):
        runner._report_session_capacity("s1")

    assert runner._capacity_ledger.overcommit("s1") == CapacityUsage(versions=2, bytes=2 * cache.bytes_per_version)
    assert cache.session_usage("s1") != (0, 0), "the report path must not release storage"


def test_overcommit_teardown_returns_the_grant_and_keeps_the_diagnosis() -> None:
    """One teardown contract settles history, grant and diagnosis together."""
    ledger = SessionCapacityLedger(budget_bytes=10**9)
    runner = _runner(ledger)
    cache = runner.noisy_kv_cache
    cache.pool.alloc(("s1", 0, 1, 0))
    cache.pool.alloc(("s1", 1, 1, 0))
    cache.pool.alloc(("s2", 0, 1, 0))
    ledger.reserve("s1", CapacityDemand(versions=1, bytes=cache.bytes_per_version))
    with pytest.raises(ReservationOvercommitError):
        runner._report_session_capacity("s1")
    overcommit = ledger.overcommit("s1")
    assert overcommit is not None

    runner.release_noisy_kv("s1", keep_diagnostics=True)

    assert cache.session_usage("s1") == (0, 0), "an overcommitted session kept its history"
    assert ledger.grant("s1") is None, "the grant was not returned"
    assert ledger.used_bytes() == 0, "the budget stayed charged to a released session"
    assert ledger.overcommit("s1") == overcommit, "the overcommit diagnosis was lost"
    assert cache.pool.has(("s2", 0, 1, 0)), "another session's history was dropped"


def test_teardown_reporting_never_raises() -> None:
    """Releasing a key must not turn a drifted grant into a second failure."""
    ledger = SessionCapacityLedger(budget_bytes=1000)
    ledger.reserve("s1", CapacityDemand(versions=6, bytes=400))
    runner = _runner(ledger, (9, 900))

    runner._report_session_capacity("s1", fail_closed=False)

    assert ledger.overcommit("s1") == CapacityUsage(versions=9, bytes=900)


def test_releasing_an_unreserved_key_leaves_no_ledger_entry() -> None:
    """Ephemeral request ids are released after every forward; none may accumulate."""
    ledger = SessionCapacityLedger(budget_bytes=1000)
    runner = _runner(ledger)

    runner.release_noisy_kv("ephemeral-1")

    assert ledger.usage("ephemeral-1") is None
    assert ledger._usage == {}


def test_releasing_a_reserved_key_reports_the_truth_then_returns_the_grant() -> None:
    """The empty holding is recorded first, so a release never hides a drift."""
    ledger = _RecordingLedger(budget_bytes=10**9)
    runner = _runner(ledger)
    runner.noisy_kv_cache.pool.alloc(("s1", 0, 1, 0))
    ledger.reserve("s1", CapacityDemand(versions=1, bytes=runner.noisy_kv_cache.bytes_per_version))

    runner.release_noisy_kv("s1")

    assert ledger.reconciled == [("s1", CapacityUsage(versions=0, bytes=0))]
    assert ledger.grant("s1") is None, "a released key kept its grant"
    assert ledger.free_bytes() == 10**9, "a released key stranded the budget it held"


def test_a_batch_wave_fails_closed_and_returns_every_grant(monkeypatch) -> None:
    from vllm_omni.diffusion.worker import diffusion_model_runner as dmr

    ledger = SessionCapacityLedger(budget_bytes=10**9)
    runner = _runner(ledger)
    cache = runner.noisy_kv_cache
    runner._ar_diffusion_noisy_kv_capability = SimpleNamespace(bind_ar_diffusion_noisy_kv=lambda _state: nullcontext())
    ledger.reserve("s1", CapacityDemand(versions=1, bytes=cache.bytes_per_version))
    ledger.reserve("s2", CapacityDemand(versions=1, bytes=cache.bytes_per_version))

    def _forward(self, scheduler_output, od_config):
        cache.pool.alloc(("s1", 0, 1, 0))
        cache.pool.alloc(("s2", 0, 1, 0))
        cache.pool.alloc(("s2", 1, 1, 0))
        return "out"

    monkeypatch.setattr(dmr.DiffusionModelRunner, "execute_model_batch", _forward)

    def _req(request_id: str, **extra) -> SimpleNamespace:
        return SimpleNamespace(
            req=OmniDiffusionRequest(
                prompt="x", request_id=request_id, sampling_params=OmniDiffusionSamplingParams(extra_args=extra)
            )
        )

    wave = SimpleNamespace(scheduled_new_reqs=[_req("r1", session_id="s1"), _req("r2", session_id="s2"), _req("r3")])
    with pytest.raises(ReservationOvercommitError):
        runner.execute_model_batch(wave, od_config=SimpleNamespace())

    # The wave fails closed as a whole: every key of this call is torn down and
    # every grant returns, so one session's overcommit cannot strand capacity.
    assert ledger.usage("s1") is None, "a failed wave kept a half-settled session"
    assert ledger.grant("s1") is None and ledger.grant("s2") is None
    assert ledger.used_bytes() == 0, "a failed wave stranded its grants"
    assert ledger.overcommit("s2") is not None, "the overcommit diagnosis was lost"
    assert cache.session_usage("s1") == (0, 0)
    assert cache.session_usage("s2") == (0, 0)


def test_without_a_ledger_nothing_is_reported() -> None:
    runner = _runner(None, (9, 900))

    runner._report_session_capacity("s1")


def test_a_closing_forward_reports_the_truth_without_raising(monkeypatch, caplog) -> None:
    """Close reports before the drop, so an overcommit is visible but never fatal."""
    from vllm_omni.diffusion.worker import diffusion_model_runner as dmr

    ledger = SessionCapacityLedger(budget_bytes=10**9)
    runner = _runner(ledger)
    cache = runner.noisy_kv_cache
    runner._ar_diffusion_noisy_kv_capability = SimpleNamespace(bind_ar_diffusion_noisy_kv=lambda _state: nullcontext())
    ledger.reserve("s1", CapacityDemand(versions=1, bytes=cache.bytes_per_version))

    def _forward(self, req, kv_prefetch_job=None, diffusion_kv_metadata=None):
        cache.pool.alloc(("s1", 0, 1, 0))
        cache.pool.alloc(("s1", 1, 1, 0))  # one version more than the grant
        return "out"

    monkeypatch.setattr(dmr.DiffusionModelRunner, "execute_model", _forward)
    req = OmniDiffusionRequest(
        prompt="x",
        request_id="r1",
        sampling_params=OmniDiffusionSamplingParams(extra_args={"session_id": "s1", "close_session": True}),
    )

    with caplog.at_level("WARNING"):
        assert runner.execute_model(req) == "out"

    assert "exceeded its capacity grant" in caplog.text, "an overcommit at close went unreported"
    # The close then drops the key and returns its grant: the budget ends where
    # the worker is, with nothing reserved for history that no longer exists.
    assert cache.session_usage("s1") == (0, 0)
    assert ledger.grant("s1") is None
    assert ledger.used_bytes() == 0

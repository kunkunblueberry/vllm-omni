# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Abort terminals and step retirement must not depend on future admission.

Real client abort, Engine loop, scheduler, executor dispatch and runner execute.
Only checkpoint kernels/native transport are isolated by the existing helpers.
"""

import asyncio
import queue
import threading
from collections import Counter
from types import SimpleNamespace

import pytest

from tests.diffusion.models.ming_image.test_batch_step_contract import _owned_request, _runner, _wave_pipeline
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine, DiffusionExecutionMode
from vllm_omni.diffusion.executor.uniproc_executor import UniProcDiffusionExecutor
from vllm_omni.diffusion.inline_stage_diffusion_client import InlineStageDiffusionClient
from vllm_omni.diffusion.models.ming_image.pipeline import get_ming_image_pre_process_func
from vllm_omni.diffusion.sched import RequestScheduler, StepScheduler
from vllm_omni.diffusion.sched.interface import DiffusionRequestStatus
from vllm_omni.diffusion.worker.utils import BatchRunnerOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_TIMEOUT = 3.0


class _ObservedStepScheduler(StepScheduler):
    def __init__(self, after_first_wave, first_updated, release):
        super().__init__()
        self.after_first_wave = after_first_wave
        self.first_updated = first_updated
        self.release = release

    def update_from_output(self, sched_output, output):
        finished = super().update_from_output(sched_output, output)
        if sched_output.step_id == 0 and self.after_first_wave:
            self.first_updated.set()
            assert self.release.wait(_TIMEOUT), "wave1 observer barrier was not released"
        return finished


class _ObservedAbortEngine(DiffusionEngine):
    def _finalize_finished_request(self, request_id, *args, **kwargs):
        self.finalizations[request_id] += 1
        return super()._finalize_finished_request(request_id, *args, **kwargs)

    def _put_output(self, request_id, output):
        super()._put_output(request_id, output)
        self.delivered.append((request_id, output))
        if output.finished:
            self.terminals[request_id].set()

    def _emit_outputs(self, finished_ids, scheduled_request_ids, runner_output, **kwargs):
        super()._emit_outputs(finished_ids, scheduled_request_ids, runner_output, **kwargs)
        if not scheduled_request_ids:
            self.retired.set()


def _abort_engine(monkeypatch, timing, peers):
    pipeline, _ = _wave_pipeline(monkeypatch)
    runner = _runner(monkeypatch, pipeline)
    monkeypatch.setattr(runner, "_sample_peak_memory_mb", lambda: 0.0)
    first_updated, entered, release, retired = (threading.Event() for _ in range(4))
    scheduler = _ObservedStepScheduler(timing == "idle", first_updated, release)
    scheduler.initialize(SimpleNamespace(max_num_seqs=2))
    executor = UniProcDiffusionExecutor.__new__(UniProcDiffusionExecutor)
    executor._closed = False
    executor.driver_worker = SimpleNamespace(model_runner=runner)
    calls = []

    def rpc(method, *, args, **kwargs):
        assert method == "execute_stepwise"
        scheduled = args[0]
        calls.append((list(scheduled.scheduled_request_ids), set(scheduled.finished_req_ids)))
        if timing == "midwave" and scheduled.step_id == 1:
            entered.set()
            assert release.wait(_TIMEOUT), "wave2 transport barrier was not released"
        return runner.execute_stepwise(scheduled)

    monkeypatch.setattr(executor, "collective_rpc", rpc)
    monkeypatch.setattr(executor, "shutdown", lambda: None)
    engine = _ObservedAbortEngine.__new__(_ObservedAbortEngine)
    engine.executor = executor
    engine.scheduler = scheduler
    engine.od_config = SimpleNamespace(streaming_output=False)
    engine.pre_process_func = get_ming_image_pre_process_func(
        SimpleNamespace(step_execution=True, tf_model_config=SimpleNamespace(multi_frame_output=False))
    )
    engine.execution_mode = DiffusionExecutionMode.STEP_BATCH
    engine.step_execution = True
    engine.supports_request_batch = False
    engine.dp_concurrent = False
    engine.execute_fn = executor.execute_step
    engine._rpc_lock = threading.RLock()
    engine._cv = threading.Condition(engine._rpc_lock)
    engine._out_streams = {}
    engine._closed = engine._shutting_down = engine._shutdown_complete = False
    engine._request_cancellations = None
    engine.abort_queue = queue.Queue()
    engine._rpc_queue = queue.Queue()
    engine._scheduling_paused = False
    engine.main_loop = None
    engine.stop_event = threading.Event()
    engine._loop_started = True
    engine.finalizations = Counter()
    engine.delivered = []
    engine.retired = retired
    engine.terminals = {name: threading.Event() for name in peers}
    errors = []

    def busy_loop():
        try:
            engine._busy_loop()
        except BaseException as exc:
            errors.append(exc)

    engine.worker_thread = threading.Thread(target=busy_loop, daemon=True)
    # Admit before starting the loop so the peer scenario shares its first wave.
    for index, name in enumerate(peers):
        engine.add_request(_owned_request(name, offset=index * 2, seed=42 + index, steps=3))
    client = InlineStageDiffusionClient.__new__(InlineStageDiffusionClient)
    client._engine = engine
    return SimpleNamespace(
        engine=engine,
        scheduler=scheduler,
        runner=runner,
        client=client,
        calls=calls,
        ready=first_updated if timing == "idle" else entered,
        release=release,
        retired=retired,
        errors=errors,
    )


@pytest.mark.parametrize(("timing", "peers"), [("idle", ("a",)), ("midwave", ("a",)), ("midwave", ("a", "b"))])
def test_client_abort_retires_step_rows_without_following_request(monkeypatch, timing, peers):
    case = _abort_engine(monkeypatch, timing, peers)
    engine = case.engine
    engine.worker_thread.start()
    try:
        assert case.ready.wait(_TIMEOUT), f"first-wave cache not established: {case.errors}"
        assert set(case.runner.state_cache) == set(peers)
        assert case.runner.input_batch is not None
        assert case.runner.state_cache["a"].step_index == 1
        # Real client entry; duplicate IDs must still produce one terminal.
        asyncio.run(case.client.abort_requests_async(["a", "a"]))
        case.release.set()
        assert engine.terminals["a"].wait(_TIMEOUT), f"abort terminal missing: {case.errors}"
        if "b" in peers:
            assert engine.terminals["b"].wait(_TIMEOUT), f"survivor terminal missing: {case.errors}"
        assert case.retired.wait(_TIMEOUT), f"last worker retirement needs another admission: {case.errors}"
        assert case.errors == []
        assert not engine._closed and engine.worker_thread.is_alive()
        assert case.runner.state_cache == {} and case.runner.input_batch is None
        assert case.scheduler._request_progress == {} and case.scheduler._request_states == {}
        assert not case.scheduler.has_requests()
        assert case.scheduler.pending_finished_request_ids() == set()
        assert case.scheduler._finished_req_ids == set()
        assert engine.finalizations == Counter({name: 1 for name in peers})
        terminals = [(name, output) for name, output in engine.delivered if output.finished]
        assert Counter(name for name, _ in terminals) == Counter({name: 1 for name in peers})
        assert next(output for name, output in terminals if name == "a").aborted
        if "b" in peers:
            survivor = next(output for name, output in terminals if name == "b")
            assert not survivor.aborted and survivor.error is None and survivor.output is not None
            assert [ids for ids, _ in case.calls if ids] == [["a", "b"], ["a", "b"], ["b"]]
        else:
            assert sum(bool(ids) for ids, _ in case.calls) == (1 if timing == "idle" else 2)
        assert any("a" in finished for _, finished in case.calls), "abort ID never reached worker"

        async def consume(name):
            return [output async for output in engine.get_output_stream(name)]

        for name in peers:
            assert engine._out_streams[name].qsize() == 1
            outputs = asyncio.run(consume(name))
            assert len(outputs) == 1 and outputs[0].finished
            assert outputs[0].aborted is (name == "a")
        assert engine._out_streams == {}
    finally:
        case.release.set()
        with engine._cv:
            engine.stop_event.set()
            engine._cv.notify_all()
        engine.worker_thread.join(_TIMEOUT)
        assert not engine.worker_thread.is_alive(), "busy loop did not stop"


def test_failed_step_retirement_stops_workers_before_scheduler_close(monkeypatch):
    case = _abort_engine(monkeypatch, "idle", ("a",))
    engine = case.engine
    rpc = engine.executor.collective_rpc
    scheduler_close = case.scheduler.close
    closed = []

    def fail_retirement(method, *, args, **kwargs):
        if not args[0].scheduled_request_ids:
            raise RuntimeError("injected retirement failure")
        return rpc(method, args=args, **kwargs)

    def close_scheduler():
        closed.append("scheduler")
        scheduler_close()

    monkeypatch.setattr(engine.executor, "collective_rpc", fail_retirement)
    monkeypatch.setattr(engine.executor, "shutdown", lambda: closed.append("worker"))
    monkeypatch.setattr(case.scheduler, "close", close_scheduler)
    engine.worker_thread.start()
    try:
        assert case.ready.wait(_TIMEOUT)
        asyncio.run(case.client.abort_requests_async(["a"]))
        case.release.set()
        assert engine.stop_event.wait(_TIMEOUT), "cleanup failure did not stop the engine"
        engine.worker_thread.join(_TIMEOUT)
        assert not engine.worker_thread.is_alive() and case.errors == []
        assert engine._closed and engine._shutdown_complete
        assert closed == ["worker", "scheduler"]
        assert engine.finalizations == Counter()
        assert engine._out_streams["a"].qsize() == 1
        output = engine._out_streams["a"].get_nowait()
        assert output.finished and "injected retirement failure" in output.error
    finally:
        case.release.set()
        with engine._cv:
            engine.stop_event.set()
            engine._cv.notify_all()
        engine.worker_thread.join(_TIMEOUT)
        assert not engine.worker_thread.is_alive()


def test_step_retirement_readiness_is_distinct_from_unconsumed_terminal():
    scheduler = StepScheduler()
    scheduler.initialize(SimpleNamespace(max_num_seqs=1))
    scheduler.add_request(_owned_request("a", steps=3))
    scheduler.schedule()
    scheduler.finish_requests("a", DiffusionRequestStatus.FINISHED_ABORTED)
    assert scheduler.pending_finished_request_ids() == {"a"}
    assert scheduler.has_requests()
    scheduler.pop_request_state("a")
    assert scheduler.pending_finished_request_ids() == set()
    assert scheduler.has_requests(), "worker cleanup remains pending after terminal consumption"
    output = scheduler.schedule()
    assert output.finished_req_ids == {"a"} and output.scheduled_request_ids == []
    assert scheduler.update_from_output(output, BatchRunnerOutput.from_list([])) == set()
    assert not scheduler.has_requests()
    # Request-mode readiness contract is not changed by step-only retirement.
    request_scheduler = RequestScheduler()
    request_scheduler.initialize(SimpleNamespace(max_num_seqs=1))
    request_scheduler.add_request(_owned_request("request-mode"))
    request_scheduler.finish_requests("request-mode", DiffusionRequestStatus.FINISHED_ABORTED)
    assert not request_scheduler.has_requests()


def test_step_pause_preserves_request_only_api_boundary():
    engine = DiffusionEngine.__new__(DiffusionEngine)
    engine.execution_mode = DiffusionExecutionMode.STEP_BATCH
    engine._scheduling_paused = False
    with pytest.raises(NotImplementedError, match="request-level execution only"):
        engine._check_pause_request({"mode": "keep"})
    assert engine._scheduling_paused is False

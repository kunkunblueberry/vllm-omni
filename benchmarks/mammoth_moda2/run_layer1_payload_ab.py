# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Layer-1 MammothModa2 payload A/B benchmark.

This benchmark deliberately stops at the AR->DiT data-path boundary.  It does
not change the scheduler, connector selection, request lifecycle, or model
topology.  Run it from a vllm-omni checkout so that the imported ``ar2dit`` and
DiT split implementation are the exact version under test.

The benchmark reports both the current EngineCore byte-payload path and an
event/pinned-memory primitive chain.  The latter is evidence for a future
streaming design only; it is not presented as a production implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from vllm_omni.diffusion.models.mammoth_moda2.pipeline_mammothmoda2_dit import (
    MammothModa2DiTPipeline,
)
from vllm_omni.engine.serialization import (
    deserialize_additional_information,
    serialize_additional_information,
)
from vllm_omni.model_executor.stage_input_processors.mammoth_moda2 import ar2dit
from vllm_omni.platforms import current_omni_platform


def _stats(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    ordered = sorted(samples)

    def percentile(q: float) -> float:
        index = (len(ordered) - 1) * q
        low = int(index)
        high = min(low + 1, len(ordered) - 1)
        fraction = index - low
        return ordered[low] + (ordered[high] - ordered[low]) * fraction

    return {
        "mean": statistics.fmean(samples),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "min": min(samples),
        "max": max(samples),
    }


def _time_cpu(fn: Callable[[], Any], warmups: int, runs: int) -> dict[str, float]:
    for _ in range(warmups):
        fn()
    samples: list[float] = []
    for _ in range(runs):
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1000.0)
    return _stats(samples)


def _cuda_sync() -> None:
    current_omni_platform.synchronize()


def _cuda_copy_stats(
    source: torch.Tensor,
    *,
    direction: str,
    pinned_source: bool,
    target_dtype: torch.dtype | None,
    warmups: int,
    runs: int,
) -> dict[str, float] | None:
    """Measure one CUDA copy with CUDA events.

    ``direction`` is ``h2d`` or ``d2h``.  A device-side event is used for the
    measured dependency; host synchronization is only used to collect the
    result, so this does not silently turn an asynchronous copy into a timing
    of unrelated work.
    """
    if not torch.cuda.is_available():
        return None
    device = source.device if source.device.type == "cuda" else torch.device("cuda")
    if direction == "h2d":
        host_source = source
        if host_source.device.type != "cpu":
            raise ValueError("h2d source must be a CPU tensor")
        if pinned_source:
            try:
                host_source = host_source.pin_memory()
            except RuntimeError:
                return None
        destination = torch.empty(
            host_source.shape,
            dtype=target_dtype or host_source.dtype,
            device=device,
        )
    elif direction == "d2h":
        device_source = source
        if device_source.device.type != "cuda":
            raise ValueError("d2h source must be a CUDA tensor")
        destination = torch.empty_like(device_source, device="cpu", pin_memory=pinned_source)
        host_source = device_source
    else:
        raise ValueError(f"Unknown copy direction: {direction}")

    stream = torch.cuda.Stream(device=device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    def copy_once() -> float:
        _cuda_sync()
        with torch.cuda.stream(stream):
            start_event.record(stream)
            if direction == "h2d":
                destination.copy_(host_source, non_blocking=True)
            else:
                destination.copy_(host_source, non_blocking=True)
            end_event.record(stream)
        end_event.synchronize()
        return float(start_event.elapsed_time(end_event))

    for _ in range(warmups):
        copy_once()
    samples = [copy_once() for _ in range(runs)]
    return _stats(samples)


def _build_prompt(prompt_rows: int, generated_rows: int, hidden_states: torch.Tensor) -> tuple[Any, dict[str, Any]]:
    prompt_token_ids = [7 + (i % 50) for i in range(prompt_rows)]
    generated_token_ids = [100_000 + i for i in range(generated_rows)]
    # ar2dit intentionally drops the look-ahead token, so include one extra id.
    completion = SimpleNamespace(
        cumulative_token_ids=generated_token_ids + [100_000 + generated_rows],
        multimodal_output={"latent": hidden_states},
    )
    ar_output = SimpleNamespace(
        request_id="layer1-benchmark-request",
        prompt_token_ids=prompt_token_ids,
        outputs=[completion],
    )
    prompt = {
        "additional_information": {
            "image_height": [256],
            "image_width": [256],
            "text_guidance_scale": [1.0],
            "cfg_range": [0.0, 1.0],
            "num_inference_steps": [2],
        },
        "mm_processor_kwargs": {},
    }
    expected_rows = prompt_rows + generated_rows
    if hidden_states.shape[0] != expected_rows:
        raise ValueError(f"hidden state rows={hidden_states.shape[0]} expected {expected_rows}")
    return ar_output, prompt


def _make_splitter() -> MammothModa2DiTPipeline:
    pipeline = object.__new__(MammothModa2DiTPipeline)
    object.__setattr__(
        pipeline,
        "config",
        SimpleNamespace(
            llm_config=SimpleNamespace(gen_vocab_start_index=100_000),
            image_token_id=20,
            video_token_id=21,
            vision_start_token_id=22,
            vision_end_token_id=23,
        ),
    )
    return pipeline


def _payload_bytes(wire: Any) -> tuple[int, int]:
    tensor_bytes = 0
    total_bytes = 0
    entries = getattr(wire, "entries", {}) or {}
    for entry in entries.values():
        data = getattr(entry, "tensor_data", None)
        if data is not None:
            tensor_bytes += len(data)
        # msgspec Struct fields are not directly available as a wire size, so
        # use the exact msgpack representation when msgspec is installed.
    try:
        import msgspec

        total_bytes = len(msgspec.msgpack.encode(wire))
    except Exception:
        total_bytes = tensor_bytes
    return tensor_bytes, total_bytes


def _hash_tensor(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _event_chain_stats(
    source_gpu: torch.Tensor,
    *,
    warmups: int,
    runs: int,
) -> dict[str, Any] | None:
    """Measure a Layer-1-only pinned D2H -> host -> pinned H2D chain.

    The host waits once for D2H readiness.  H2D completion is consumed by a
    device-side ``wait_event`` before a tiny consumer kernel, which models the
    DiT stream dependency without adding another host-side wait to the design.
    """
    if not torch.cuda.is_available():
        return None
    device = source_gpu.device
    try:
        d2h_host = torch.empty_like(source_gpu, device="cpu", pin_memory=True)
    except RuntimeError:
        return None
    h2d_destination = torch.empty_like(d2h_host, device=device)
    copy_stream = torch.cuda.Stream(device=device)
    dit_stream = torch.cuda.Stream(device=device)
    producer_stream = torch.cuda.current_stream(device)

    def once() -> float:
        _cuda_sync()
        host_started = time.perf_counter()
        with torch.cuda.stream(copy_stream):
            copy_stream.wait_stream(producer_stream)
            d2h_host.copy_(source_gpu, non_blocking=True)
            d2h_ready = torch.cuda.Event()
            d2h_ready.record(copy_stream)
        # This is the one host-side dependency needed before CPU serialization.
        d2h_ready.synchronize()
        with torch.cuda.stream(dit_stream):
            h2d_done = torch.cuda.Event()
            h2d_destination.copy_(d2h_host, non_blocking=True)
            h2d_done.record(dit_stream)
            # DiT consumes only after the device-side event; use a tiny reduction
            # to ensure the destination remains live through the consumer.
            dit_stream.wait_event(h2d_done)
            _ = h2d_destination[:1].sum()
        _cuda_sync()
        return (time.perf_counter() - host_started) * 1000.0

    for _ in range(warmups):
        once()
    samples = [once() for _ in range(runs)]
    return {"wall_ms": _stats(samples), "host_waits": 1, "device_waits": 1}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.dtype != "bfloat16":
        raise ValueError("The Mammoth AR path under test is BF16 on the A100 benchmark host")
    source_cpu = torch.arange(args.rows * args.hidden_size, dtype=torch.bfloat16).reshape(args.rows, args.hidden_size)
    ar_output, prompt = _build_prompt(args.prompt_rows, args.rows - args.prompt_rows, source_cpu)

    ar2dit_result: list[Any] = []

    def run_ar2dit() -> None:
        nonlocal ar2dit_result
        ar2dit_result = ar2dit([ar_output], prompt)

    ar2dit_stats = _time_cpu(run_ar2dit, args.warmups, args.runs)
    dit_input = ar2dit_result[0]
    info = dit_input["additional_information"]
    transferred = info["full_hidden_states"]
    if not isinstance(transferred, torch.Tensor):
        raise TypeError("ar2dit did not return a tensor payload")

    wire_holder: list[Any] = []

    def run_serialize() -> None:
        nonlocal wire_holder
        wire_holder = [serialize_additional_information(info)]

    serialize_stats = _time_cpu(run_serialize, args.warmups, args.runs)
    wire = wire_holder[0]
    if wire is None:
        raise RuntimeError("serialize_additional_information returned None")
    tensor_bytes, msgpack_bytes = _payload_bytes(wire)

    restored_holder: list[Any] = []

    def run_deserialize() -> None:
        nonlocal restored_holder
        restored_holder = [deserialize_additional_information(wire)]

    deserialize_stats = _time_cpu(run_deserialize, args.warmups, args.runs)
    restored_info = restored_holder[0]
    restored = restored_info["full_hidden_states"]
    if not isinstance(restored, torch.Tensor):
        raise TypeError("deserialize_additional_information did not return a tensor")

    splitter = _make_splitter()
    split_holder: list[Any] = []

    def run_split() -> None:
        nonlocal split_holder
        split_holder = [
            splitter._split_ar_conditions(
                full_hidden_states=restored,
                full_token_ids=restored_info["full_token_ids"],
                answer_start_index=int(restored_info["answer_start_index"][0]),
            )
        ]

    split_stats = _time_cpu(run_split, args.warmups, args.runs)
    text_cond, image_cond = split_holder[0]

    precision_source = source_cpu.float()
    restored_fp32 = restored.float()
    consumer_cast = restored.to(dtype=torch.bfloat16)
    max_abs_diff = float((restored_fp32 - precision_source).abs().max().item())
    consumer_exact = bool(torch.equal(consumer_cast, source_cpu))

    gpu_metrics: dict[str, Any] = {}
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        source_gpu = source_cpu.to(device=device)
        restored_gpu_source = restored.to(device="cpu")
        gpu_metrics = {
            "source_d2h_pageable_ms": _cuda_copy_stats(
                source_gpu,
                direction="d2h",
                pinned_source=False,
                target_dtype=None,
                warmups=args.warmups,
                runs=args.runs,
            ),
            "source_d2h_pinned_ms": _cuda_copy_stats(
                source_gpu,
                direction="d2h",
                pinned_source=True,
                target_dtype=None,
                warmups=args.warmups,
                runs=args.runs,
            ),
            "handoff_h2d_pageable_ms": _cuda_copy_stats(
                restored_gpu_source,
                direction="h2d",
                pinned_source=False,
                target_dtype=torch.bfloat16,
                warmups=args.warmups,
                runs=args.runs,
            ),
            "handoff_h2d_pinned_ms": _cuda_copy_stats(
                restored_gpu_source,
                direction="h2d",
                pinned_source=True,
                target_dtype=torch.bfloat16,
                warmups=args.warmups,
                runs=args.runs,
            ),
            "event_chain": _event_chain_stats(
                source_gpu,
                warmups=args.warmups,
                runs=args.runs,
            ),
        }

    return {
        "schema_version": 1,
        "rows": args.rows,
        "prompt_rows": args.prompt_rows,
        "generated_rows": args.rows - args.prompt_rows,
        "hidden_size": args.hidden_size,
        "source_dtype": str(source_cpu.dtype),
        "ar2dit_dtype": str(transferred.dtype),
        "restored_dtype": str(restored.dtype),
        "text_condition_shape": list(text_cond.shape),
        "image_condition_shape": list(image_cond.shape),
        "ar2dit_ms": ar2dit_stats,
        "serialize_ms": serialize_stats,
        "deserialize_ms": deserialize_stats,
        "condition_split_ms": split_stats,
        "wire_tensor_bytes": tensor_bytes,
        "wire_msgpack_bytes": msgpack_bytes,
        "source_bytes": source_cpu.numel() * source_cpu.element_size(),
        "consumer_target_dtype": str(torch.bfloat16),
        "max_abs_diff_after_restore_float32": max_abs_diff,
        "consumer_cast_exact": consumer_exact,
        "source_sha256": _hash_tensor(source_cpu),
        "restored_sha256": _hash_tensor(restored),
        "gpu": gpu_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", type=int, default=1824)
    parser.add_argument("--prompt-rows", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()
    if args.prompt_rows <= 0 or args.prompt_rows >= args.rows:
        raise ValueError("prompt rows must be between 1 and total rows")
    result = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Run a reproducible MammothModa2 baseline/candidate A/B experiment.

The script intentionally uses the Omni API directly so deploy YAML owns stage
configuration and the text-to-image example's global CLI defaults cannot add
unowned overrides.  It records initialization time, per-request latency,
output hashes/statistics, and peak process-level GPU memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers.utils import numpy_to_pil

from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.entrypoints.openai.stage_params import clone_sampling_params
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_extras import (
    build_text_to_image_prompt as build_model_text_to_image_prompt,
)
from vllm_omni.model_extras import (
    get_model_class_name,
    should_init_extra_args_for_non_diffusion_stages,
)


def _normalize_images(images: list[Any]) -> list[Any]:
    normalized: list[Any] = []
    for image in images:
        if isinstance(image, np.ndarray):
            normalized.extend(numpy_to_pil(image))
        else:
            normalized.append(image)
    return normalized


def _sample_gpu_memory(stop: threading.Event, samples: list[dict[str, int]]) -> None:
    while not stop.is_set():
        try:
            raw = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=2,
            )
            for line in raw.strip().splitlines():
                fields = [field.strip() for field in line.split(",")]
                if len(fields) == 3:
                    samples.append(
                        {
                            "gpu": int(fields[0]),
                            "memory_used_mib": int(fields[1]),
                            "utilization_gpu": int(fields[2]),
                        }
                    )
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        stop.wait(0.2)


def _image_record(image: Any) -> dict[str, Any]:
    if hasattr(image, "convert"):
        array = np.asarray(image.convert("RGB"))
    else:
        array = np.asarray(image)
    raw = np.ascontiguousarray(array).tobytes()
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "min": int(array.min()),
        "max": int(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
    }


def _build_sampling_params(omni: Omni, prompt: dict[str, Any], seed: int, steps: int) -> list[Any]:
    height = int(prompt["additional_information"]["image_height"][0])
    width = int(prompt["additional_information"]["image_width"][0])
    diffusion_params = OmniDiffusionSamplingParams(
        height=height,
        width=width,
        seed=seed,
        generator=torch.Generator(device="cuda").manual_seed(seed),
        true_cfg_scale=4.0,
        guidance_scale=4.0,
        num_inference_steps=steps,
        num_outputs_per_prompt=1,
        extra_args={
            "text_guidance_scale": 1.0,
            "cfg_range": [0.0, 1.0],
            "num_inference_steps": steps,
        },
    )
    sampling_params = [clone_sampling_params(p) for p in (omni.default_sampling_params_list or [])]
    if not sampling_params:
        return [diffusion_params]

    replaced_diffusion = False
    if should_init_extra_args_for_non_diffusion_stages(get_model_class_name(omni)):
        for stage_id, params in enumerate(sampling_params):
            if isinstance(params, OmniDiffusionSamplingParams):
                sampling_params[stage_id] = diffusion_params
                replaced_diffusion = True
                continue
            if hasattr(params, "extra_args"):
                params.extra_args = dict(getattr(params, "extra_args", None) or {})
                params.extra_args.update(diffusion_params.extra_args)
            if hasattr(params, "seed"):
                params.seed = seed
            if stage_id == 0:
                ar_width = int(prompt["additional_information"]["ar_width"][0])
                ar_height = int(prompt["additional_information"]["ar_height"][0])
                params.max_tokens = ar_height * (ar_width + 1) + 1
    if not replaced_diffusion and len(sampling_params) == 1:
        return [diffusion_params]
    return sampling_params


def run(args: argparse.Namespace) -> dict[str, Any]:
    samples: list[dict[str, int]] = []
    stop = threading.Event()
    sampler = threading.Thread(target=_sample_gpu_memory, args=(stop, samples), daemon=True)
    sampler.start()
    started = time.perf_counter()
    omni: Omni | None = None
    profile_active = False
    profile_stages = [int(stage) for stage in args.profile_stages.split(",") if stage.strip()]
    try:
        omni = Omni(model=args.model, deploy_config=args.deploy_config, mode="text-to-image")
        init_seconds = time.perf_counter() - started
        model_class_name = get_model_class_name(omni)
        prompt = build_model_text_to_image_prompt(
            model_class_name=model_class_name,
            prompt={"prompt": args.prompt, "modalities": ["image"]},
            height=args.height,
            width=args.width,
        )
        records: list[dict[str, Any]] = []
        total_requests = args.warmups + args.runs
        for request_index in range(total_requests):
            if profile_stages and request_index == args.warmups:
                # CUDA-profiler start/stop is used by Nsight Systems as its capture
                # range.  Starting after warmup keeps model loading and cache setup
                # out of the measured window while covering both stage workers.
                omni.start_profile(stages=profile_stages)
                profile_active = True
            params = _build_sampling_params(omni, prompt, args.seed, args.steps)
            request_started = time.perf_counter()
            outputs = omni.generate(prompt, sampling_params_list=params, use_tqdm=False)
            elapsed = time.perf_counter() - request_started
            images: list[Any] | None = None
            for output in outputs:
                images = getattr(output, "images", None)
                if images:
                    break
            if not images:
                images = extract_images_from_outputs(outputs)
            if not images:
                raise RuntimeError("No images found in Omni output")
            normalized = _normalize_images(images)
            record = {
                "request_index": request_index,
                "warmup": request_index < args.warmups,
                "elapsed_seconds": elapsed,
                "image": _image_record(normalized[0]),
            }
            records.append(record)
            if request_index == args.warmups - 1 or request_index == total_requests - 1:
                normalized[0].save(Path(args.output_dir) / f"request_{request_index}.png")
        if profile_active:
            omni.stop_profile(stages=profile_stages)
            profile_active = False
        return {
            "model": args.model,
            "deploy_config": args.deploy_config,
            "prompt": args.prompt,
            "height": args.height,
            "width": args.width,
            "steps": args.steps,
            "seed": args.seed,
            "warmups": args.warmups,
            "runs": args.runs,
            "profile_stages": profile_stages,
            "init_seconds": init_seconds,
            "requests": records,
            "gpu_samples": samples,
            "peak_gpu_memory_mib": {
                str(gpu): max((s["memory_used_mib"] for s in samples if s["gpu"] == gpu), default=0)
                for gpu in sorted({s["gpu"] for s in samples})
            },
        }
    finally:
        if omni is not None:
            if profile_active:
                try:
                    omni.stop_profile(stages=profile_stages)
                except Exception:
                    pass
            omni.close()
        stop.set()
        sampler.join(timeout=3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--deploy-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", default="A cat sitting on a laptop keyboard")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=142)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--profile-stages",
        default="",
        help="Comma-separated stage IDs to profile after warmup; requires profiler_config in deploy YAML.",
    )
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    result = run(args)
    Path(args.output_dir, "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

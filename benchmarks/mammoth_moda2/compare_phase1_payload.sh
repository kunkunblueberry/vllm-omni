#!/usr/bin/env bash
# Compare MammothModa2's pre-Phase-1 payload path with the request-end path.
# Run from the optimized checkout on a two-GPU host.
#
# PROFILE_BACKEND=none (default) performs a clean latency comparison.
# PROFILE_BACKEND=torch writes child-worker PyTorch traces and reports stage-0
# aten::to call counts. PROFILE_BACKEND=nsys performs a separate Nsight Systems
# run. Never enable torch and nsys in one process tree: both subscribe to CUPTI.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel)}"
PYTHON_BIN="${PYTHON_BIN:-/data/vllm-workspace/.venv/bin/python}"
MODEL="${MODEL:-/data/vllm-workspace/models/MammothModa2-Preview}"
DRIVER_SCRIPT="$REPO_ROOT/examples/offline_inference/text_to_image/text_to_image.py"
PHASE1_COMMIT="${PHASE1_COMMIT:-$(git -C "$REPO_ROOT" log --format=%H --fixed-strings --grep='Optimize MammothModa2 request-end AR to DiT payload' -1)}"
if [[ -z "$PHASE1_COMMIT" ]]; then
    echo "Set PHASE1_COMMIT or BASE_COMMIT: the Phase 1 implementation commit was not found." >&2
    exit 1
fi
BASE_COMMIT="${BASE_COMMIT:-${PHASE1_COMMIT}^}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/results/mammoth_moda2_phase1_$(date +%Y%m%d_%H%M%S)}"
PROMPT="${PROMPT:-A small red cabin beside a quiet mountain lake at sunrise}"
WIDTH="${WIDTH:-512}"
HEIGHT="${HEIGHT:-512}"
PROFILE_BACKEND="${PROFILE_BACKEND:-none}"
PAYLOAD_STATS="${VLLM_OMNI_MAMMOTH_MODA2_PAYLOAD_STATS:-0}"
REQUIRE_IDLE_GPUS="${REQUIRE_IDLE_GPUS:-1}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
MEASURED_RUNS="${MEASURED_RUNS:-5}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

if [[ ! -f "$DRIVER_SCRIPT" ]]; then
    echo "Repeated-run driver not found: $DRIVER_SCRIPT" >&2
    exit 1
fi

if ! [[ "$WARMUP_RUNS" =~ ^[0-9]+$ ]] || ! [[ "$MEASURED_RUNS" =~ ^[1-9][0-9]*$ ]]; then
    echo "WARMUP_RUNS must be non-negative and MEASURED_RUNS must be positive." >&2
    exit 1
fi

if ! [[ "$WIDTH" =~ ^[1-9][0-9]*$ ]] || ! [[ "$HEIGHT" =~ ^[1-9][0-9]*$ ]]; then
    echo "WIDTH and HEIGHT must be positive integers." >&2
    exit 1
fi

if [[ "$PROFILE_BACKEND" != "none" && "$PROFILE_BACKEND" != "torch" && "$PROFILE_BACKEND" != "nsys" ]]; then
    echo "PROFILE_BACKEND must be 'none', 'torch', or 'nsys', got: $PROFILE_BACKEND" >&2
    exit 1
fi

require_idle_gpus() {
    if [[ "$REQUIRE_IDLE_GPUS" != "1" ]]; then
        return
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi is required for the idle-GPU safety check." >&2
        exit 1
    fi

    local active_processes
    active_processes="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
        --format=csv,noheader 2>/dev/null | grep -v 'No running processes found' || true)"
    if [[ -n "$active_processes" ]]; then
        echo "Refusing to start the benchmark because compute processes already own GPU memory:" >&2
        echo "$active_processes" >&2
        echo "Inspect and stop only the verified stale process, then rerun." >&2
        echo "Set REQUIRE_IDLE_GPUS=0 only when intentionally sharing the GPUs." >&2
        exit 1
    fi
}

require_idle_gpus

mkdir -p "$RESULTS_DIR"
{
    printf 'prompt=%s\n' "$PROMPT"
    printf 'width=%s\n' "$WIDTH"
    printf 'height=%s\n' "$HEIGHT"
    printf 'warmup_runs=%s\n' "$WARMUP_RUNS"
    printf 'measured_runs=%s\n' "$MEASURED_RUNS"
    printf 'profile_backend=%s\n' "$PROFILE_BACKEND"
} > "$RESULTS_DIR/run_config.txt"
BASE_WORKTREE="$(mktemp -d /tmp/vllm-omni-mammoth-baseline.XXXXXX)"
rmdir "$BASE_WORKTREE"

cleanup() {
    git -C "$REPO_ROOT" worktree remove --force "$BASE_WORKTREE" 2>/dev/null || true
}
trap cleanup EXIT

git -C "$REPO_ROOT" diff --check
git -C "$REPO_ROOT" rev-parse --verify "$BASE_COMMIT^{commit}" >/dev/null
git -C "$REPO_ROOT" worktree add --detach "$BASE_WORKTREE" "$BASE_COMMIT"

write_deploy_config() {
    local deploy_path="$1"
    local profiler_backend="$2"
    local label="$3"

    cat > "$deploy_path" <<'YAML'
async_chunk: false
pipeline: mammoth_moda2

stages:
  - stage_id: 0
    devices: "0"
    max_num_seqs: 1
    max_model_len: 2048
    max_num_batched_tokens: 2048
    gpu_memory_utilization: 0.85
    enforce_eager: true
    trust_remote_code: true
YAML

    if [[ "$profiler_backend" == "torch" ]]; then
        cat >> "$deploy_path" <<YAML
    profiler_config:
      profiler: torch
      torch_profiler_dir: $RESULTS_DIR/torch_${label}_stage0
      torch_profiler_record_shapes: false
      torch_profiler_with_memory: false
      torch_profiler_with_stack: false
YAML
    elif [[ "$profiler_backend" == "nsys" ]]; then
        cat >> "$deploy_path" <<'YAML'
    profiler_config:
      profiler: cuda
YAML
    fi

    cat >> "$deploy_path" <<'YAML'
    enable_prefix_caching: false

  - stage_id: 1
    devices: "1"
    max_num_seqs: 1
    gpu_memory_utilization: 0.3
    enforce_eager: true
    trust_remote_code: true
YAML

    if [[ "$profiler_backend" == "torch" ]]; then
        cat >> "$deploy_path" <<YAML
    profiler_config:
      profiler: torch
      torch_profiler_dir: $RESULTS_DIR/torch_${label}_stage1
      torch_profiler_record_shapes: false
      torch_profiler_with_memory: false
      torch_profiler_with_stack: false
YAML
    elif [[ "$profiler_backend" == "nsys" ]]; then
        cat >> "$deploy_path" <<'YAML'
    profiler_config:
      profiler: cuda
YAML
    fi

    cat >> "$deploy_path" <<'YAML'
    enable_prefix_caching: false
    default_sampling_params:
      extra_args:
        text_guidance_scale: 4.0
        cfg_range: [0.0, 1.0]
        num_inference_steps: 20
YAML
}

write_deploy_config "$RESULTS_DIR/deploy.yaml" "none" ""

TORCH_PROFILER_JSON_TEMPLATE='{"profiler":"torch","torch_profiler_dir":"%s","torch_profiler_record_shapes":false,"torch_profiler_with_memory":false,"torch_profiler_with_stack":false}'

run_case() {
    local label="$1"
    local checkout="$2"
    local mode="$3"
    local output="$RESULTS_DIR/${label}_${mode}.png"
    local log="$RESULTS_DIR/${label}_${mode}.log"
    local deploy_config="$RESULTS_DIR/deploy.yaml"
    if [[ "$mode" == "profile" ]]; then
        deploy_config="$RESULTS_DIR/deploy_${label}_${mode}.yaml"
        write_deploy_config "$deploy_config" "$PROFILE_BACKEND" "$label"
    fi
    local -a command=(
        env "CUDA_VISIBLE_DEVICES=0,1" "PYTHONPATH=$checkout" \
        "VLLM_OMNI_MAMMOTH_MODA2_PAYLOAD_STATS=$PAYLOAD_STATS" "$PYTHON_BIN"
        # The current driver owns repeated same-engine measurement. PYTHONPATH
        # selects the baseline or optimized implementation under test.
        "$DRIVER_SCRIPT"
        --model "$MODEL"
        --deploy-config "$deploy_config"
        --prompt "$PROMPT"
        --width "$WIDTH" --height "$HEIGHT"
        --num-inference-steps 20
        --guidance-scale 4.0
        --seed 42
        --num-warmups "$WARMUP_RUNS"
        --num-runs "$MEASURED_RUNS"
        --output "$output"
    )

    if [[ "$mode" == "profile" && "$PROFILE_BACKEND" == "torch" ]]; then
        local trace_dir="$RESULTS_DIR/torch_${label}"
        local profiler_json
        printf -v profiler_json "$TORCH_PROFILER_JSON_TEMPLATE" "$trace_dir"
        command+=(--profiler-config "$profiler_json")
    elif [[ "$mode" == "profile" && "$PROFILE_BACKEND" == "nsys" ]]; then
        # Existing vLLM CUDA-profiler support opens the worker capture range.
        command+=(--profiler-config '{"profiler":"cuda"}')
    fi

    echo "=== $label / $mode ===" | tee "$log"
    if [[ "$mode" == "profile" && "$PROFILE_BACKEND" == "nsys" ]]; then
        local -a nsys_command=(
            "$NSYS_BIN" profile --force-overwrite true --trace=cuda,nvtx,osrt
            --cuda-graph-trace=node --capture-range=cudaProfilerApi
            --capture-range-end=repeat --sample=none
            --output "$RESULTS_DIR/nsys_${label}"
        )
        if "$NSYS_BIN" profile --help 2>&1 | grep -q -- '--trace-fork-before-exec'; then
            nsys_command+=(--trace-fork-before-exec=true)
        fi
        env VLLM_OMNI_MAMMOTH_MODA2_NVTX=1 "${nsys_command[@]}" "${command[@]}" 2>&1 | tee -a "$log"
    else
        "${command[@]}" 2>&1 | tee -a "$log"
    fi

    test -s "$output"
}

if [[ "$PROFILE_BACKEND" == "nsys" ]] && command -v nsys >/dev/null 2>&1; then
    NSYS_BIN="$(command -v nsys)"
    "$NSYS_BIN" --version | tee "$RESULTS_DIR/nsys_version.txt"
    NSYS_ANALYZER="$REPO_ROOT/benchmarks/mammoth_moda2/analyze_nsys_transfer.py"
    if [[ ! -f "$NSYS_ANALYZER" ]]; then
        echo "Nsight analyzer is missing: $NSYS_ANALYZER" >&2
        echo "Restore benchmarks/mammoth_moda2/analyze_nsys_transfer.py before starting a costly run." >&2
        exit 1
    fi
else
    NSYS_BIN=""
    NSYS_ANALYZER=""
    if [[ "$PROFILE_BACKEND" == "nsys" ]]; then
        echo "PROFILE_BACKEND=nsys requires an nsys executable on PATH." >&2
        exit 1
    fi
    echo "PROFILE_BACKEND=$PROFILE_BACKEND; Nsight is intentionally disabled." | tee "$RESULTS_DIR/nsys_version.txt"
fi

git -C "$REPO_ROOT" rev-parse HEAD > "$RESULTS_DIR/optimized_commit.txt"
git -C "$BASE_WORKTREE" rev-parse HEAD > "$RESULTS_DIR/baseline_commit.txt"

# Each invocation creates one engine, warms it in-process, and then collects
# repeated timed requests. Do not add a separate process-level warmup here.
run_case baseline "$BASE_WORKTREE" profile
run_case optimized "$REPO_ROOT" profile

"$PYTHON_BIN" - "$RESULTS_DIR/baseline_profile.png" "$RESULTS_DIR/optimized_profile.png" <<'PY'
from PIL import Image, ImageChops, ImageStat
import sys

baseline = Image.open(sys.argv[1]).convert("RGB")
optimized = Image.open(sys.argv[2]).convert("RGB")
if baseline.size != optimized.size:
    raise SystemExit(f"image size mismatch: {baseline.size} != {optimized.size}")
diff = ImageChops.difference(baseline, optimized)
stats = ImageStat.Stat(diff)
print(f"image_size={baseline.size}")
print(f"pixel_max_abs_diff={max(max(channel) for channel in diff.getextrema())}")
print(f"pixel_mean_abs_diff={sum(stats.mean) / len(stats.mean):.6f}")
PY

for label in baseline optimized; do
    grep -E 'Total generation time|stage_0_gen_ms|stage_1_gen_ms|hidden_(d2h|snapshot)|mammoth_moda2 payload stats' \
        "$RESULTS_DIR/${label}_profile.log" > "$RESULTS_DIR/${label}_summary.txt" || true
    if [[ "$PROFILE_BACKEND" == "nsys" && -f "$RESULTS_DIR/nsys_${label}.nsys-rep" ]]; then
        "$NSYS_BIN" export --type sqlite --force-overwrite true \
            --output "$RESULTS_DIR/nsys_${label}" \
            "$RESULTS_DIR/nsys_${label}.nsys-rep" \
            > "$RESULTS_DIR/${label}_nsys_export.txt" 2>&1
        for report in cuda_api_sum cuda_gpu_mem_time_sum cuda_gpu_mem_size_sum; do
            "$NSYS_BIN" stats --force-export true --report "$report" \
                "$RESULTS_DIR/nsys_${label}.nsys-rep" \
                > "$RESULTS_DIR/${label}_${report}.txt" 2>&1 || true
        done
    fi
done

"$PYTHON_BIN" - "$RESULTS_DIR" <<'PY'
import json
from pathlib import Path
import re
import sys

results_dir = Path(sys.argv[1])
time_re = re.compile(r"Total generation time: [^(]+\(([0-9.]+) ms\) \[run (\d+)/(\d+)\]")
stage_prefix = "Stage timing summary: "

def percentile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    low, high = int(index), min(int(index) + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (index - low)

def distribution(values):
    return {
        "min": min(values),
        "p50": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }

def stage_timing_percentiles(stage_summaries):
    aggregate = {}
    for stage_summary in stage_summaries:
        for stage_id, metrics in stage_summary.items():
            for metric_name, value in metrics.items():
                aggregate.setdefault(stage_id, {}).setdefault(metric_name, []).append(value)
    return {
        stage_id: {
            metric_name: distribution(values)
            for metric_name, values in sorted(metrics.items())
        }
        for stage_id, metrics in sorted(aggregate.items())
    }

for label in ("baseline", "optimized"):
    log_path = results_dir / f"{label}_profile.log"
    lines = log_path.read_text(errors="replace").splitlines()
    samples = [float(match.group(1)) for line in lines if (match := time_re.search(line))]
    stage_summaries = [json.loads(line[len(stage_prefix):]) for line in lines if line.startswith(stage_prefix)]
    if not samples:
        raise SystemExit(f"No timed samples found in {log_path}")
    if len(samples) != len(stage_summaries):
        raise SystemExit(f"Timed sample/stage summary mismatch in {log_path}: {len(samples)} != {len(stage_summaries)}")
    summary = {
        "sample_count": len(samples),
        "e2e_ms": distribution(samples),
        "samples_ms": samples,
        "stage_summaries": stage_summaries,
        "stage_timing_ms": stage_timing_percentiles(stage_summaries),
    }
    output = results_dir / f"{label}_latency_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[{label}] samples={len(samples)} p50={summary['e2e_ms']['p50']:.2f}ms p95={summary['e2e_ms']['p95']:.2f}ms")
PY

if [[ "$PROFILE_BACKEND" == "torch" ]]; then
"$PYTHON_BIN" - "$RESULTS_DIR" <<'PY'
from pathlib import Path
import re
import sys

results_dir = Path(sys.argv[1])
launch_counts = {}
for label in ("baseline", "optimized"):
    trace_roots = sorted(results_dir.glob(f"torch_{label}_stage*"))
    traces = [trace for root in trace_roots for trace in root.rglob("*.json")]
    traces.extend(trace for root in trace_roots for trace in root.rglob("*.json.gz"))
    stage0_tables = sorted((results_dir / f"torch_{label}_stage0").rglob("profiler_out_0.txt"))
    to_calls = None
    if len(stage0_tables) == 1:
        for line in stage0_tables[0].read_text(errors="replace").splitlines():
            if line.strip().startswith("aten::to"):
                match = re.search(r"(\d+)\s*$", line)
                if match:
                    to_calls = int(match.group(1))
                break
    summary = results_dir / f"{label}_torch_trace_markers.txt"
    lines = [f"trace_files={len(traces)}"]
    lines.append(f"stage0_aten_to_calls={to_calls}")
    summary.write_text("\n".join(lines) + "\n")
    print(f"[{label}] " + ", ".join(lines))
PY
elif [[ "$PROFILE_BACKEND" == "nsys" ]]; then
"$PYTHON_BIN" "$NSYS_ANALYZER" "$RESULTS_DIR"
fi

find "$RESULTS_DIR" -maxdepth 3 -type f -printf '%p\n' | sort
echo "Results directory: $RESULTS_DIR"

# Ming-Image 0.1 Design

> Text-to-image, image editing, and layer decomposition.

## Summary

- Vendor: inclusionAI
- Models: `inclusionAI/Ming-Image-0.1-Design` and `inclusionAI/Ming-Image-0.1-Design-Layer`
- Runtime: vLLM-Omni two-stage online serving
- API: OpenAI-compatible chat completions

## Architecture

Stage 0 uses a Qwen2.5-VL vision tower plus a 20-layer BailingMoeV2 language model.
It appends 256 learned image-query tokens and exports the final query states plus direct VLM states from layers 5, 12, and 20.
The query-token checkpoint is in the root sibling `mlp/` directory, so remote stage-0 materialization downloads both `mllm/` and `mlp/`.

Stage 1 projects those conditions through the checkpoint connector and runs a 30-layer Z-Image DiT with a Qwen-Image RGBA VAE.

For Design-Layer, the first returned image is the reconstructed composite and the remaining images are the requested layers.

## CUDA

### Environment

- OS: Linux
- Python: 3.10+
- CUDA: 13.0
- vLLM version: 0.29.0
- vLLM-Omni version or commit: 6daf5b30f
- 2x H100 80GB (1xH100 to be validated)

### Commands

```bash
MODEL=inclusionAI/Ming-Image-0.1-Design
vllm serve "$MODEL" --omni --deploy-config vllm_omni/deploy/ming_image.yaml --port 8091
```

For layer decomposition, set `MODEL` to `inclusionAI/Ming-Image-0.1-Design-Layer`.

## Text-to-image

Note that a prompt refiner is expected to describe the prompts with details; we will refine with more example inputs soon.

```bash
curl -s http://127.0.0.1:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "inclusionAI/Ming-Image-0.1-Design",
    "messages": [{"role": "user", "content": "A clean editorial botanical poster"}],
    "modalities": ["image"],
    "extra_body": {
      "height": 1024, "width": 1024,
      "num_inference_steps": 12, "guidance_scale": 1.0, "seed": 42
    }
  }' \
  | jq -r '.choices[0].message.content[0].image_url.url | split(",")[1]' \
  | base64 -d > ming_design_smoke.png
```

## Image editing

Pass one input image and an editing instruction:

```bash
MODEL=inclusionAI/Ming-Image-0.1-Design
INPUT_IMAGE=/path/to/input.png

jq -n \
  --arg model "$MODEL" \
  --rawfile image <(base64 -w0 "$INPUT_IMAGE") \
  '{
    model: $model,
    messages: [{role: "user", content: [
      {type: "image_url", image_url: {url: ("data:image/png;base64," + $image)}},
      {type: "text", text: "Change the background to blue"}
    ]}],
    modalities: ["image"],
    extra_body: {
      height: 1024, width: 1024,
      num_inference_steps: 12, guidance_scale: 1.0, seed: 42
    }
  }' |
curl -sS http://127.0.0.1:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  --data-binary @- \
  | jq -r '.choices[0].message.content[0].image_url.url | split(",")[1]' \
  | base64 -d > ming_edit.png
```

## Layer decomposition

Use the Design-Layer checkpoint and set `INPUT_IMAGE` to a local flattened design image.
Note that the prompt should better depict each layer to be decomposed, we will refine with more example inputs soon.

```bash
MODEL=inclusionAI/Ming-Image-0.1-Design-Layer
INPUT_IMAGE=/path/to/input.png
PROMPT="Decompose this image into 6 layers with the following specifications: Number of layers: 6 \nLayer 1: Central text ... Layer 2: ... Layer 6: ..."

jq -n \
  --arg model "$MODEL" \
  --rawfile image <(base64 -w0 "$INPUT_IMAGE") \
  --arg prompt "$PROMPT" \
  '{
    model: $model,
    messages: [{role: "user", content: [
      {type: "image_url", image_url: {url: ("data:image/png;base64," + $image)}},
      {type: "text", text: $prompt}
    ]}],
    modalities: ["image"],
    extra_body: {
      num_layers: 6,
      height: 1024, width: 1024,
      num_inference_steps: 12, guidance_scale: 2.0, seed: 42
    }
  }' |
curl -sS http://127.0.0.1:8091/v1/chat/completions \
  -H "Content-Type: application/json" \
  --data-binary @- > response.json

jq -r '.choices[0].message.content[].image_url.url | split(",")[1]' response.json |
  nl -v 0 |
  while read -r index data; do
    printf "%s" "$data" | base64 -d > "ming_layer_${index}.png"
  done
```

## Optional placement profiles

`ming_image_single_gpu.yaml` places both stages on device 0 and opts the
single diffusion replica into the existing inline client. It removes the
diffusion subprocess transport hop; the AR-to-DiT shared-memory connector
remains. The typed stage config owns the inline selection and leaves the
registered topology unchanged. Check checkpoint memory capacity on the target
GPU before using this profile.

`ming_image_cfg_parallel.yaml` places AR on device 0 and the two diffusion CFG
ranks on devices 1 and 2. CFG is active only for `guidance_scale > 1`:
Design defaults to 1, while Design-Layer defaults to 2. Both ranks combine the
gathered predictions and step the scheduler locally. These profiles keep one
request, one output per prompt, and one inflight stage edge.

```bash
# Run on a provisioned server with the checkpoints already available.
MODEL=/path/to/Ming-Image-0.1-Design
vllm serve "$MODEL" --omni --deploy-config vllm_omni/deploy/ming_image_single_gpu.yaml --port 8091

# Use the Layer request above with a reference image and guidance_scale=2.
MODEL=/path/to/Ming-Image-0.1-Design-Layer
vllm serve "$MODEL" --omni --deploy-config vllm_omni/deploy/ming_image_cfg_parallel.yaml --port 8091
```

## Denoising contract and regression tests

The bridge selects the final 256 query states and concatenates direct states
from layers 5, 12, and 20, excluding all 258 suffix tokens. Stage 1 keeps latents
as `[B,C,F,H,W]`, including Design's single frame; schedule shifting uses the
last two spatial axes. The transformer receives `[C,F,H,W]` samples and crops
reference-frame predictions before the scheduler. Decode returns the composite
first, followed by layers, preserving RGBA channels.

For positive prediction `p`, negative prediction `n`, and guidance `s`, Ming
uses `u = p + s*(p-n)` and passes `-u` to the FP32 scheduler. The shared loop's
optional norm clamp considers the entire C/F/H/W sample and only shrinks values
above `q*norm(p)`; normalization is disabled by default and can be selected with
`cfg_normalize`. Negative
query and projected direct conditions are zero; zeroing raw direct states before
the biased projector would produce a different condition. Each actual branch
gets its own projected conditions, the same reference, and a graph step marker.
Request and branch scopes restore the caller's context even after exceptions.

The weight-free tests import production modules normally. Their provenance is:

| Scenario | Input source | Why valid | Expected | Expected source | Bug prevented |
| --- | --- | --- | --- | --- | --- |
| Design CFG0/1/2 and Layer CFG2 | Real `thinker2image` output from a four-token prefix and the documented query suffix | Uses query `(256,2048)` and direct `(4,6144)` from the actual bridge | One or four frames; query indices 5..260; direct layer values 5/12/20 | AR producer layout and Design/Layer defaults | Wrong model, suffix leakage, extra frame dimension, F1 squeeze |
| CFG and clamp | Scalar `p=2,n=1,s=2`; samples `(3,4)`, zero, and unchanged positive | Independent float tensors spanning the frame axis | `u=4`, scheduler `-4`; clamp `(4.5,6)`, finite zero, unchanged `(3,4)` | Hand arithmetic and the correct-model fused foundation loop | Standard CFG substitution, wrong sign, row-wise norm, division by zero |
| Projected direct and reference routing | Real conditioning forward with biased projector double; RGBA reference via real reference encoder adapter | Bias exposes zero-before-projection errors; parent DiT receives actual Ming adapter arguments | Positive direct 12/19/27; negative all zero; same one-row reference; crop reference sentinel 99 | Projector equation and branch-local batch contract | Positive condition leaked into negative, fused 2B context silently zipped |
| RGBA consumer | Real latent preparation/flatten/decode/postprocessor with VAE sentinels | Independent frame/channel values, composite plus three layers | Composite first; four PIL RGBA images; alpha 0/128/255/128 | Explicit fixture channels and PIL pixel expectations | Reordered frames, discarded alpha, accidental reference output |
| Marker and lifecycle | Successful forward, injected positive/negative DiT errors, invalid reference encoding | Exercises real request and branch scopes with preexisting context values | Marker before every call; outer tensor identity restored; pending/frame state cleaned | Caller ownership and per-invocation CUDA Graph Trees requirement | Missing second marker, clearing caller context, error-state leakage |
| Typed co-location | Actual deploy YAML through `VllmOmniConfig.from_pipeline_config`, StageRuntime launch and client factory | Covers the typed runtime consumer instead of only legacy YAML projection | Default subprocess; single GPU inline; CFG profile subprocess; explicit false respected | Profile placement and existing runtime/client selection API | Inline flag lost in typed conversion, registry mutation, string boolean coercion |

Run these without checkpoints in a supported vLLM-Omni environment:

```bash
python -m pytest -q tests/diffusion/models/ming_image/test_a6_contract.py \
  tests/config/test_ming_image_a6_profile.py \
  tests/model_executor/models/ming_image/test_config_and_pipeline.py
```

Real two-rank collectives, compiled graph replay over repeated shapes/layer
counts, checkpoint numerical parity, memory capacity, and serving output are
server validation boundaries. Weight-free doubles cannot establish them.

## Optional request and continuous batching

The default deployment still uses one inflight request. Select
`vllm_omni/deploy/ming_image_throughput.yaml` to use the real Ming pipeline with
the existing step scheduler, two inflight requests, AR on device 0 and DiT on
device 1. This profile is opt-in; its memory capacity needs server validation.

Each request owns its projected query/direct conditions, reference latent,
seed/generator, `[1,C,F,H,W]` latent envelope and private Euler scheduler. Request
mode returns a list of `DiffusionOutput` in scheduler order, including a singleton
list for one request. Each output contains only that request's generated frames.
Height, width, frame count, direct-prefix structure, reference presence and nominal
CFG settings define compatible admission. Seeds and step progress stay private.

`prepare_encode`, `denoise_step`, `step_scheduler` and `post_decode` implement
the shared step protocol. A wave can contain unequal progress or a newly admitted
request. CFG truncation uses each request's current timestep, so rows with
different effective CFG run in separate transformer groups and predictions are
scattered back to request order. Scheduler updates use private cursors. Completion,
abort and errors invalidate the runner's cached batch when it retains a retired
row. The next wave rebuilds from live states.

The B1/B2 tests use the real `thinker2image` bridge, request/state dataclasses,
`InputBatch`, `StepScheduler`, runner step core and RGBA consumer. Kernel doubles
give independent `p=17+2*offset+ref+t+frame`, `n=ref+t+frame` predictions and a
literal Euler delta of 1/4. Expectations use those formulas and seeded torch
draws, rather than another pipeline execution. Coverage includes two distinct
requests, both variants, per-sample normalization, custom schedules and supplied
latents, unequal progress/effective CFG, late admission, early retirement, errors
and a following request. Negative projected direct conditioning remains zero.

On a supported server environment (no checkpoints required for these tests):

```bash
python -m pytest -q tests/diffusion/models/ming_image/test_a6_contract.py \
  tests/diffusion/models/ming_image/test_batch_step_contract.py \
  tests/config/test_ming_image_a6_profile.py \
  tests/model_executor/models/ming_image/test_config_and_pipeline.py \
  tests/diffusion/models/z_image/test_zimage_cfg_parallel.py
python -m pytest -q -m cpu tests/diffusion/test_diffusion_step_pipeline.py \
  tests/diffusion/batching/test_diffusion_batching.py
```

After those pass, serve a checkpoint that is already available on the server:

```bash
MODEL=/path/to/Ming-Image-0.1-Design
vllm serve "$MODEL" --omni --deploy-config vllm_omni/deploy/ming_image_throughput.yaml --port 8091
# Repeat with Design-Layer and the reference-image request above.
```

Compare two concurrent requests with their own seeded single-request outputs,
including Layer composite order/alpha. Validate uneven step counts, late arrival,
repeated compiled shapes, CFG ranks and checkpoint numerical parity on the server.
Restricted local source diagnostics do not establish full production imports,
native scheduler/VAE parity, serving or CUDA graph replay.

## Validated end-to-end scope

The Design and Design-Layer checkpoints have been exercised through the complete
online image-generation path: HTTP chat completions, the AR model, captured query
and direct conditions, the shared-memory stage connector, step-batched DiT
execution, VAE decode, and RGBA responses. This is functional end-to-end validation,
not an image-quality comparison against the vendor implementation.

The validated placement used three RTX 4080 SUPER GPUs with the original BF16
checkpoints: AR TP2 on devices 0 and 1, and DiT TP1 / CFG parallel size 1 on device
2. Both stages were eager, diffusion attention used `TORCH_SDPA`, and exact
batch-versus-singleton comparisons enabled `VLLM_BATCH_INVARIANT=1`. The deployment
fixture in the expansion module builds this placement from the throughput profile;
the unmodified throughput YAML uses a different placement.

Coverage includes eight distinct prompts with seeds 0 through 7 and unequal
3/4/5-step schedules for both checkpoints, exact decoded-pixel comparisons with
each request's singleton output, actual multi-request denoise waves, Layer
reference input and composite-plus-layer RGBA outputs, non-square geometry,
guidance defaults, and shape/frame-count replay. An additional live-server error
sequence verifies that rejected requests do not prevent a later valid request
from reproducing its seeded output. Unsupported negative prompts return HTTP 400;
some worker-side invalid geometry, step-count, and reference errors still return
request-scoped HTTP 500 responses.

The weight-backed late-admission and cancellation checks separately exercise the
production scheduler, InputBatch, runner and engine with transport boundaries
isolated. They are not proof of HTTP-disconnect or native-KV cancellation. Full
checkpoint CFG-parallel-size-2 serving, full-model compiled/CUDA-graph serving,
default-mode bitwise batch invariance, and vendor image accuracy remain outside
this end-to-end validation claim. Continuous batching does not imply a throughput
improvement for every checkpoint or workload.

With a matching Linux runtime and both checkpoints already present locally:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 model_prefix= MING_IMAGE_CHECKPOINT_ROOT=/models \
  python -m pytest -sv \
  tests/e2e/online_serving/test_ming_image_0_1_design_expansion.py \
  -m 'full_model and diffusion and H100 and cards_3' --run-level full_model

CUDA_VISIBLE_DEVICES=0,1,2 model_prefix= MING_IMAGE_CHECKPOINT_ROOT=/models \
  python -m pytest -sv \
  tests/dfx/reliability/invalid_param_test/test_invalid_omni_chat.py \
  -k ming_chat_error_retirement --run-level full_model
```

`MING_IMAGE_CHECKPOINT_ROOT` must contain both model directories including their
`mllm`, sibling `mlp`, `transformer`, and VAE components. The H100 marker is the
repository's test-routing convention, not the hardware used for the reported
manual run. These local-checkpoint expansion cases are not yet wired to an
automatic nightly checkpoint-provisioning job.

## Notes

- The default deployment keeps Stage 0 eager and enables dynamic regional
  compilation with CUDA Graph Trees for the repeated Stage 1 DiT blocks.
  The first request for each new image shape or layer count pays compilation
  and graph-capture cost; warm up every production shape before measuring or
  serving latency-sensitive traffic.
- Each request supports one reference image and one output per prompt. Compatible
  requests can share a denoise wave; each Layer output contains its own composite
  and layers.
- Design-Layer requires a reference image except during warmup.
- A non-empty `negative_prompt` is rejected; Ming-Image uses zero negative conditioning.
- Height and width must be divisible by 16.
- Returned images retain the checkpoint's four RGBA channels.

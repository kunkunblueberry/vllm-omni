# MammothModa2 AR-to-DiT Dataflow Audit

## Scope and Status

This is an audit of the current standard MammothModa2 path at base commit
`422a63fd`, with only the local low-precision payload change applied. It is a
Layer 0/1 description, not a proposal to change the scheduler, connector API,
or request lifecycle. Prefix caching is disabled in the default Mammoth deploy
configuration, so the direct-path statements below apply to that configuration.

## Current Path

```text
AR GPU hidden states
  |  GPUARModelRunner._to_cpu_contiguous()
  v
CPU hidden-state payload, associated with one AR request
  |  Stage 0 OmniRequestOutput / orchestrator transition
  v
Stage 1 ar2dit(source_output, original prompt)
  |  validate row count; retain dtype; build token/metadata payload
  v
EngineCore AdditionalInformationPayload
  |  tensor -> CPU uint8 bytes -> process transport -> bytearray tensor
  v
DiT CPU conditions selected by logical row/token-id masks
  |  final .to(model_device, model_dtype)
  v
DiT GPU condition tensors
```

The stage topology has AR as stage 0 and DiT as stage 1; DiT receives its input
through `custom_process_input_func=ar2dit`. The topology does **not** declare a
Mammoth full-payload producer hook. Therefore this condition handoff is not an
instance of `SharedMemoryConnector` transport. Replacing that connector cannot
optimize this path without first changing the topology and lifecycle contract.

## Data Contract

| Item | Current contract | Owner at boundary |
| --- | --- | --- |
| Request identity | `ar_output.request_id`; stage transition keeps the source/target relationship | orchestrator/stage clients |
| Hidden states | 2-D `[logical_token_rows, hidden_size]`; AR dtype, contiguous before EngineCore payload | AR output until serialized bytes are accepted |
| Token ids | `prompt_token_ids + cumulative_token_ids[:-1]` | `ar2dit` metadata payload |
| Look-ahead rule | Final sampled token is excluded because it has no corresponding hidden-state row | `ar2dit` validates row count |
| Sequence boundary | `answer_start_index = len(prompt_token_ids)` | `ar2dit` metadata payload |
| Text condition | Rows before the boundary excluding visual/generative IDs | DiT forward temporary |
| Image condition | Rows after the boundary with `token_id >= gen_vocab_start_index` | DiT forward temporary |

The DiT condition split uses `torch.arange` over payload rows. These are
logical sequence rows, not M-RoPE coordinates. Consequently, a future
prefix-cache or chunked-prefill path must reconstruct the complete logical
row/token sequence before this boundary. The DiT consumer cannot infer or
materialize rows that upstream execution skipped.

## Dependency Classification

| Boundary | Dependency | Can be made asynchronous? | Required safety condition |
| --- | --- | --- | --- |
| AR compute -> D2H | execution and memory | yes, with a dedicated copy stream/event | source GPU storage cannot be reused before copy event completes |
| D2H -> byte serialization | data and memory | no for the same payload bytes | CPU serializer must wait until its host staging buffer is ready |
| byte serialization -> EngineCore handoff | data and request lifecycle | not independently with the current payload object | serialized bytes must remain owned until process transport accepts them |
| byte decode -> H2D | data and memory | only if the decoded source is pinned | current `bytearray` decode is pageable; direct non-blocking H2D is not established |
| H2D -> DiT forward | execution and data | yes, by `DiT_stream.wait_event(h2d_done)` | destination GPU buffer must remain live through DiT consumption |
| DiT completion -> release | request lifecycle | no | only release after consumer completion, cancellation, or explicit acknowledgement |

The standard path performs conservative CPU/GPU materialization rather than
sharing a cross-stage GPU buffer. That costs copies, but makes the current
request completion semantics simple: there is no device handle whose lifetime
outlives the stage worker call.

## Observed Copy and Allocation Sites

| Code location | Operation | Consequence |
| --- | --- | --- |
| `gpu_ar_model_runner.py:_to_cpu_contiguous` | GPU hidden states -> contiguous CPU tensor | AR-side D2H and CPU allocation |
| `gpu_ar_model_runner.py:_build_omni_model_runner_output_from_snapshot` | builds per-request latent payload after staged CPU copy | must preserve request-to-row mapping |
| `stage_input_processors/mammoth_moda2.py:ar2dit` | validates alignment, constructs EngineCore metadata | low-precision patch avoids FP32 expansion here |
| `data_entry_keys.py:_serialize_tensor` | CPU tensor -> `uint8` bytes | creates serialized byte ownership boundary |
| `data_entry_keys.py:_deserialize_tensor` | `bytearray` -> CPU tensor | decoded tensor is pageable |
| `pipeline_mammothmoda2_dit.py:_split_ar_conditions` | mask/gather rows on the payload device | preserves source dtype after local patch |
| `pipeline_mammothmoda2_dit.py:forward` | selected conditions -> DiT model device/dtype | final H2D and precision conversion |

## Lifecycle Risks Preventing a Runtime Async Patch Today

1. A per-request staging slot would need an owner from AR ready-event record
   through DiT completion acknowledgement. No such cross-stage slot protocol
   exists in the standard Mammoth path.
2. Cancellation can occur after AR has queued work but before DiT runs. The
   current bytes handoff can be discarded with the request; a reusable pinned
   slot would need a cancellation fence before reuse.
3. Request preemption/replay requires a generation or epoch token. A delayed
   completion event for an older incarnation must not make a newer request with
   the same external identity appear ready.
4. Multiple AR requests can enqueue copies concurrently. An allocation pool
   requires a bounded queue/backpressure policy; otherwise it exchanges a
   synchronization bottleneck for unbounded pinned memory.
5. The full request-end experiment previously attempted to change this
   lifecycle and was reverted before `422a63fd`. That history is evidence to
   retain the normal path until a dedicated Layer 2 contract is reviewed.

## Minimal Future Payload-Handle Contract

This is a design boundary, not code to add now. A safe experimental handle
would need at least:

```text
request_id + generation
producer device + dtype + shape
staging allocation identity + owner
d2h_ready_event
h2d_done_event
consumer acknowledgement / release
cancelled and failed states
bounded pool admission + CPU pageable fallback
```

The required API is larger than a local `non_blocking=True` change. It crosses
Layer 1 into Layer 2, and therefore needs focused replay, cancellation,
cross-request isolation, and buffer-reuse tests before any runtime insertion.

## Decisions From This Audit

| Candidate | Evidence | Decision |
| --- | --- | --- |
| Preserve FP16/BF16 through EngineCore payload | Functional A/B and V100 proxy halve payload bytes | local code change retained for user validation |
| Producer pinned D2H + stream events | Correct transport microbenchmark with one final consumer wait | technical candidate only |
| Re-pin decoded byte payload before H2D | Byte-codec proxy improved total time but requires slot lifecycle | defer to Layer 2 design |
| Producer-side selected-condition payload | At most 1.81% representative row reduction | reject as standalone optimization |
| Change to SharedMemory/Mooncake connector | Standard Mammoth condition path does not use it | no architecture change |
| GPU-direct handoff | Needs handle/event/ownership/cancellation protocol | defer |

Raw results and methodology are recorded in `v100_precision_ab_20260910.md`
and `v100_precision_gpu_20260910.json` in this directory.

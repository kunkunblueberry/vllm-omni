# MammothModa2 V100 Precision Handoff A/B

## Scope

This record evaluates the low-precision handoff change on the local branch
`kunkun/expend_MammothModa2_lifestyle` at base commit `422a63fd`, with the
three uncommitted precision-path edits. No change in this experiment was
committed or pushed.

The V100 server runs `vllm==0.19.0` and `vllm-omni==0.19.0rc1`, while the local
branch contains a newer Mammoth stage interface. The server therefore tests an
isolated compatibility proxy of the same operation:

```text
selected FP16 AR condition -> FP32 expansion -> CPU handoff -> FP16 DiT consume
selected FP16 AR condition -> FP16 preserved -> CPU handoff -> FP16 DiT consume
```

It is not an end-to-end Mammoth inference result.

## Environment

| Item | Value |
| --- | --- |
| Remote host | `vllm` / shared 7x Tesla V100S 32GB |
| Runtime | Torch 2.10.0+cu126, vLLM 0.19.0, vLLM-Omni 0.19.0rc1 |
| Permitted dtype | FP16; V100 does not support BF16 inference |
| Worktree | `/data/sunjingbo/kun/work/mammoth_v100_precision_20260910` |
| Safety isolation | Baseline and candidate are copies under the user's `/data/sunjingbo/kun/work`; editable source was not modified |
| Mammoth checkpoint | Not present in the configured HF cache |

At the first availability check, all seven GPUs had existing compute processes.
After explicit authorization to share GPU 3, the benchmarks ran there with a
per-process allocator cap of 8% (about 2.6 GiB) while its existing service held
28,448 MiB. The launcher verifies the selected GPU is 3 or 4 in shared mode,
rejects use above 29,000 MiB, and releases all experiment allocations when its
short-lived process exits. No existing process was stopped or altered.

## Functional A/B

The test imported each isolated copy's real `ar2dit` function, supplied FP16
synthetic AR hidden states, and checked the selected text/image rows and the
values observed after the final FP16 consumer conversion.

| Metric | Baseline: force FP32 | Candidate: preserve FP16 | Result |
| --- | ---: | ---: | --- |
| Text condition shape | `[2, 4]` | `[2, 4]` | identical |
| Image condition shape | `[3, 4]` | `[3, 4]` | identical |
| Condition payload dtype | FP32 | FP16 | expected |
| Condition payload bytes | 80 | 40 | 50% lower |
| Value equivalence after FP16 consume | true | true | passed |

An additional dependency-light check ran the current local `ar2dit` file in
the remote isolated candidate tree with the remote EngineCore codec. Its FP16
payload survived `ar2dit -> serialize -> deserialize` with shape `[5, 4]`,
40 bytes, matching dtype, and exact values. This is a direct bridge/wire
contract result, not a full current-branch DiT import: the remote release
predates the current `SupportsComponentDiscovery` interface and cannot import
the current DiT pipeline. The remote old codec also rejects BF16 NumPy
conversion. That limitation does not apply to the local current serializer,
which serializes BF16 through a `uint8` raw-byte view, but it prevents remote
dynamic BF16 validation; V100 inference itself is FP16-only.

## V100 GPU A/B

The prepared GPU benchmark uses the isolated real adapter with eight synthetic
requests, 256 prompt rows, 1568 image rows, hidden size 4096, five warmups, and
20 timed runs. It records condition payload bytes, peak allocation delta, and
wall-clock mean/p50/p95 for the selected-condition D2H and final FP16 H2D
sequence.

| Check | Status | Result |
| --- | --- | --- |
| Candidate source patch and syntax | passed | isolated remote worktree created and compiled |
| CPU functional A/B | passed | values and row masks preserved |
| V100 GPU latency/memory A/B | passed | shared GPU 3, capped at 8% per-process allocator memory |
| Pinned staging/event-chain feasibility A/B | passed | transport-only test; no runtime architecture change |
| Mammoth end-to-end A/B | blocked by environment | checkpoint absent and remote stage API predates local branch |

The GPU precision proxy used the real isolated `ar2dit` function, eight
synthetic requests, 256 prompt rows, 1,568 image rows, hidden size 4,096, five
warmups, and 20 timed runs. It isolates the FP32 expansion that the candidate
removes; it is not full Mammoth inference.

| Metric | Baseline: expand to FP32 | Candidate: preserve FP16 | Change |
| --- | ---: | ---: | --- |
| Cross-stage payload | 227.875 MiB | 113.938 MiB | 50.0% lower |
| Peak allocation delta | 366.422 MiB | 234.056 MiB | 36.1% lower |
| Adapter mean latency | 113.49 ms | 56.96 ms | 49.8% lower |
| Adapter p50 latency | 100.23 ms | 56.65 ms | 43.5% lower |
| Adapter p95 latency | 220.72 ms | 57.94 ms | lower; baseline contained noisy shared-GPU outliers |

The staging benchmark transferred 8 x 16 MiB FP16 payloads in each direction,
verified source, pinned host, and destination values, then compared a pageable,
per-request synchronized path against dedicated D2H/H2D streams linked by CUDA
events with exactly one final consumer wait.

| Metric | Serial per-request sync | Event chain, one consumer wait | Change |
| --- | ---: | ---: | --- |
| Mean end-to-end consumer wait | 74.90 ms | 17.40 ms | 76.8% lower |
| p50 end-to-end consumer wait | 64.95 ms | 15.90 ms | lower |
| p95 end-to-end consumer wait | 115.92 ms | 25.79 ms | lower |
| Event-chain enqueue time | n/a | 1.11 ms mean | measured separately |

## EngineCore Byte-Codec Boundary

The current standard Mammoth condition handoff does not opt into the
full-payload connector path. Stage 1 has only `custom_process_input_func=ar2dit`;
it receives an EngineCore payload. The relevant local codec turns a tensor into
CPU-contiguous `uint8` bytes and reconstructs it from a `bytearray`, yielding a
pageable CPU tensor. `SharedMemoryConnector` is therefore not the condition
transport to optimize on this path; its host shared-memory serialization would
only apply to the previously reverted request-end full-payload route.

An isolated V100 proxy reproduced that byte-codec contract for 8 x
`[1824, 4096]` FP16 tensors (14.25 MiB/request, 114 MiB total), checked every
source/destination tensor for exact equality, and timed ten samples per
variant. It uses the local serializer's operational sequence, but cannot be an
end-to-end EngineCore-process measurement because the server source predates
the local stage interface.

| Variant | D2H | Serialize | Deserialize | Re-pin | H2D | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pageable, serial per request | 36.52 ms | 21.94 ms | 48.33 ms | 0.00 ms | 24.95 ms | 131.74 ms |
| Pinned producer D2H event chain, pageable receive | 9.23 ms | 17.08 ms | 17.52 ms | 0.00 ms | 46.49 ms | 90.33 ms |
| Event chain, re-pin decoded receive tensor | 10.48 ms | 19.50 ms | 17.74 ms | 5.54 ms | 15.65 ms | 68.91 ms |

Compared with the serial proxy, the producer event chain reduced mean total
time 31.4%; re-pinning decoded receive tensors reduced it 47.7%. The latter
adds an explicit host copy but lowered the H2D component 66.3% versus using
the pageable decoded tensor directly. Shared-GPU variation means these are
design-direction measurements, not production latency claims.

The dependency analysis is more important than the percentages: a CPU serializer
cannot read the pinned D2H staging buffer until its ready event completes, so
that wait is required by data validity. After `bytearray` deserialization the
consumer buffer is pageable, so merely replacing a `synchronize()` with a CUDA
event does not make its H2D non-blocking. A production re-pin path would need
per-request ownership, reuse fencing, completion acknowledgement, cancellation
cleanup, backpressure, and a pageable fallback. It remains a Layer 2 proposal,
not a runtime patch.

## Producer-Side Selection Bound

The checked 512x768 request fixture declares `ar_width=48` and `ar_height=32`,
so the AR trajectory contains `32 * (48 + 1) = 1568` generated rows before the
final look-ahead token. Of those, 1536 are visual grid tokens and 32 are EOL
tokens. The current DiT mask consumes generated rows by
`token_id >= gen_vocab_start_index`; the exact EOL classification depends on
the checkpoint config, which is not available on the server.

Even under the most favorable old-mask assumption, where all 32 EOL rows and a
single prompt vision placeholder are omitted before D2H, producer-side
selection removes only 33 rows from a representative `256 + 1568 = 1824` row
handoff: 1.81%. It cannot approach the 50% reduction from preserving FP16.
This rules out reviving request-end producer-side selection as a standalone
optimization without a profile showing an unusually large prompt component.

## Layer 0 Contract Boundary

The current DiT consumer reconstructs its conditions from the payload using
logical row order: rows before `answer_start_index` are questions, rows after
it are answers, and token-id masks select text or generated visual conditions.
This is distinct from M-RoPE position coordinates. It assumes the upstream AR
path has already produced a complete, correctly ordered hidden-state/token-id
sequence; it neither materializes rows skipped by a prefix-cache hit nor
reassembles chunked-prefill outputs.

The bridge tests cover the ordinary completed-AR handoff, metadata, shape
rejection, and the one-token look-ahead convention (the final sampled token
does not have a corresponding hidden-state row). They do not establish
prefix-cache reconstruction, chunked-prefill concatenation, end-to-end image
equivalence, or cross-request isolation. Those must be explicit Layer 0 and
Layer 2 tests before changing the transport contract beyond this dtype fix.

## Optimization Decision Matrix

| Objective | Layer | Evidence | Decision |
| --- | --- | --- | --- |
| Preserve AR low precision through the handoff | 1 | CPU and V100 proxy A/B passed; payload halved | implemented locally; keep uncommitted for user validation |
| Pinned D2H/H2D event chain with one consumer wait | 1 | V100 transport microbenchmark passed | keep as a design candidate; do not add a runtime interface yet |
| Re-pin decoded EngineCore payload before H2D | 1/2 boundary | V100 byte-codec proxy reduced total time but adds host copy and lifecycle ownership | defer pending a dedicated payload-handle lifecycle design |
| Producer-side condition selection | 1/0 boundary | at most about 1.81% representative row reduction | do not revive request-end path for this alone |
| SharedMemoryConnector replacement | 1/architecture boundary | current Mammoth condition path does not route through it | do not modify connector selection for Mammoth |
| GPU-direct payload connector | 1/2 boundary | current bytes IPC and shared memory are host paths | defer: requires a separate handle, event, ownership, cancellation, and fallback design |

## Interpretation Boundary

The functional result and compatibility-proxy V100 measurement prove the
requested dtype optimization: FP16 conditions need not become FP32 before a
consumer that converts them to FP16. The measured 50% payload reduction applies
to these selected conditions. It does not prove end-to-end throughput
improvement, eliminate the current AR worker's earlier D2H, or establish
asynchronous cross-request safety. Those claims require a matching Mammoth
checkpoint and a separate Layer 2 lifecycle design.

The separate staging feasibility test compares per-request synchronized
pageable D2H/H2D with pinned host buffers, two CUDA streams, a D2H-ready event,
and exactly one final H2D completion wait. It keeps the source, pinned host,
and destination buffers live through that event. The result establishes that
the dependency chain is technically viable, but it is not itself a production
connector: a production payload-handle design still needs request identity,
allocation ownership, release acknowledgement, cancellation, stale-handle
protection, backpressure, and a CPU fallback.

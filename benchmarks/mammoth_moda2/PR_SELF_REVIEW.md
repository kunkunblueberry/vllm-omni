# MammothModa2 AR-to-DiT payload precision: PR self-review

## 1. Review scope and decision

This change is a Layer-1 data-path optimization for MammothModa2. It preserves
the AR hidden-state dtype across the existing AR-to-DiT payload boundary and
defers conversion to the DiT consumer's actual model dtype.

The final patch deliberately does **not** implement a new cross-stage runtime,
request-end producer, connector, staging pool, or production streaming path.
That boundary is intentional: the measured, low-risk optimization is complete,
while the remaining streaming ideas require a separate Layer-2 lifecycle
design.

Review base:

- branch: `kunkun/expend_MammothModa2_lifestyle`
- base commit: `422a63fd`
- related work: MammothModa2 umbrella issue #7075 and AR-to-DiT transfer issue
  #7087

## 2. What changed

The previous path expanded the AR representation before the stage boundary:

```text
AR BF16/FP16 hidden states
  -> FP32 expansion in ar2dit
  -> EngineCore byte payload
  -> CPU decode
  -> DiT condition split
  -> model dtype
```

The candidate path is:

```text
AR BF16/FP16 hidden states
  -> ar2dit keeps the source dtype
  -> EngineCore raw-byte payload
  -> CPU decode with recorded dtype/shape
  -> DiT selects logical rows
  -> one final device/model-dtype conversion
```

The production diff is intentionally narrow:

| File | Change |
| --- | --- |
| `vllm_omni/model_executor/stage_input_processors/mammoth_moda2.py` | Stop converting `full_hidden_states` to FP32 before serialization. |
| `vllm_omni/diffusion/models/mammoth_moda2/pipeline_mammothmoda2_dit.py` | Keep selected text/image rows in the received dtype until the existing final consumer conversion. |
| `tests/model_executor/stage_input_processors/test_mammoth_moda2.py` | Add FP16/BF16 payload round-trip and condition-split regression coverage. |

No public connector or stage interface changed.

## 3. Previous attempts and why they were not retained

Earlier commits explored a request-end payload path and producer-side condition
selection. Those experiments touched shared runtime code, including the GPU AR
runner and the Omni connector/model-runner mixin. They were useful for mapping
the lifecycle, but they changed the ownership and timing of the request-end
handoff:

```text
AR execution -> request-end accumulation -> producer selection
             -> cross-stage handoff -> DiT
```

That path made multi-request scheduling, cancellation, replay, and buffer
ownership part of the optimization itself. It therefore crossed from the
data-path question into the request-lifecycle contract before the latter had
dedicated tests. The representative producer-side row reduction was also only
about 1.81%, far below the 50% reduction obtained by removing the FP32
expansion.

Commit `422a63fd` restored standard AR-to-DiT forwarding and withdrew that
request-end architecture path. This history matters for review: it would be
incorrect to claim that the branch history never touched shared architecture
files. The accurate claim is that the **final candidate diff under review no
longer contains those shared-runtime changes**.

## 4. Architecture and intrusion audit

The final candidate changes only MammothModa2-specific producer/consumer code
and its tests. It does not modify:

- scheduler or request admission;
- `gpu_ar_model_runner.py`;
- `omni_connector_model_runner_mixin.py`;
- EngineCore or the generic serializer API;
- connector selection, SharedMemory, or Mooncake;
- request cancellation, cleanup, queueing, or backpressure;
- stage topology or device placement;
- prefix-cache or chunked-prefill logic.

The generic serializer is only used through its existing contract. It stores a
contiguous tensor as raw `uint8` bytes plus shape and dtype metadata, so the
change does not require a serializer protocol extension.

This is the minimum local alternative to the earlier architecture attempts:
retain the ordinary stage handoff, remove only the unnecessary precision
expansion, and convert at the point where the DiT model actually consumes the
conditions.

## 5. Correctness contract

The payload contract remains:

| Field | Contract |
| --- | --- |
| hidden-state shape | 2-D `[logical_token_rows, hidden_size]` |
| hidden-state dtype | AR source dtype (BF16 or FP16 in the tested path) |
| layout | contiguous row-major tensor before serialization |
| token IDs | prompt IDs followed by generated IDs except the final look-ahead ID |
| sequence boundary | `answer_start_index = len(prompt_token_ids)` |
| text rows | question-side rows excluding visual and generated-token IDs |
| image rows | answer-side rows with `token_id >= gen_vocab_start_index` |

The DiT consumer still performs the same logical masks and then executes the
existing final:

```python
tensor.to(device=model_device, dtype=target_dtype, non_blocking=True)
```

The patch therefore changes transport representation, not token ordering,
mask semantics, look-ahead handling, or the model's arithmetic dtype.

This contract assumes that upstream AR execution has already reconstructed a
complete logical hidden-state/token sequence. Prefix-cache reconstruction and
chunked-prefill reassembly are not inferred by the DiT stage and are not part
of this patch.

## 6. A/B methodology

All comparisons used the same prompt, seed, resolution, diffusion steps, model
checkpoint, deployment YAML, and request order. Baseline and candidate were
run as separate processes on the same host. Warmups were excluded from timing.
Output images were recorded as shape, statistics, and SHA256 values.

The primary end-to-end run used:

- 2 x NVIDIA A100-PCIE-40GB;
- Torch `2.13.0+cu132`, CUDA `13.2`;
- checkpoint `/data/vllm-workspace/models/MammothModa2-Preview`;
- `256 x 256`, 2 diffusion steps, seed `142`;
- 2 warmups and 10 timed requests.

## 7. End-to-end A/B result

The ten-run result is directionally faster, but its variance is large enough
that this should not be described as a statistically significant end-to-end
speedup:

| Metric | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| mean request time | 6.367506 s | 6.306694 s | -60.812 ms (-0.96%) |
| standard deviation | 80.246 ms | 194.437 ms | higher variance |
| p50 request time | 6.340186 s | 6.254637 s | -1.35% |
| p95 request time | 6.496411 s | 6.654906 s | +2.44% |
| peak GPU 0 | 38195 MiB | 38195 MiB | unchanged |
| peak GPU 1 | 7559 MiB | 7559 MiB | unchanged |

Every timed output image had the same SHA256 in baseline and candidate. A
separate three-run smoke A/B also produced identical output hashes and means
(6.222329 s baseline versus 6.206147 s candidate), but the ten-run result is
the more useful estimate and is the one used for the interpretation above.

The honest conclusion is: the end-to-end result is compatible with a small
benefit, but the strong evidence is at the payload/transfer boundary below.

## 8. Layer-1 payload microbenchmark

Workload: 1,824 logical rows (`256` prompt rows + `1,568` generated rows),
hidden size `4,096`, source dtype BF16, 100 timed samples.

| Metric (p50) | Baseline: FP32 expansion | Candidate: dtype preserved | Change |
| --- | ---: | ---: | ---: |
| tensor wire payload | 29,884,416 B | 14,942,208 B | -50.0% |
| serialization | 4.896 ms | 2.402 ms | -50.9% |
| deserialization | 5.099 ms | 2.372 ms | -53.5% |
| pinned H2D | 2.492 ms | 1.226 ms | -50.8% |

The serializer round trip was exact: shape, dtype, contiguity, and tensor
values were preserved. The two selected condition shapes remained identical:
text `[236, 4096]` and image `[1568, 4096]`.

## 9. Nsight Systems evidence

The Nsight comparison used the same A100 host, checkpoint, prompt, seed,
resolution, and two diffusion steps. It captured stage-0 and stage-1 worker
activity, CUDA kernels, memcpy records, and GPU allocation records.

| Capture metric | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| stage-1 pageable H2D payload | 4,358,152 B | 2,179,080 B | -50.0% |
| total H2D bytes | 4,951,236 B | 2,772,164 B | -44.0% |
| total D2H bytes | 2,588,298 B | 2,588,298 B | unchanged |
| GPU H2D time | 4.0349 ms | 3.5950 ms | -10.9% |
| `cudaMemcpyAsync` API time | 176.865 ms | 159.443 ms | -9.85% |
| stage-0 kernel time | 3510.632 ms | 3483.736 ms | -0.77% |
| stage-1 kernel time | 92.192 ms | 92.097 ms | -0.10% |
| GPU allocation events | 8,620 | 8,620 | unchanged |

The trace confirms the expected payload-volume effect. It also confirms what
the patch does **not** do: D2H bytes are unchanged, allocation events are
unchanged, and the existing runtime synchronization structure remains.

## 10. V100 compatibility proxy

An isolated proxy was run on a shared Tesla V100S-PCIE-32GB (GPU 3, with an 8%
per-process allocator cap) because the available V100 environment did not have
the current Mammoth checkpoint/API. It used Torch `2.10.0+cu126`, vLLM
`0.19.0`, and vLLM-Omni `0.19.0rc1`; V100 inference was FP16-only.

This is supporting transport evidence, not a current-branch Mammoth
end-to-end result:

| Metric | Baseline: FP32 expansion | Candidate: FP16 preserved | Change |
| --- | ---: | ---: | ---: |
| cross-stage payload | 227.875 MiB | 113.938 MiB | -50.0% |
| peak allocation delta | 366.422 MiB | 234.056 MiB | -36.1% |
| adapter mean latency | 113.49 ms | 56.96 ms | -49.8% |
| serial staging -> event-chain consumer wait | 74.90 ms | 17.40 ms | -76.8% |

The functional proxy preserved selected rows and values after final FP16
consumption. The remote old codec could not dynamically validate BF16, so BF16
claims rely on the current branch's raw-byte serializer tests and the A100
payload benchmark.

## 11. Why production streaming is deferred

The current Mammoth condition path is:

```text
AR GPU hidden states
  -> existing AR runner D2H
  -> ar2dit
  -> EngineCore AdditionalInformationPayload bytes
  -> pageable CPU tensor after decode
  -> DiT H2D
  -> DiT
```

It does not use `SharedMemoryConnector` for this condition handoff. Changing
that connector would therefore be an architectural change rather than a local
optimization.

A transport-only V100 experiment showed that a dedicated D2H stream, a
ready-event, a dedicated H2D stream, and one final consumer wait are technically
viable. It did **not** establish a production-safe request protocol. A real
cross-request streaming path would need, at minimum:

```text
request_id + generation
staging-slot ownership
D2H-ready event
H2D-completion event
consumer acknowledgement/release
cancellation and failure cleanup
stale-event protection
bounded pool admission/backpressure
pageable fallback
```

Without that contract, a stale event or reused staging slot can make one
request consume another request's data. This is the exact class of lifecycle
failure that the earlier request-end experiments exposed. The current patch
does not claim to remove the existing waits, provide cross-request overlap, or
support streaming.

## 12. Compatibility and risk assessment

The expected compatibility risk is limited because:

1. `full_hidden_states` is a MammothModa2-specific payload field;
2. the generic serializer already records arbitrary supported torch dtypes as
   raw bytes plus dtype/shape metadata;
3. the DiT consumer still converts to the refiner/transformer parameter dtype;
4. token IDs, answer boundary, masks, and look-ahead behavior are unchanged;
5. no shared scheduler, connector, runner, or lifecycle code is in the final
   diff.

The following remain unproven and are intentionally listed as future work:

- prefix-cache hit reconstruction and logical-position alignment;
- chunked-prefill sequence concatenation;
- production pinned staging and event ownership;
- request cancellation/replay during an in-flight transfer;
- all project CI combinations and every accelerator backend;
- high-concurrency streaming behavior.

## 13. Tests and validation status

Completed:

- CPU unit coverage for ordinary completed-AR bridge behavior;
- FP16 and BF16 EngineCore serialization round trips;
- FP16 and BF16 DiT condition-split dtype/row-selection checks;
- `python -m compileall` on changed Python files;
- `git diff --check`;
- A100 end-to-end baseline/candidate A/B;
- 100-run Layer-1 payload A/B;
- Nsight Systems comparison;
- isolated V100 functional and transport proxies.

Not completed locally:

- the full pytest module could not start on Windows because the local
  environment does not provide `vllm`
  (`ModuleNotFoundError: No module named 'vllm'`);
- this is not evidence of a test failure in the patch, but it means full local
  CI should not be claimed from this checkout.

## 14. Evidence files and submission hygiene

The reviewable evidence kept in the branch is:

- `cross_stage_dataflow_20260910.md`;
- `v100_precision_ab_20260910.md`;
- `v100_precision_functional_20260910.json`;
- `v100_precision_gpu_20260910.json`;
- `ab_e2e_summary_20260911.json`;
- `layer1_baseline_metrics_20260910.json` and
  `layer1_candidate_metrics_20260910.json`;
- `layer1_baseline_metrics100_20260910.json`;
- `layer1_candidate_metrics100_20260910.json`;
- `nsys_transfer_20260911.md`;
- the two reproducible benchmark scripts and deployment YAMLs;
- `quality_audit_20260911.md` and `final_closeout_20260911.md`.

Raw run logs, PNG outputs, Nsight binary traces, the expanded remote-results
directory, and the 42 MB archive are deliberately excluded from the commit.
The local archive remains available for forensic review and has SHA256:

```text
081651e4e08a54fe6a2b20833a84be5d0d132e16fd9d9a1b7ebbd877214734ef
```

The commit must pass the repository's pre-commit hooks and carry a DCO line:

```text
Signed-off-by: kunkunblueberry <1833921874@qq.com>
```

The intended local commit command is:

```text
git commit -s -m "[Performance] Preserve MammothModa2 AR payload precision"
```

No `--no-verify` bypass is justified by this patch. If a hook cannot run
because a tool is absent from the local Windows environment, that limitation
must be recorded explicitly rather than reported as a pass.

## 15. Reviewer checklist

- [ ] Confirm the final production diff is limited to the two
  MammothModa2 runtime files.
- [ ] Confirm the generic serializer preserves dtype/shape through raw bytes.
- [ ] Confirm DiT performs the only required model-dtype conversion.
- [ ] Confirm token masks and look-ahead semantics are unchanged.
- [ ] Confirm benchmark claims distinguish local payload evidence from
  end-to-end significance.
- [ ] Confirm streaming, prefix-cache, chunked-prefill, and lifecycle work are
  explicitly deferred rather than implied to be implemented.
- [ ] Confirm pre-commit output and the DCO trailer on the final commit.

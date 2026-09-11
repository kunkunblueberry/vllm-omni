# MammothModa2 Nsight Systems transfer A/B

This is a profiling-only comparison on the same two-A100 host, checkpoint,
prompt, seed, resolution, and two diffusion steps. It captured one request
after one warmup for each variant. The profiling shim only instantiated the
existing CUDA profiler wrapper; it did not modify the production worktree.

The binary `.nsys-rep` and `.sqlite` traces remain in the local evidence
archive and are intentionally not committed. This checked-in summary records
the measurements needed to review the claim.

| Capture metric | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| Stage-1 pageable H2D payload | 4,358,152 B | 2,179,080 B | -50.0% |
| Total H2D bytes | 4,951,236 B | 2,772,164 B | -44.0% |
| Total D2H bytes | 2,588,298 B | 2,588,298 B | unchanged |
| Total GPU H2D time | 4.0349 ms | 3.5950 ms | -10.9% |
| `cudaMemcpyAsync` API time | 176.865 ms | 159.443 ms | -9.85% |
| Stage-0 kernel time | 3510.632 ms | 3483.736 ms | -0.77% |
| Stage-1 kernel time | 92.192 ms | 92.097 ms | -0.10% |
| GPU allocation events | 8,620 | 8,620 | unchanged |

The trace shows the expected reduction in the stage-1 condition payload and
total H2D volume. D2H is unchanged because the existing AR runner still
materializes its CPU output before `ar2dit`. Allocation counts and model/kernel
times are effectively unchanged.

This is evidence for a Layer-1 payload-volume optimization. It is not evidence
that the production path is already streaming, pinned, or safe for
cross-request buffer reuse. The existing synchronization structure remains;
the trace recorded 1,084 `cudaStreamSynchronize` calls in both variants.

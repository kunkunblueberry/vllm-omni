# SolarWM-H3 reconciliation

Integration foundation adopted from vllm-project/vllm-omni PR 8752, pinned
`d087c253bcb2ed7a8549be7483f3f6b507ed4ba3`. Model semantic oracle remains
SolarWM `ce1da4e7705391eda8eeda6016c0fd3f614b975e`.

## Decisions

- One standalone pipeline, not inheritance from the full traditional H3 pipeline.
- Native H3 projections/blocks plus scoped attention context; no `__class__` mutation.
- Strict published EMA validation and BF16 low-rank sidecars, not premerged weights.
- Nested published bundle discovery, capability metadata and explicit AR deploy.
- Offline and stepwise execution only: deprecated tick requests fail explicitly.
- Initial supported horizon capped at 158 frames. Smaller valid profiles remain
  available for low-cost correctness tests; default is 158 frames at 768x1344.
- Raw KV adapter resides in `models/solarwm_h3/raw_kv.py`, not the generic AR
  runtime. It reuses existing runner allocation and block lookup methods.
- History retention is 25 frames; model attention concatenates 5 current frames
  for a 30-frame visible video window. This path does not use the fused
  post-RoPE attention window truncation helper.
- Streaming overlap decoder is adopted, but has not been locally device-qualified.
- TP1 and optional strict Ulysses follow the reference PR restrictions. TP,
  offload and A100 deployment qualification are not claimed.

## Preserved work

Prior uncommitted implementation and unrelated working files are stored in the
stash named `solarwm pre-8752 reconciliation backup`. Earlier committed code
remains in git history. Duplicate vendored training/replay code and its old
fixtures were removed from the live implementation, not from history.

## Validation

Executed this turn: actual adopted layout and sampling modules imported in an
isolated package namespace; 158-to-47/50 geometry, 6638 packed rows, 526 audio
rows, 5040 video rows and the four-step closed-form schedule passed. Python
compileall and git diff whitespace checks passed.

Not executed: full pytest/framework imports (pytest, vllm and Ruff unavailable
locally), published checkpoint inference, CUDA parity and production-resolution
quality/performance. PR author's prior test results are not local evidence.
Do not claim Base Code qualification from these local static checks.

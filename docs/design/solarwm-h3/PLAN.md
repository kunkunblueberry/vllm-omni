# SolarWM-H3 Stage 2 → vLLM-Omni: plan and open decisions

Baseline: `kunkun/solarwm-h3` branched from `origin/main` @ `639abb2a9`.
Reference implementation: SolarWM `ce1da4e7` (Apache-2.0), local clone at
`solarwm-ref/SolarWM`.

## What this integration is

SolarWM turns H3 into an autoregressive world model: 10 chunks × 5 latent
frames, each chunk produced by **4 denoising forwards plus one `t=1.0` commit
forward**, attending to a sliding window of the previous 5 committed chunks.
KV is cached **before** MM-RoPE and camera PRoPE, because the window's position
encoding is re-derived on every read.

## Integration point (corrected)

The mainline is **stepwise** (`SupportsStepExecution`), not tick. The tick path
(`experimental/ar_diffusion/tick_protocol.py`, `session.py`, `consumer.py`) is
deprecated and must not be copied. LingBot World 2.0 is the template: its
`dmd_block.py` is structurally the same "4 probes + 1 clean-xv commit" loop as
SolarWM's `h3_sgf_rollout`, differing only in the direction of the flow
convention (LingBot's `t=0` commit is SolarWM's `t=1.0`).

## Reuse / adapt / new

| Part | Class | Note |
| --- | --- | --- |
| `geometry.py` | reuse (vendored) | NumPy-only constants and position grids |
| `flow.py` | reuse (vendored) | `torch_flow` variant: `v = x0 - noise`, `t=1` clean |
| `layout.py` | reuse (vendored) | Packed document: text / anchor / audio / target video |
| `row_timesteps.py` | reuse (ported) | Only the rollout entry point of `torch_layout` |
| H3 DiT | adapt | `MiniMaxH3DiTModel` needs a SolarWM attention path |
| Camera PRoPE | **new** | vLLM-Omni's H3 has none; see below |
| Raw-KV write timing | **new** | Must write pre-RoPE K/V, not post-RoPE |
| SGF rollout loop | **new** | Drives the 4 + 1 forwards and the commit |
| AR-Diffusion binding | adapt | Session/tick binding, following LingBot |

## Findings that shape the design

1. **vLLM-Omni's H3 has no camera PRoPE.** There is no camera/prope module
   under `models/minimax_h3/`, and `MiniMaxH3Attention.forward` only applies
   3-axis MM-RoPE via `fused_qk_norm_rope`. SolarWM's camera conditioning
   (a projection on the head dims `[96:128)`, applied *after* MM-RoPE) is
   entirely new code here.

2. **The two packed layouts share their conventions.** vLLM-Omni's
   `packed_sequence.py` uses the same `5/3 × (1,4,4,4,4)` temporal grid, the
   same `×32` endpoint-excluded spatial grid, and the same tag values
   (VIDEO=0, TEXT=1, AUDIO=2) as SolarWM's `layout.py`. It does *not* publish
   the camera-row, row-role or chunk-id arrays the SGF attention needs, and it
   pads to a 64-row boundary. Vendoring SolarWM's layout keeps the existing
   fl2va/ref2va paths untouched.

3. **Raw K/V cannot use the runtime's fused write+attend op.** The paged
   `paged_write_attn` attends directly from the pool, so it leaves no seam for
   "read history, then apply this window's RoPE and PRoPE". The pipeline must
   read the pool itself, re-encode the history, and call attention directly,
   while still letting the runtime own allocation, commit and eviction.

4. **`window_frames` needs setting to 30, not 25.** In
   `paged_attention.py:167-185` the visible set is the last `window_frames`
   frames *including the current chunk*. SolarWM's window is 5 history chunks
   plus the current one (`sgf_attention.py:164-166` builds rotations for
   `frames ∈ {5..30}`, and uses 30 at chunk index 5), so 25 would show only 4
   history chunks. To be re-confirmed once a real rollout can be compared.

5. **`euler_step` is not fp32 accumulation end to end.** It recovers x0 through
   `predict_clean_sample` in the *input* dtype and only upcasts the final
   blend. A bf16 step therefore carries bf16 error, which the flow tests pin
   rather than assume away.

## Status

Landed: the model-semantics layer plus its tests.

| File | State |
| --- | --- |
| `solarwm_h3/{geometry,flow,layout,row_timesteps}.py` | vendored, reference-parity checked |
| `tests/diffusion/models/solarwm_h3/` | 20 contract tests, 1 committed fixture |
| Pipeline, AR-Diffusion binding | not started |

## Next step

Build the offline oracle first: one request, request-local cache, all 10
chunks in a single `forward()`, compared chunk-by-chunk against SolarWM's
`h3_sgf_rollout` latents. That separates model semantics from the AR runtime —
if the realtime path later mismatches, the semantic layer is already ruled out.

## Blockers

- **Stage 2 weights are not available.** The server holds only MiniMax-H3
  FL2VA. SolarWM needs `junchaoh-cs/SolarWM-H3-33B` (base + Stage 2 LoRA r384 +
  `H3_SUPPORT/` silence latents), a gated Hugging Face repo.
- **The server is unreachable from here** (`162.105.195.172:22` times out).
- **33B needs 4× A100** at bf16, so GPU validation is a multi-GPU run.

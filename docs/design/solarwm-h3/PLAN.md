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
| `torch_geometry.py` | reuse (vendored) | The torch position grids the SGF attention reads |
| `flow.py` | reuse (vendored) | `torch_flow` variant: `v = x0 - noise`, `t=1` clean |
| `layout.py` | reuse (vendored) | Packed document: text / anchor / audio / target video |
| `row_timesteps.py` | reuse (ported) | Only the rollout entry point of `torch_layout` |
| `camera.py` | reuse (vendored) | Trajectory validation and the fused-PRoPE contract |
| `prope.py` | reuse (vendored) | The head-dim `[96:128)` camera projection |
| `sgf.py` | reuse (vendored) | Window geometry and the four-evaluation schedule |
| `sgf_attention.py` | reuse (vendored) | Raw-KV window and the segmented attention |
| `sgf_rollout.py` | reuse (vendored) | The 4-probe + 1-commit chunk loop |
| H3 DiT | adapt | `MiniMaxH3DiTModel` needs a SolarWM attention path |
| AR-Diffusion binding | adapt | Stepwise binding, following LingBot World |
| Offline oracle pipeline | **new** | One request, request-local KV, ten chunks |

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

6. **The whole rollout can be verified without weights.** SolarWM's
   `h3_sgf_rollout` takes the student as a callable, so the reference and the
   port can be driven by one shared stub that routes its output through the
   real SGF attention. With identical inputs the two produce **bit-identical**
   `cache_target` and `noisy_at_exit`, which is what the committed golden
   values encode.

## Status

Landed: the model-semantics layer, the SGF rollout, and their tests. Nothing
here needs the checkpoint or a GPU.

| File | State |
| --- | --- |
| `solarwm_h3/geometry.py` | vendored, byte-identical to upstream |
| `solarwm_h3/layout.py` | vendored, byte-identical |
| `solarwm_h3/camera.py` | vendored, byte-identical |
| `solarwm_h3/torch_geometry.py` | vendored, byte-identical |
| `solarwm_h3/flow.py` | vendored from `torch_flow.py`, byte-identical |
| `solarwm_h3/prope.py` | vendored from `torch_prope.py`, byte-identical (keeps PRoPE's MIT notice) |
| `solarwm_h3/sgf.py` | vendored, one recorded import retarget |
| `solarwm_h3/sgf_attention.py` | vendored, one recorded import retarget |
| `solarwm_h3/sgf_rollout.py` | vendored, three recorded edits (two retargets, SP shim) |
| `solarwm_h3/row_timesteps.py` | ported subset of `torch_layout.py` (rollout entry point only) |
| `tests/diffusion/models/solarwm_h3/` | 30 contract tests, 2 committed fixtures |
| H3 transformer wiring, pipeline, AR-Diffusion binding | not started |

Evidence run locally: every vendored body regenerates from upstream plus only
the recorded edits; a stub-driven 10-chunk rollout is bit-equal to the
reference, including the per-call history (no forward sees its own or a later
chunk) and the commit pattern (four non-committing probes, then one committing
forward, all under `no_grad`).

## Next step

Wire the SGF rollout into a `SolarWMH3Pipeline` that runs as a single request
with request-local KV: the offline oracle. That is the layer that then has to
be re-expressed against the AR-Diffusion stepwise contract, and the one that
needs the checkpoint to be meaningful.

## Scope boundaries

- **Sequence parallelism is not supported.** `sgf_rollout`'s SP broadcast is
  replaced by a shim that raises, so an `sp_enabled=True` run fails loudly
  instead of silently computing the wrong thing. Everything else in the
  vendored layer is SP-agnostic.
- **SGF replay and the Stage 1 / `anyflow` training paths are not carried
  over.** They are not on the inference path.

## Blockers

- **Stage 2 weights are not available.** The server holds only MiniMax-H3
  FL2VA. SolarWM needs `junchaoh-cs/SolarWM-H3-33B` (base + Stage 2 LoRA r384 +
  `H3_SUPPORT/` silence latents), a gated Hugging Face repo.
- **The server is unreachable from here** (`162.105.195.172:22` times out).
- **33B needs 4× A100** at bf16, so GPU validation is a multi-GPU run.

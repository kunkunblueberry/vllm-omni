# AR-Diffusion KV management layers

This page records the implemented KV-management boundary for the experimental paged/segmented and Noisy PP backends. It is deliberately narrower than the full T1 target architecture: it documents shared operations that production code actually calls, and separates backend strategy where storage units or synchronization differ.

## Implemented ownership

| Responsibility | Paged / segmented | Noisy PP | Shared boundary |
| --- | --- | --- | --- |
| Configuration and capacity geometry | `config.py`, `manager.py` | `noisy_spec.py`, `noisy_manager.py` | Same module-layer roles, different geometry and logical capacity units |
| Physical pool and identity | vLLM block pool plus `manager.py` request adapters | `noisy_pool.py` `VersionPool`, keyed by `(request, chunk, step, block)` | `allocate_kv_pool_with_views` and underlying paged attention kernels |
| Per-invocation/session state | `state.py` and `segmented.py` | `noisy_state.py` | Allocation-retention operation in `lifecycle.py` |
| Read/write views | paged contexts or exact-token segment staging | `noisy_context.py` cell-version view | paged attention input contract; layout/view construction remains backend-owned |
| Cross-rank transport | backend-specific; segmented state itself has no Cell P2P protocol | `noisy_transport.py` cell-version P2P and receive visibility | No shared wire representation or implicit layout conversion |
| History policy | window/sink/segment handles | `noisy_retention.py` clean or per-step cells | Session owns cross-request history; policy selects the retained identifiers |

`noisy.py` is kept as a compatibility facade. Existing imports of `NoisyKVCache`, `NoisyKVState`, `VersionPool`, `NoisyKVTransport`, `NoisyLayerContext`, the spec and retention helpers still resolve to the same class/function objects defined in the split modules.

## Shared lifecycle operation

`release_unretained(allocations, retained, release, before_release=None)` is the intentionally small common primitive:

1. The backend validates its retention request and establishes that compute/communication last-use is complete.
2. The primitive calculates the backend-owned allocation keys absent from the retained set.
3. If present, `before_release` runs for **all** dropped keys before any storage is returned. Noisy uses it to retire pending transport receives.
4. The backend callback releases each dropped allocation.

Production callers:

- `ARDiffusionSegmentedKVState._publish_retention`: after commit validation/publication, releases excluded `SegmentHandle` allocations through the paged manager.
- `NoisyKVState.end_request`: after `transport.wait_sends()`, discards pending receives for all excluded `VersionKey`s, then returns their `VersionPool` slots.

The helper does not own storage, last-use tables, transport, transaction publication, or the retained ordering. Paged validates handle ownership/token bounds; Noisy derives the retained version keys from its cell policy. Those rules remain where the backend has enough information to enforce them.

## Deliberate backend differences

A paged token/segment and a Noisy cell version are not the same allocation unit. Their pool indexes, views, attention staging, transfer payloads, and visibility events differ. They continue to use separate storage and transfer implementations and share the attention kernel only where the existing view contract permits it. This refactor adds no paged-to-Noisy conversion.

This is **not** the complete T1 resource protocol. Admission and reporting are one loop inside the worker that owns the pool: the runner installs a ledger over the bytes the pool reserved, admits a session against the window the pool sized for it, reconciles the worker's report, and returns the grant on reset, close, or a failed forward. What remains unimplemented is the CPU service-level side — handing a per-replica budget down, and an eviction policy when a full pool could reclaim an idle session. Failure paths for each backend also retain their existing owners; this helper does not replace the Engine/Runner/Worker cleanup protocol.

## Architecture cross-check

| Definition | Status after this refactor |
| --- | --- |
| `diffusion-unification-analysis.md` §1.2: KV operations and safe retention | Scoped partial implementation: shared allocation, attention and allocation-retention membership/order; backend visibility and physical layouts remain separate. |
| §1.3–1.4: common planning/execution/Batch interface | Noisy PP already runs through the request entry and retains Cell microbatching, but this refactor does not unify paged step/chunk planning with Cell planning or introduce one common Batch object. |
| §1.5: acceptance across models, platforms, output quality and performance | Regression tests cover the affected state/storage and request-path components. This refactor does not establish full-model quality, long-running stability or performance parity on every supported platform. |
| §2.3: Scheduler reserves capacity; Worker reports actual use and returns it | Partial. Admission, the `CapacityUsage` report, reconciliation, and grant return are wired inside the worker that owns the pool. A CPU-side reservation that hands a per-replica budget down, and distributed per-worker grant distribution, are still open. |
| §2.4: separate planning from execution | Existing Noisy PP plan/executor and capability adapter are preserved; KV refactor does not merge planning strategies or complete the shared execution interface. |
| §2.5: model adaptation, conditioning/latent/decoder persistence | Existing model capability binding and output paths are untouched. Cross-request conditioning/latent/VAE temporal-state persistence is not added or claimed by this KV-only refactor. |
| `diffusion_noisy_pp_architecture.md` §2.7: common KV representation/operations plus unified eviction conditions | Partial and scoped: allocation-retention membership/order is shared and used by both backends. Physical representation, pool access, publication, visibility/last-use, communication and release callbacks remain backend-specific. |
| §4.2: reuse storage capabilities; common KV lifecycle contract | Partially implemented: shared pool-allocation helper, attention kernel and lifecycle-set operation. There is not one universal pool or state object, intentionally; logical units and physical layouts differ. |
| §4.3: affinity, state ownership, visibility, last-use, cross-request retention, failures | Affinity, retained history, per-backend visibility/last-use and existing failure paths remain separately implemented. Shared helper receives only allocations whose backend already considers reclaimable. Cross-backend lifecycle parity is not claimed. |

The tests pin both sides of the shared operation: direct ordering/ownership/failure behavior, actual segmented retention, actual Noisy end-request transport-before-slot-release, and compatibility-facade object identity.

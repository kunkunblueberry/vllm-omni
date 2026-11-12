# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-invocation Noisy PP state: plans, last-use, visibility and prepare views."""

from __future__ import annotations

from collections.abc import Collection, Iterable
from typing import Any

import torch

from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.noisy_pp_plan import (
    PRIOR_HANDOFF_SLOT,
    Cell,
    CellPlan,
    Inflight,
    RequestKVTransfer,
    union_transfers,
    union_wait_ready,
)
from vllm_omni.experimental.ar_diffusion.kv_cache.lifecycle import release_unretained
from vllm_omni.experimental.ar_diffusion.kv_cache.noisy_context import NoisyLayerContext
from vllm_omni.experimental.ar_diffusion.kv_cache.noisy_manager import NoisyKVCache
from vllm_omni.experimental.ar_diffusion.kv_cache.noisy_pool import VersionKey, _version_key


class NoisyKVState:
    """Per-invocation bridge used by the chunk executor over a shared pool."""

    def __init__(self, cache: NoisyKVCache) -> None:
        self.cache = cache
        self._plans: dict[str, CellPlan] = {}
        self._chunk_tokens: dict[str, int] = {}
        self._last_use_global: dict[str, dict[Cell, int]] = {}
        self._retain: dict[str, set[Cell]] = {}
        self.bytes_sent = 0
        self.bytes_received = 0
        self.transfers_deduped = 0
        self._inflight: tuple[Inflight, ...] = ()
        self._rank = 0
        self._pp_group: Any | None = None
        self._write_slots_cache: dict[tuple[int, int], torch.Tensor] = {}
        self._query_start_cache: dict[int, torch.Tensor] = {}
        self._block_table_dev: torch.Tensor | None = None
        self._block_table_cpu: torch.Tensor | None = None
        self._block_table_h2d_event: Any | None = None
        self._seq_lens_dev: torch.Tensor | None = None
        self._prepare_buf_width: int = -1

    @property
    def resident_versions(self) -> int:
        return len(self.cache.pool.keys)

    @property
    def uses_comm_stream(self) -> bool:
        return self.cache.transport.uses_comm_stream

    def bind_rank(self, rank: int, pp_group: Any | None) -> None:
        self._rank = rank
        self._pp_group = pp_group

    def wait_sends(self) -> None:
        self.cache.transport.wait_sends()

    def protect(self, req: str, cells: Iterable[Cell]) -> None:
        """Keep ``cells`` for ``req`` out of last-use eviction and end_request release."""
        bucket = self._retain.setdefault(req, set())
        bucket.update(cells)

    def begin_request(
        self,
        req: str,
        plan: CellPlan,
        *,
        chunk_tokens: int,
        t0: int = 0,
        retain: Collection[Cell] | None = None,
    ) -> None:
        if chunk_tokens <= 0 or chunk_tokens % self.cache.spec.block_size != 0:
            raise ValueError("chunk_tokens must be a positive multiple of block_size")
        if chunk_tokens > self.cache.spec.max_chunk_tokens:
            raise ValueError("chunk_tokens exceeds max_chunk_tokens")
        self._plans[req] = plan
        self._chunk_tokens[req] = chunk_tokens
        mapped = {version: t0 + slot for version, slot in plan.last_use(self._rank).items()}
        self._last_use_global[req] = mapped
        if retain:
            self.protect(req, retain)

    def chunk_tokens(self, req: str) -> int:
        """Public read for mainline ``SupportsNoisyPPKV`` (admit token-uniformity)."""
        return self._chunk_tokens[req]

    def set_inflight(self, inflight: tuple[Inflight, ...]) -> None:
        self._inflight = inflight

    def _is_retained(self, req: str, version: Cell) -> bool:
        return version in self._retain.get(req, ())

    def end_request(self, req: str, *, retain: Collection[Cell] | None = None) -> None:
        """Finish one admitted request.

        ``retain is None`` keeps the existing protect set (normal plan
        completion). ``retain=()`` clears retained versions for ``req``.
        """
        self.cache.transport.wait_sends()
        if retain is not None:
            self._retain[req] = set(retain)
        keep = {_version_key(req, cell) for cell in self._retain.get(req, ())}
        owned = tuple(key for key in self.cache.pool.keys if key[0] == req)
        release_unretained(
            owned,
            keep,
            self.cache.pool.release,
            before_release=self.cache.transport.discard,
        )
        self._plans.pop(req, None)
        self._chunk_tokens.pop(req, None)
        self._last_use_global.pop(req, None)
        if not keep:
            self._retain.pop(req, None)

    def release_session(self, session_id: str) -> None:
        """Drop every pool version and bookkeeping for one session id."""
        self.end_request(session_id, retain=())
        leftover = [key for key in self.cache.pool.keys if key[0] == session_id]
        for key in leftover:
            self.cache.transport.discard(key)
            self.cache.pool.release(key)
        self._retain.pop(session_id, None)
        self.cache.clear_session_structure(session_id)

    def reset_all(self) -> None:
        for req in list(set(self._plans) | set(self._retain)):
            self.end_request(req, retain=())
        leftover = list(self.cache.pool.keys)
        for key in leftover:
            self.cache.transport.discard(key)
            self.cache.pool.release(key)
        self._plans.clear()
        self._chunk_tokens.clear()
        self._last_use_global.clear()
        self._retain.clear()
        self.cache._session_structural_keys.clear()
        self._inflight = ()
        self._clear_prepare_caches()

    def _clear_prepare_caches(self) -> None:
        self._write_slots_cache.clear()
        self._query_start_cache.clear()
        self._block_table_dev = None
        self._block_table_cpu = None
        self._block_table_h2d_event = None
        self._seq_lens_dev = None
        self._prepare_buf_width = -1

    def _ensure_prepare_bufs(self, width: int, device: torch.device) -> None:
        if (
            self._block_table_dev is not None
            and self._prepare_buf_width == width
            and self._block_table_dev.device == device
        ):
            return
        pin = device.type == "cuda"
        self._block_table_cpu = torch.zeros(1, width, dtype=torch.int32, pin_memory=pin)
        self._block_table_dev = torch.zeros(1, width, dtype=torch.int32, device=device)
        self._seq_lens_dev = torch.zeros(1, dtype=torch.int32, device=device)
        self._prepare_buf_width = width
        self._block_table_h2d_event = None
        self._write_slots_cache.clear()
        self._query_start_cache.clear()

    def _wait_block_table_h2d(self) -> None:
        """Keep pinned ``_block_table_cpu`` live until the prior async H2D finishes."""
        event = self._block_table_h2d_event
        if event is not None:
            event.synchronize()

    def _record_block_table_h2d(self, device: torch.device) -> None:
        if device.type != "cuda":
            self._block_table_h2d_event = None
            return
        if self._block_table_h2d_event is None:
            self._block_table_h2d_event = torch.cuda.Event()
        self._block_table_h2d_event.record()

    def _cached_query_start(self, chunk_tokens: int, device: torch.device) -> torch.Tensor:
        hit = self._query_start_cache.get(chunk_tokens)
        if hit is not None and hit.device == device:
            return hit
        qsl = torch.tensor([0, chunk_tokens], dtype=torch.int32, device=device)
        self._query_start_cache[chunk_tokens] = qsl
        return qsl

    def _cached_write_slots(self, write_slot: int, chunk_tokens: int, device: torch.device) -> torch.Tensor:
        """Contiguous write-slot mapping for one VersionPool slot."""
        key = (write_slot, chunk_tokens)
        hit = self._write_slots_cache.get(key)
        if hit is not None and hit.device == device:
            return hit
        pool = self.cache.pool
        base = write_slot * pool.chunk_blocks * pool.block_size
        slots = torch.arange(chunk_tokens, dtype=torch.long, device=device) + base
        self._write_slots_cache[key] = slots
        return slots

    def prepare(self, tasks: tuple[tuple[str, Cell], ...]) -> list[list[NoisyLayerContext]]:
        """One ``NoisyLayerContext`` per cell (single block)."""
        spec = self.cache.spec
        pool = self.cache.pool
        h = spec.max_history_chunks
        sink = int(any(plan.schedule.same_step_has_sink for plan in self._plans.values()))
        width = (h + 1 + sink) * pool.chunk_blocks
        max_seq_len = (h + 1 + sink) * spec.max_chunk_tokens
        max_query_len = self.cache.max_batch_size * spec.max_chunk_tokens
        batch: list[list[NoisyLayerContext]] = []
        device = pool.k_pools[0].device
        self._ensure_prepare_bufs(width, device)
        assert self._block_table_cpu is not None and self._block_table_dev is not None
        assert self._seq_lens_dev is not None
        share_bufs = len(tasks) <= 1
        for req, task in tasks:
            chunk_tokens = self._chunk_tokens[req]
            num_chunk_blocks = chunk_tokens // pool.block_size
            plan = self._plans[req]
            srcs = plan.sources(task, self._rank)
            write_key = _version_key(req, task)
            write_slot = pool.alloc(write_key)
            write_blocks = pool.block_ids(write_slot)[:num_chunk_blocks]
            blocks: list[int] = []
            for src in srcs:
                src_key = _version_key(req, src.version)
                if not pool.has(src_key):
                    raise RuntimeError(f"I5: source {src_key} missing at prepare")
                blocks.extend(pool.block_ids(pool.slot_of(src_key))[:num_chunk_blocks])
            blocks.extend(write_blocks)
            if len(blocks) > width:
                raise RuntimeError(f"block_table width {width} too small for {len(blocks)} blocks")
            # Do not overwrite pinned CPU source while a prior non_blocking H2D is in flight.
            self._wait_block_table_h2d()
            self._block_table_cpu[0, : len(blocks)] = torch.as_tensor(blocks, dtype=torch.int32)
            if len(blocks) < width:
                self._block_table_cpu[0, len(blocks) :].zero_()
            seq_len = (len(srcs) + 1) * chunk_tokens
            self._block_table_dev.copy_(self._block_table_cpu, non_blocking=device.type == "cuda")
            self._record_block_table_h2d(device)
            self._seq_lens_dev.fill_(seq_len)
            query_start_loc = self._cached_query_start(chunk_tokens, device)
            block_table = self._block_table_dev if share_bufs else self._block_table_dev.clone()
            seq_lens = self._seq_lens_dev if share_bufs else self._seq_lens_dev.clone()
            video_slots = self._cached_write_slots(write_slot, chunk_tokens, device)
            local_layer = pool.local_layer(write_key)
            batch.append(
                [
                    NoisyLayerContext(
                        layer_idx=local_layer,
                        key_pool=pool.k_pools[local_layer],
                        value_pool=pool.v_pools[local_layer],
                        block_size=pool.block_size,
                        video_slots=video_slots,
                        block_table=block_table,
                        query_start_loc=query_start_loc,
                        seq_lens=seq_lens,
                        max_query_len=max_query_len,
                        max_seq_len=max_seq_len,
                        seq_len=seq_len,
                    )
                ]
            )
        self.cache.resident_peak = max(self.cache.resident_peak, self.resident_versions)
        return batch

    def publish(self, tasks: tuple[tuple[str, Cell], ...]) -> None:
        del tasks

    def exchange(self, slot: int) -> list[Any]:
        transfers = union_transfers(self._inflight, slot)
        self.transfers_deduped = len(transfers)
        handles = self.cache.transport.exchange(
            transfers, rank=self._rank, pp_group=self._pp_group, chunk_tokens_by_request=self._chunk_tokens
        )
        self.bytes_sent = self.cache.transport.bytes_sent
        self.bytes_received = self.cache.transport.bytes_received
        return handles

    def exchange_and_await_prior_handoffs(self, global_slot: int) -> list[Any]:
        """Exchange prior-history KV for requests whose first cell is ``global_slot``.

        Must run before ``prepare`` of that first cell: the normal per-slot
        ``exchange`` happens after forward, which is too late for continuation
        history that is only resident on the owner rank.
        """
        transfers: list[RequestKVTransfer] = []
        wanted: set[VersionKey] = set()
        for item in self._inflight:
            if global_slot != item.t0:
                continue
            for xfer in item.plan.transfers(PRIOR_HANDOFF_SLOT):
                transfers.append(RequestKVTransfer(req=item.req, version=xfer.version, src=xfer.src, dst=xfer.dst))
                if xfer.dst == self._rank:
                    wanted.add(_version_key(item.req, xfer.version))
        if not transfers:
            return []
        ordered = tuple(sorted(transfers, key=lambda x: (x.src, x.dst, x.req, x.version)))
        handles = self.cache.transport.exchange(
            ordered, rank=self._rank, pp_group=self._pp_group, chunk_tokens_by_request=self._chunk_tokens
        )
        self.bytes_sent = self.cache.transport.bytes_sent
        self.bytes_received = self.cache.transport.bytes_received
        for handle in handles:
            if handle is not None:
                handle.wait()
        if self.uses_comm_stream:
            self.cache.transport.wait_sends()
        if wanted:
            self.cache.transport.await_ready(frozenset(wanted))
            self.bytes_received = self.cache.transport.bytes_received
        return handles

    def evict(self, slot: int) -> None:
        if self.uses_comm_stream:
            self.cache.transport.wait_sends()
        drop: list[VersionKey] = []
        for req, used in self._last_use_global.items():
            for version, last in used.items():
                if last == slot and not self._is_retained(req, version):
                    drop.append(_version_key(req, version))
        for key in drop:
            self.cache.transport.discard(key)
            self.cache.pool.release(key)
        self.cache.resident_peak = max(self.cache.resident_peak, self.resident_versions)

    def await_ready(self, slot: int) -> int:
        wanted = {_version_key(req, version) for req, version in union_wait_ready(self._inflight, slot, self._rank)}
        pending = self.cache.transport.await_ready(frozenset(wanted))
        self.bytes_sent = self.cache.transport.bytes_sent
        self.bytes_received = self.cache.transport.bytes_received
        return pending

    def await_task_sources(self, tasks: tuple[tuple[str, Cell], ...]) -> int:
        """Wait remote history KV before attention reads it (first or later consume)."""
        wanted: set[VersionKey] = set()
        for req, task in tasks:
            plan = self._plans.get(req)
            if plan is None:
                continue
            for src in plan.sources(task, self._rank):
                if src.owner != self._rank:
                    wanted.add(_version_key(req, src.version))
        pending = self.cache.transport.await_ready(frozenset(wanted))
        self.bytes_sent = self.cache.transport.bytes_sent
        self.bytes_received = self.cache.transport.bytes_received
        return pending

    def drain(self) -> None:
        self.cache.transport.drain_all()
        self.bytes_received = self.cache.transport.bytes_received

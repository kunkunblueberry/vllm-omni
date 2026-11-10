# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cell-version KV transport and asynchronous visibility tracking."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import torch.distributed as dist

from vllm_omni.diffusion.distributed.noisy_pipeline_parallel.noisy_pp_plan import RequestKVTransfer
from vllm_omni.experimental.ar_diffusion.kv_cache.noisy_pool import VersionKey, VersionPool, _version_key


def _payload_bytes(payload: dict[str, torch.Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for key, tensor in payload.items() if key in ("k", "v"))


def _wait_handles(handles: list[Any] | None) -> None:
    for handle in handles or ():
        if handle is not None:
            handle.wait()


def kv_comm_stream_enabled() -> bool:
    """Dedicated CUDA stream + deferred send wait (disabled; default transport only)."""
    return False


def kv_use_tensor_dict() -> bool:
    """Legacy Gloo-metadata KV path (disabled; hot path only)."""
    return False


def _pg_for_tensor(pp_group: Any, tensor: torch.Tensor) -> Any:
    return pp_group.cpu_group if tensor.is_cpu else pp_group.device_group


def _global_rank(pp_group: Any, group_rank: int) -> int:
    return int(pp_group.ranks[group_rank])


class NoisyKVTransport:
    def __init__(self, pool: VersionPool) -> None:
        self.pool = pool
        self._pending_recv: dict[VersionKey, tuple[list[Any], dict[str, torch.Tensor] | None, int]] = {}
        self._send_handles: list[Any] = []
        self._send_keepalives: list[torch.Tensor] = []
        self._comm_stream: torch.cuda.Stream | None = None
        self.bytes_sent = 0
        self.bytes_received = 0

    @property
    def uses_comm_stream(self) -> bool:
        return kv_comm_stream_enabled() and not kv_use_tensor_dict()

    def _ensure_comm_stream(self) -> torch.cuda.Stream | None:
        if not self.uses_comm_stream or self.pool.device.type != "cuda":
            return None
        if self._comm_stream is None:
            self._comm_stream = torch.cuda.Stream(device=self.pool.device)
        return self._comm_stream

    def exchange(
        self,
        transfers: tuple[RequestKVTransfer, ...],
        *,
        rank: int,
        pp_group: Any | None,
        chunk_tokens_by_request: dict[str, int],
    ) -> list[Any]:
        """Post transfers. Returns send handles, or ``[]`` when deferred to ``wait_sends``."""
        self._send_handles = []
        self._send_keepalives = []
        if pp_group is None or getattr(pp_group, "world_size", 1) <= 1:
            return []
        if kv_use_tensor_dict():
            return self._exchange_tensor_dict(
                transfers, rank=rank, pp_group=pp_group, chunk_tokens_by_request=chunk_tokens_by_request
            )
        defer = self.uses_comm_stream
        self._exchange_fixed_nccl(
            transfers,
            rank=rank,
            pp_group=pp_group,
            chunk_tokens_by_request=chunk_tokens_by_request,
            use_stream=defer,
        )
        if defer:
            return []
        # Caller owns waiting these handles (cell_executor / tests). Clear so
        # wait_sends/end_request do not wait again — gloo double-wait hangs.
        handles = list(self._send_handles)
        self._send_handles = []
        return handles

    def _exchange_tensor_dict(
        self,
        transfers: tuple[RequestKVTransfer, ...],
        *,
        rank: int,
        pp_group: Any,
        chunk_tokens_by_request: dict[str, int],
    ) -> list[Any]:
        n_send = n_recv = 0
        for xfer in transfers:
            key = _version_key(xfer.req, xfer.version)
            if xfer.src == rank:
                num_blocks = chunk_tokens_by_request[xfer.req] // self.pool.block_size
                payload = self.pool.tensor_dict(key, num_blocks)
                self.bytes_sent += _payload_bytes(payload)
                self._send_handles.extend(pp_group.isend_tensor_dict(payload, dst=xfer.dst))
                n_send += 1
            elif xfer.dst == rank:
                self.pool.alloc(key)
                payload, recv_handles, _ = pp_group.irecv_tensor_dict(src=xfer.src)
                nbytes = _payload_bytes(payload)
                merged = list(self._pending_recv.get(key, ([], None, 0))[0] or [])
                merged.extend(recv_handles)
                self._pending_recv[key] = (merged, payload, nbytes)
                n_recv += 1
        handles = list(self._send_handles)
        self._send_handles = []
        return handles

    def _exchange_fixed_nccl(
        self,
        transfers: tuple[RequestKVTransfer, ...],
        *,
        rank: int,
        pp_group: Any,
        chunk_tokens_by_request: dict[str, int],
        use_stream: bool,
    ) -> None:
        """Fixed-shape K/V P2P via one ``batch_isend_irecv``."""
        stream = self._ensure_comm_stream() if use_stream else None
        n_send = n_recv = 0
        if stream is not None:
            stream.wait_stream(torch.cuda.current_stream(self.pool.device))

        ops: list[dist.P2POp] = []
        send_kept: list[torch.Tensor] = []
        recv_jobs: list[tuple[VersionKey, dict[str, torch.Tensor] | None, int]] = []

        for xfer in transfers:
            key = _version_key(xfer.req, xfer.version)
            num_blocks = chunk_tokens_by_request[xfer.req] // self.pool.block_size
            if xfer.src == rank:
                k_view, v_view = self.pool.kv_block_views(key, num_blocks)
                k = k_view if k_view.is_contiguous() else k_view.contiguous()
                v = v_view if v_view.is_contiguous() else v_view.contiguous()
                nbytes = k.numel() * k.element_size() + v.numel() * v.element_size()
                self.bytes_sent += nbytes
                dst_global = _global_rank(pp_group, xfer.dst)
                for tensor in (k, v):
                    group = _pg_for_tensor(pp_group, tensor)
                    ops.append(dist.P2POp(dist.isend, tensor, dst_global, group))
                    if tensor.is_cuda:
                        tensor.record_stream(stream or torch.cuda.current_stream(tensor.device))
                    send_kept.append(tensor)
                n_send += 1
            elif xfer.dst == rank:
                self.pool.alloc(key)
                k_view, v_view = self.pool.kv_block_views(key, num_blocks)
                if k_view.is_contiguous() and v_view.is_contiguous():
                    k_buf, v_buf = k_view, v_view
                    staging: dict[str, torch.Tensor] | None = None
                else:
                    k_buf, v_buf = k_view.contiguous(), v_view.contiguous()
                    staging = {
                        "k": k_buf,
                        "v": v_buf,
                        "layer": torch.tensor([self.pool.local_layer(key)], dtype=torch.int32),
                    }
                nbytes = k_buf.numel() * k_buf.element_size() + v_buf.numel() * v_buf.element_size()
                src_global = _global_rank(pp_group, xfer.src)
                for tensor in (k_buf, v_buf):
                    group = _pg_for_tensor(pp_group, tensor)
                    ops.append(dist.P2POp(dist.irecv, tensor, src_global, group))
                    if tensor.is_cuda and stream is not None:
                        tensor.record_stream(stream)
                recv_jobs.append((key, staging, nbytes))
                n_recv += 1

        ctx = torch.cuda.stream(stream) if stream is not None else nullcontext()
        with ctx:
            works = list(dist.batch_isend_irecv(ops)) if ops else []
        if n_send:
            self._send_handles.extend(works)
            self._send_keepalives.extend(send_kept)
        for key, staging, nbytes in recv_jobs:
            merged = list(self._pending_recv.get(key, ([], None, 0))[0] or [])
            if not merged:
                merged = list(works)
            else:
                merged.extend(works)
            self._pending_recv[key] = (merged, staging, nbytes)

    def wait_sends(self) -> None:
        """Host-wait deferred outbound KV (comm-stream mode)."""
        if self._send_handles:
            _wait_handles(self._send_handles)
        self._send_handles = []
        self._send_keepalives = []

    def await_ready(self, wanted: frozenset[VersionKey]) -> int:
        if not self._pending_recv:
            return 0
        ready = [key for key in self._pending_recv if key in wanted]
        for key in ready:
            handles, payload, nbytes = self._pending_recv.pop(key)
            _wait_handles(handles)
            self.bytes_received += nbytes
            if payload is not None:
                self.pool.copy_into(key, payload)
            if self._comm_stream is not None and self.pool.device.type == "cuda":
                torch.cuda.current_stream(self.pool.device).wait_stream(self._comm_stream)
        return len(self._pending_recv)

    def drain_all(self) -> None:
        self.wait_sends()
        for key in list(self._pending_recv):
            handles, payload, nbytes = self._pending_recv.pop(key)
            _wait_handles(handles)
            self.bytes_received += nbytes
            if payload is not None:
                self.pool.copy_into(key, payload)
        if self._comm_stream is not None and self.pool.device.type == "cuda":
            torch.cuda.current_stream(self.pool.device).wait_stream(self._comm_stream)

    def discard(self, key: VersionKey) -> None:
        pending = self._pending_recv.pop(key, None)
        if pending is not None:
            handles, payload, nbytes = pending
            _wait_handles(handles)
            self.bytes_received += nbytes
            del payload

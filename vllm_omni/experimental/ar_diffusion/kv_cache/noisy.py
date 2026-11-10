# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compatibility exports for Noisy PP KV components.

Implementation is split by ownership: spec, physical version pool, layer context,
transport, cache allocation, retention policy and per-invocation state. Keep this
module as the stable import path used by existing model and test integrations.
"""

from vllm_omni.experimental.ar_diffusion.kv_cache.noisy_context import NoisyLayerContext
from vllm_omni.experimental.ar_diffusion.kv_cache.noisy_manager import NoisyKVCache
from vllm_omni.experimental.ar_diffusion.kv_cache.noisy_pool import VersionKey, VersionPool, _layer_of, _version_key
from vllm_omni.experimental.ar_diffusion.kv_cache.noisy_retention import clean_history_cells, prior_history_in_pool
from vllm_omni.experimental.ar_diffusion.kv_cache.noisy_spec import ARDiffusionNoisyKVSpec
from vllm_omni.experimental.ar_diffusion.kv_cache.noisy_state import NoisyKVState
from vllm_omni.experimental.ar_diffusion.kv_cache.noisy_transport import (
    NoisyKVTransport,
    _global_rank,
    _payload_bytes,
    _pg_for_tensor,
    _wait_handles,
    kv_comm_stream_enabled,
    kv_use_tensor_dict,
)

__all__ = [
    "ARDiffusionNoisyKVSpec",
    "NoisyKVCache",
    "NoisyKVState",
    "NoisyKVTransport",
    "NoisyLayerContext",
    "VersionKey",
    "VersionPool",
    "clean_history_cells",
    "kv_comm_stream_enabled",
    "kv_use_tensor_dict",
    "prior_history_in_pool",
    "_global_rank",
    "_layer_of",
    "_payload_bytes",
    "_pg_for_tensor",
    "_version_key",
    "_wait_handles",
]

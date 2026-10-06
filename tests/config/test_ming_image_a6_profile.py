# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Typed deploy-to-runtime tests for the correct Ming-Image model."""

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.engine.test_async_omni_engine_stage_init import _make_diffusion_plan
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.omni_config import VllmOmniConfig
from vllm_omni.config.stage_config import load_deploy_config
from vllm_omni.diffusion import inline_stage_diffusion_client, stage_diffusion_client
from vllm_omni.engine import stage_init_utils, stage_runtime
from vllm_omni.engine.stage_engine_startup import StageReplicaResources
from vllm_omni.model_executor.models.ming_image.pipeline import MING_IMAGE_PIPELINE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("profile", "devices", "cfg_size", "inline"),
    [
        ("ming_image.yaml", ["0", "1"], 1, False),
        ("ming_image_single_gpu.yaml", ["0", "0"], 1, True),
        ("ming_image_cfg_parallel.yaml", ["0", "1,2"], 2, False),
    ],
)
def test_typed_ming_profiles_select_runtime_client(monkeypatch, profile, devices, cfg_size, inline):
    deploy = load_deploy_config(get_deploy_config_path(profile))
    config = VllmOmniConfig.from_pipeline_config(MING_IMAGE_PIPELINE, user_deploy_config=deploy)
    assert [stage.runtime_config.devices for stage in config.stage_configs] == devices
    assert config.stage_by_id(0).scheduler_config.max_num_seqs == 1
    stage = config.stage_by_id(1)
    assert stage.parallel_config.cfg_parallel_size == cfg_size
    assert stage.stage_pipeline_config.inline_diffusion is inline
    assert MING_IMAGE_PIPELINE.stages[1].inline_diffusion is False

    runtime = stage_runtime.StageRuntime(
        stage_configs=list(config.stage_configs),
        model="local-ming",
        config_path=None,
        stage_init_timeout=1,
        async_chunk=False,
    )
    plan = _make_diffusion_plan(1, stage_id=1).replicas[0]
    plan.stage_cfg = stage
    plan.metadata.runtime_cfg = {"devices": devices[1]}
    captured = {}
    monkeypatch.setattr(stage_runtime, "inject_kv_stage_info", lambda *args: None)
    monkeypatch.setattr(runtime, "_stage_device_scope", lambda *args: nullcontext())

    def launch(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(), StageReplicaResources()

    monkeypatch.setattr(stage_runtime, "launch_diffusion_stage_replica", launch)
    runtime._initialize_local_diffusion_replica(plan, stage_init_timeout=1)
    assert captured["use_inline"] is inline

    # Keep the actual typed engine-args and OmniDiffusionConfig conversion.
    # Only device enumeration is replaced; no client, GPU or model is loaded.
    monkeypatch.setattr(
        stage_init_utils,
        "current_omni_platform",
        SimpleNamespace(device_control_env_var=None, get_device_count=lambda: 3),
    )
    od_config = stage_init_utils.build_diffusion_config("local-ming", stage, plan.metadata)
    assert od_config.parallel_config.cfg_parallel_size == cfg_size
    assert od_config.num_gpus == cfg_size

    # Run the real factory, replacing client constructors rather than the selection.
    inline_client, proc_client = object(), object()
    monkeypatch.setattr(inline_stage_diffusion_client, "InlineStageDiffusionClient", lambda *args: inline_client)
    monkeypatch.setattr(
        stage_diffusion_client,
        "StageDiffusionProcManager",
        lambda **kwargs: SimpleNamespace(addresses=SimpleNamespace(inputs=["in"], outputs=["out"])),
    )
    monkeypatch.setattr(
        stage_diffusion_client.StageDiffusionClient, "from_addresses", lambda *args, **kwargs: proc_client
    )
    result = stage_diffusion_client.create_diffusion_client(
        "local-ming", od_config, plan.metadata, 1, use_inline=captured["use_inline"]
    )
    assert result is (inline_client if inline else proc_client)


def test_explicit_false_overrides_topology_inline_without_mutating_registry():
    deploy = load_deploy_config(get_deploy_config_path("ming_image_single_gpu.yaml"))
    deploy.stages[1].engine_extras["inline_diffusion"] = False
    topology = replace(
        MING_IMAGE_PIPELINE,
        stages=(MING_IMAGE_PIPELINE.stages[0], replace(MING_IMAGE_PIPELINE.stages[1], inline_diffusion=True)),
    )
    config = VllmOmniConfig.from_pipeline_config(topology, user_deploy_config=deploy)
    assert config.stage_by_id(1).stage_pipeline_config.inline_diffusion is False
    assert topology.stages[1].inline_diffusion is True


def test_typed_inline_rejects_string_boolean():
    deploy = load_deploy_config(get_deploy_config_path("ming_image_single_gpu.yaml"))
    deploy.stages[1].engine_extras["inline_diffusion"] = "false"
    with pytest.raises(ValueError, match="inline_diffusion must be a boolean"):
        VllmOmniConfig.from_pipeline_config(MING_IMAGE_PIPELINE, user_deploy_config=deploy)

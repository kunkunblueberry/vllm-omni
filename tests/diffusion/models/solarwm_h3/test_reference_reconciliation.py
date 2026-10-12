# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Independent fixed-profile invariants retained from the original port."""
import pytest
import torch

from vllm_omni.diffusion.models.solarwm_h3.layout import build_layout, full_length_geometry
from vllm_omni.diffusion.models.solarwm_h3.sampling import stage2_timesteps

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_released_profile_and_modality_timesteps():
    geometry = full_length_geometry(158)
    assert geometry.decode_latents == 47
    assert geometry.rollout_latents == 50
    layout = build_layout(torch.tensor([1] * 40 + [0] * 24), latent_height=48, latent_width=84)
    assert layout.sequence_length == 64 + 1008 + 526 + 5040
    assert layout.condition_video_indices.numel() == 1008
    assert layout.audio_indices.numel() == 526
    assert layout.noisy_video_indices.numel() == 5040
    assert not torch.isin(layout.video_indices, layout.text_indices).any()
    # Independent closed-form grid; four probes use five sigma grid points.
    base = torch.tensor([1.0, 0.75, 0.5, 0.25])
    expected = 1.0 - 12 * base / (1 + 11 * base)
    torch.testing.assert_close(stage2_timesteps(), expected, atol=0, rtol=0)

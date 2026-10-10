# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""H3 row-timestep grouping for SolarWM's SGF rollout.

Ported from SolarWM (Apache-2.0), commit ce1da4e7705391eda8eeda6016c0fd3f614b975e,
``src/solarwm/backends/minimax_h3/torch_layout.py:build_row_timesteps``. Only the
rollout entry point is carried over; that module's stage-1-only helpers
(``build_stage1_row_timesteps`` and the ``(t, r)`` interval plans) belong to SGF
training/replay and are not on the inference path.

``build_row_timesteps`` collapses the packed document's per-row timesteps into
the ``(distinct_timesteps, row_to_index)`` pair the DiT consumes as AdaLN
indices. Every row class keeps its own time: text inherits the video time in
Stage0.5, the VisualVAE anchor stays at its fixed noise-augmentation level,
audio carries an independent shift-3 sample, and only noisy video rows follow
the sampled video timestep.
"""

from __future__ import annotations

from typing import Any

import torch

from .layout import H3PackedLayout


def build_row_timesteps(
    layout: H3PackedLayout,
    video_timestep: Any,
    audio_timestep: Any,
    *,
    text_timestep: Any | None = None,
    condition_video_timestep: Any = 0.999,
    clean_video_timestep: Any = 1.0,
    device: Any = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build H3's ``(distinct_timesteps, row_to_timestep_index)`` pair.

    ``layout`` must already be materialized on ``device`` (its index arrays are
    read through ``.to(device=...)``). Stage0.5 requires a scalar
    ``video_timestep``; the Stage1 branch accepts a scalar, one value per chunk,
    or one value per latent frame.
    """

    def tensor(value: Any) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=device).flatten()

    def scalar(value: Any, name: str) -> torch.Tensor:
        resolved = torch.as_tensor(value, dtype=torch.float32, device=device)
        if resolved.numel() != 1:
            raise ValueError(f"{name} must be scalar, got {tuple(resolved.shape)}")
        return resolved.reshape(())

    video_values = tensor(video_timestep)
    if layout.stage == "stage0p5":
        if video_values.numel() != 1:
            raise ValueError("Stage0.5 video_timestep must be scalar")
        noisy_row_values = video_values.expand(layout.noisy_video_indices.numel())
        default_text = video_values[0]
    elif layout.stage == "stage1":
        num_frames = layout.noisy_video_indices.numel() // layout.rows_per_video_frame
        num_chunks = num_frames // int(layout.chunk_latent_frames)  # type: ignore[arg-type]
        if video_values.numel() == 1:
            frame_values = video_values.expand(num_frames)
        elif video_values.numel() == num_chunks:
            frame_values = video_values.repeat_interleave(int(layout.chunk_latent_frames))  # type: ignore[arg-type]
        elif video_values.numel() == num_frames:
            frame_values = video_values
        else:
            raise ValueError(
                "Stage1 video_timestep must be scalar, [num_chunks] or [num_latent_frames]; "
                f"got {video_values.numel()} values for {num_chunks} chunks/{num_frames} frames"
            )
        noisy_row_values = frame_values.repeat_interleave(layout.rows_per_video_frame)
        default_text = torch.tensor(1.0, dtype=torch.float32, device=device)
    else:
        raise ValueError(f"unsupported layout stage {layout.stage!r}")

    row_timesteps = torch.zeros(layout.sequence_length, dtype=torch.float32, device=device)
    row_timesteps[layout.text_indices.to(device=device)] = scalar(
        default_text if text_timestep is None else text_timestep, "text_timestep"
    )
    row_timesteps[layout.condition_video_indices.to(device=device)] = scalar(
        condition_video_timestep, "condition_video_timestep"
    )
    row_timesteps[layout.audio_indices.to(device=device)] = scalar(audio_timestep, "audio_timestep")
    if int(layout.clean_video_indices.numel()):
        row_timesteps[layout.clean_video_indices.to(device=device)] = scalar(
            clean_video_timestep, "clean_video_timestep"
        )
    row_timesteps[layout.noisy_video_indices.to(device=device)] = noisy_row_values
    return torch.unique(row_timesteps, sorted=True, return_inverse=True)


__all__ = ["build_row_timesteps"]

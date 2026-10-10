# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Contract tests for the vendored SolarWM-H3 packed-row layout.

The expected values in ``fixtures/solarwm_h3_stage0p5_layout.npz.fixture`` were
generated from the SolarWM reference implementation (Apache-2.0, commit
ce1da4e7, ``src/solarwm/backends/minimax_h3/layout.py``) for the exact rollout
document the SGF student consumes::

    build_stage0p5_layout(tags, num_latent_frames=5, latent_height=48,
                          latent_width=84, num_audio_latents=263)

They are the independent oracle for the vendored copy -- the expectations are
never taken from the code under test. The independent scalars asserted here
(1008 rows per latent, 263 audio latents per channel, five latent frames per
chunk) come from the H3 checkpoint config, not from the layout builder.

These are plain ``assert`` functions with no pytest fixtures so they also run
outside pytest.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from vllm_omni.diffusion.models.solarwm_h3.geometry import (
    frame_position_grid,
    temporal_position_grid,
)
from vllm_omni.diffusion.models.solarwm_h3.layout import (
    AUDIO_TAG,
    TEXT_TAG,
    VIDEO_TAG,
    build_stage0p5_layout,
    patchify_video,
    unpatchify_video,
)
from vllm_omni.diffusion.models.solarwm_h3.row_timesteps import build_row_timesteps

_FIXTURE_PATH = Path(__file__).with_name("fixtures") / "solarwm_h3_stage0p5_layout.npz.fixture"

# H3 checkpoint contract: [1, 2, 2] patch over a 48x84 latent grid.
ROWS_PER_LATENT = 1008
LATENT_FRAMES = 5
LATENT_HEIGHT = 48
LATENT_WIDTH = 84
AUDIO_LATENTS = 263  # round(158 / 24 * 40) per channel
AUDIO_CHANNELS = 2
ANCHOR_ROWS = ROWS_PER_LATENT
AUDIO_ROWS = AUDIO_LATENTS * AUDIO_CHANNELS
TARGET_ROWS = LATENT_FRAMES * ROWS_PER_LATENT

TEXT_ROWS = 64  # 40 text tokens followed by a 24-row vision run inside the text segment
KEYFRAME_TIMESTEP = 0.999


def _fixture() -> dict[str, np.ndarray]:
    with np.load(_FIXTURE_PATH) as data:
        return {name: data[name] for name in data.files}


def _tags() -> np.ndarray:
    # A vision run live inside the text segment: camera rows must follow
    # camera_video_indices, never token_tags.
    return np.array([TEXT_TAG] * 40 + [VIDEO_TAG] * 24, dtype=np.int64)


def _layout():
    return build_stage0p5_layout(
        _tags(),
        LATENT_FRAMES,
        LATENT_HEIGHT,
        LATENT_WIDTH,
        AUDIO_LATENTS,
    )


def test_document_shape_matches_checkpoint_contract() -> None:
    layout = _layout()
    assert int(layout.text_indices.size) == TEXT_ROWS
    assert int(layout.condition_video_indices.size) == ANCHOR_ROWS
    assert int(layout.audio_indices.size) == AUDIO_ROWS
    assert int(layout.noisy_video_indices.size) == TARGET_ROWS
    assert layout.rows_per_video_frame == ROWS_PER_LATENT
    assert layout.sequence_length == TEXT_ROWS + ANCHOR_ROWS + AUDIO_ROWS + TARGET_ROWS
    # Document order is [text | anchor | audio | target video].
    assert int(layout.text_indices[0]) == 0
    assert int(layout.condition_video_indices[0]) == TEXT_ROWS
    assert int(layout.audio_indices[0]) == TEXT_ROWS + ANCHOR_ROWS
    assert int(layout.noisy_video_indices[0]) == TEXT_ROWS + ANCHOR_ROWS + AUDIO_ROWS


def test_token_tags_are_segment_based() -> None:
    layout = _layout()
    # Text rows keep their incoming tags, including the vision run.
    assert np.array_equal(layout.token_tags[layout.text_indices], _tags())
    assert np.all(layout.token_tags[layout.video_indices] == VIDEO_TAG)
    assert np.all(layout.token_tags[layout.audio_indices] == AUDIO_TAG)
    # Every row belongs to exactly one segment.
    assert int(layout.text_indices.size + layout.video_indices.size + layout.audio_indices.size) == (
        layout.sequence_length
    )


def test_camera_indices_exclude_text_vision_run() -> None:
    layout = _layout()
    expected_camera_rows = np.concatenate(
        (layout.condition_video_indices, layout.noisy_video_indices)
    )
    assert np.array_equal(layout.camera_video_indices, expected_camera_rows)
    # camera_frame_ids covers the anchor and every target frame. The anchor
    # reuses frame slot 0, so frame id 0 appears twice.
    expected_frame_ids = np.concatenate(
        (
            np.zeros(ANCHOR_ROWS, dtype=np.int64),
            np.repeat(np.arange(LATENT_FRAMES), ROWS_PER_LATENT),
        )
    )
    assert expected_frame_ids.size == ANCHOR_ROWS + TARGET_ROWS
    assert np.array_equal(layout.camera_frame_ids, expected_frame_ids)
    counts = np.bincount(layout.camera_frame_ids, minlength=LATENT_FRAMES)
    assert counts.tolist() == [ANCHOR_ROWS + ROWS_PER_LATENT] + [ROWS_PER_LATENT] * (LATENT_FRAMES - 1)
    # The text vision run must NOT be a camera row.
    assert not np.intersect1d(layout.camera_video_indices, layout.text_indices).size


def test_position_ids_follow_h3_three_axis_grid() -> None:
    layout = _layout()
    positions = layout.position_ids
    frame_grid, width_grid = frame_position_grid(LATENT_HEIGHT, LATENT_WIDTH)

    # Text rows advance one temporal unit each.
    assert np.array_equal(positions[layout.text_indices, 0], np.arange(TEXT_ROWS, dtype=np.float64))
    # The anchor sits at the text length and uses the shared spatial grid.
    assert np.all(positions[layout.condition_video_indices, 0] == float(TEXT_ROWS))
    assert np.array_equal(positions[layout.condition_video_indices, 1:], frame_grid)

    # Target video time is the non-uniform cumulative rope grid, not a linspace.
    temporal = temporal_position_grid(LATENT_FRAMES, float(TEXT_ROWS))
    target_t = np.repeat(temporal, ROWS_PER_LATENT)
    assert np.array_equal(positions[layout.noisy_video_indices, 0], target_t)
    steps = np.diff(temporal)
    assert not np.allclose(steps, steps[0]), "target time must not be uniform"

    # Audio is channel-major and pinned to the w-grid extremes.
    audio_w = positions[layout.audio_indices, 2]
    assert np.array_equal(audio_w[:AUDIO_LATENTS], np.full(AUDIO_LATENTS, width_grid[0]))
    assert np.array_equal(audio_w[AUDIO_LATENTS:], np.full(AUDIO_LATENTS, width_grid[-1]))
    audio_t = positions[layout.audio_indices, 0]
    expected_audio_t = np.tile(np.arange(AUDIO_LATENTS) + TEXT_ROWS, AUDIO_CHANNELS)
    assert np.array_equal(audio_t, expected_audio_t)


def test_structural_arrays_match_reference_fixture() -> None:
    layout = _layout()
    fixture = _fixture()
    assert np.array_equal(layout.position_ids, fixture["position_ids"])
    assert np.array_equal(layout.token_tags, fixture["token_tags"])
    assert np.array_equal(layout.camera_video_indices, fixture["camera_video_indices"])
    assert np.array_equal(layout.camera_frame_ids, fixture["camera_frame_ids"])
    assert np.array_equal(layout.row_roles, fixture["row_roles"])
    assert np.array_equal(layout.target_video_chunk_ids, fixture["target_video_chunk_ids"])
    assert np.array_equal(layout.noisy_video_indices, fixture["noisy_video_indices"])
    assert np.array_equal(layout.audio_indices, fixture["audio_indices"])
    assert np.array_equal(layout.condition_video_indices, fixture["condition_video_indices"])
    assert np.array_equal(layout.text_indices, fixture["text_indices"])


def test_row_roles_and_chunk_ids() -> None:
    layout = _layout()
    # Roles: 0 condition, 1 audio, 3 noisy video (2 is clean video, absent here).
    assert np.all(layout.row_roles[layout.condition_indices] == 0)
    assert np.all(layout.row_roles[layout.audio_indices] == 1)
    assert np.all(layout.row_roles[layout.noisy_video_indices] == 3)
    # Stage0.5 is a full-attention document: no chunk grouping.
    assert np.all(layout.target_video_chunk_ids == -1)


def test_patchify_round_trip_uses_96_wide_rows() -> None:
    rng = np.random.default_rng(0)
    latents = rng.standard_normal((1, 24, LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH)).astype(np.float32)
    rows = patchify_video(latents)
    # 24 channels x (1x2x2 patch) = 96 wide, one row per 2x2 spatial patch.
    assert rows.shape == (1, TARGET_ROWS, 96)
    restored = unpatchify_video(rows, LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH)
    assert np.array_equal(restored, latents)


def test_row_timesteps_group_by_row_class() -> None:
    layout = _layout()
    video_t, audio_t = 0.31, 0.77
    times, index = build_row_timesteps(
        layout.to("cpu"),
        video_t,
        audio_t,
        condition_video_timestep=KEYFRAME_TIMESTEP,
        device="cpu",
    )
    distinct = np.sort(times.numpy())
    # Text inherits the video time, so only three classes are present. Stage0.5
    # has no clean-video rows, so clean_video_timestep=1.0 never appears.
    assert np.allclose(distinct, [video_t, audio_t, KEYFRAME_TIMESTEP])
    assert not np.isclose(distinct, 1.0).any()

    row_times = times.numpy()[index.numpy()]
    assert np.allclose(row_times[layout.text_indices], video_t)
    assert np.allclose(row_times[layout.condition_video_indices], KEYFRAME_TIMESTEP)
    assert np.allclose(row_times[layout.audio_indices], audio_t)
    assert np.allclose(row_times[layout.noisy_video_indices], video_t)

    fixture = _fixture()
    assert np.allclose(times.numpy(), fixture["row_times"])
    assert np.array_equal(index.numpy(), fixture["row_time_index"])


def test_row_timesteps_track_the_video_timestep() -> None:
    layout = _layout()
    times, index = build_row_timesteps(
        layout.to("cpu"), 0.05, 0.9, condition_video_timestep=KEYFRAME_TIMESTEP, device="cpu"
    )
    row_times = times.numpy()[index.numpy()]
    assert np.allclose(row_times[layout.noisy_video_indices], 0.05)
    assert np.allclose(row_times[layout.text_indices], 0.05)
    assert times.numel() == 3


def test_layout_rejects_malformed_input() -> None:
    # Two-dimensional tags are not a token sequence.
    try:
        build_stage0p5_layout(np.zeros((2, 4), dtype=np.int64), LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH)
    except ValueError:
        pass
    else:
        raise AssertionError("2-D text_token_tags must be rejected")

    # A latent grid that is not divisible by the [1, 2, 2] patch.
    try:
        build_stage0p5_layout(_tags(), LATENT_FRAMES, LATENT_HEIGHT + 1, LATENT_WIDTH)
    except ValueError:
        pass
    else:
        raise AssertionError("a misaligned latent grid must be rejected")


def test_materialized_layout_index_arrays_are_torch() -> None:
    # ``build_row_timesteps`` reads the index arrays through ``.to(device=...)``,
    # so ``.to()`` must convert the numpy structural arrays rather than keep them.
    layout = _layout().to("cpu")
    assert isinstance(layout.text_indices, torch.Tensor)
    assert isinstance(layout.position_ids, torch.Tensor)
    assert layout.position_ids.dtype == torch.float64

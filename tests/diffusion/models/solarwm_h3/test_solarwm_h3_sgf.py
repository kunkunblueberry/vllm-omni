# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Contract tests for the vendored SolarWM-H3 SGF rollout.

The rollout is exercised with a small *stub* student whose output is routed
through the real SGF attention path, so layout, camera PRoPE, segmentation and
raw-KV timing all reach the result without needing the 33B checkpoint. The stub
is not a stand-in for a test oracle: the values it produces were checked
bit-for-bit against SolarWM's own ``h3_sgf_rollout`` with the identical stub
(see ``docs/design/solarwm-h3/PLAN.md``), and the committed golden subsample
below comes from that reference run.

Plain ``assert`` functions, so they run with or without pytest.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.models.solarwm_h3 import sgf_attention, sgf_rollout
from vllm_omni.diffusion.models.solarwm_h3.layout import patchify_video

_FIXTURE_PATH = Path(__file__).with_name("fixtures") / "solarwm_h3_sgf_rollout.npz.fixture"

TEXT_ROWS = 64
AUDIO_LATENTS = 263
AUDIO_CHANNELS = 2
CHUNKS = 10
FRAMES_PER_CHUNK = 5
HEADS = 2
HEAD_DIM = 128
WIDTH = 96  # 24 channels x (1, 2, 2) patch
EXIT_INDEX = 3


# --------------------------------------------------------------------------
# stub student
# --------------------------------------------------------------------------


def _rope(positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = torch.arange(HEAD_DIM, dtype=torch.float64) * 0.01
    angles = positions[..., 0:1].to(torch.float64) * freqs
    return torch.cos(angles), torch.sin(angles)


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.to(x.dtype).unsqueeze(0).unsqueeze(2)
    sin = sin.to(x.dtype).unsqueeze(0).unsqueeze(2)
    rotated = torch.cat((-x[..., 1:], x[..., :1]), dim=-1)
    return x * cos + rotated * sin


def _attend(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """The SGF segmentation hands this ``[B, S, H, D]`` with unequal lengths."""

    out = F.scaled_dot_product_attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2))
    return out.transpose(1, 2)


class _StubStudent:
    """Minimal packed-document transformer that runs the real SGF attention.

    It mirrors what the H3 model does around attention: scatter the modality
    embeddings into the packed document, attend, then gather the video rows
    back. Every call also records the history it was allowed to see, which is
    what pins causality and the commit pattern.
    """

    def __init__(self, device: torch.device, attention_module=sgf_attention) -> None:
        generator = torch.Generator().manual_seed(20261010)
        self.qkv = torch.randn(WIDTH, 3 * HEADS * HEAD_DIM, generator=generator, device=device) * 0.02
        self.out = torch.randn(HEADS * HEAD_DIM, WIDTH, generator=generator, device=device) * 0.02
        self.audio_proj = torch.randn(32, WIDTH, generator=generator, device=device) * 0.02
        self.device = device
        self.attention = attention_module
        # (chunk_index, history chunk ids, commit_cache, grad_enabled)
        self.calls: list[tuple[int, tuple[int, ...], bool, bool]] = []

    def __call__(self, **kwargs):
        hidden_states = kwargs["hidden_states"].to(self.device)
        encoder_hidden_states = kwargs["encoder_hidden_states"].to(self.device)
        control = kwargs["sgf_attention"]
        layout = control.layout

        history: tuple[int, ...] = ()
        if control.cache is not None:
            history = tuple(entry[0] for entry in control.cache.layers.get(0, ()))
        self.calls.append((control.chunk_index, history, bool(control.commit_cache), torch.is_grad_enabled()))

        batch = hidden_states.shape[0]
        packed = torch.zeros(batch, layout.sequence_length, WIDTH, device=self.device)
        packed[:, : encoder_hidden_states.shape[1]] = encoder_hidden_states[:, :, :WIDTH]
        packed[:, layout.audio_indices.to(self.device)] = (
            kwargs["audio_hidden_states"].to(self.device) @ self.audio_proj
        )
        packed[:, layout.video_indices.to(self.device)] = hidden_states

        query, key, value = (packed @ self.qkv).split(HEADS * HEAD_DIM, dim=-1)
        shape = (batch, packed.shape[1], HEADS, HEAD_DIM)
        query, key, value = (tensor.reshape(shape) for tensor in (query, key, value))

        prepared = self.attention.prepare_sgf_rotaries(control, _rope)
        out = self.attention.sgf_attention(
            query, key, value, control=prepared, layer=0, apply_rotary=_apply_rotary, attend=_attend
        )
        out = out.reshape(out.shape[0], out.shape[1], HEADS * HEAD_DIM)
        return out[:, layout.video_indices.to(self.device)] @ self.out, None


# --------------------------------------------------------------------------
# inputs and rollout
# --------------------------------------------------------------------------


def _inputs(device: torch.device, rollout_module=sgf_rollout, patchify=patchify_video):
    generator = torch.Generator().manual_seed(7)
    tags = torch.tensor([1] * 40 + [0] * 24, dtype=torch.long, device=device)
    prompt = torch.randn(1, TEXT_ROWS, WIDTH, generator=generator).to(device)
    anchor = torch.randn(1, 24, 1, 48, 84, generator=generator).to(device)
    audio = torch.randn(1, AUDIO_LATENTS * AUDIO_CHANNELS, 32, generator=generator).to(device)

    # A camera that translates along z at a constant rate, with a fixed focal
    # length: the anchor slot plus all fifty rollout slots.
    viewmats = torch.eye(4, device=device).repeat(1, CHUNKS * FRAMES_PER_CHUNK + 1, 1, 1).clone()
    viewmats[0, :, 2, 3] = torch.arange(viewmats.shape[1], dtype=torch.float32, device=device) * 0.05
    intrinsics = torch.zeros(1, viewmats.shape[1], 3, 3, device=device)
    intrinsics[0, :, 0, 0] = 0.5051
    intrinsics[0, :, 1, 1] = 0.8979
    intrinsics[0, :, 2, 2] = 1.0

    return rollout_module.H3SGFInputs(
        prompt=prompt,
        text_tags=tags,
        anchor_rows=patchify(anchor),
        audio_rows=audio,
        audio_timestep=torch.tensor(0.77, device=device),
        camera_viewmats=viewmats,
        camera_K=intrinsics,
        latent_height=48,
        latent_width=84,
        keyframe_timestep=0.999,
        sp_enabled=False,
    )


def _run_rollout(
    device: torch.device,
    rollout_module=sgf_rollout,
    attention_module=sgf_attention,
    patchify=patchify_video,
):
    generator = torch.Generator().manual_seed(3)
    noise = torch.randn(1, 24, CHUNKS * FRAMES_PER_CHUNK, 48, 84, generator=generator, device=device)
    student = _StubStudent(device, attention_module)
    result = rollout_module.h3_sgf_rollout(
        student=student,
        inputs=_inputs(device, rollout_module, patchify),
        noise=noise,
        exit_index=EXIT_INDEX,
        generator=torch.Generator(device=device).manual_seed(99),
        video_shift=12.0,
        last_step_only=True,
    )
    return result, student


def _fixture() -> dict[str, np.ndarray]:
    with np.load(_FIXTURE_PATH) as data:
        return {name: data[name] for name in data.files}


# --------------------------------------------------------------------------
# window and schedule contract (cheap, no model)
# --------------------------------------------------------------------------


def test_window_definition_matches_the_published_contract() -> None:
    windows = sgf_rollout.h3_sgf_windows()
    assert len(windows) == CHUNKS
    for index, window in enumerate(windows):
        assert window.chunk_index == index
        assert window.current_start == index * FRAMES_PER_CHUNK
        assert window.stop == (index + 1) * FRAMES_PER_CHUNK
        # The window reaches back at most five chunks, and the supervision
        # stops at the 47th latent.
        assert window.start == max(0, index - 5) * FRAMES_PER_CHUNK

    # The first six chunks each add one chunk of context, and the window then
    # stops growing: rotations are only precomputed for 5..30 frames.
    local = [window.local_latents for window in windows]
    assert local == [5, 10, 15, 20, 25, 30, 30, 30, 30, 30]
    assert windows[-1].supervised_latents == 2


def test_sgf_schedule_is_four_evaluations() -> None:
    schedule, audio = sgf_rollout.h3_sgf_schedule(video_shift=12.0, audio_shift=3.0)
    assert schedule.timesteps.numel() == 4
    assert audio.timesteps.numel() == 4
    assert float(schedule.sigmas[-1]) == 0.0


def test_dynamic_window_construction_for_longer_rollouts() -> None:
    # Two chunks of five latents: the second chunk sees the first one.
    windows = tuple(
        sgf_rollout.H3SGFWindow(i, max(0, i - 5) * FRAMES_PER_CHUNK, i * FRAMES_PER_CHUNK,
                                (i + 1) * FRAMES_PER_CHUNK, (i + 1) * FRAMES_PER_CHUNK)
        for i in range(2)
    )
    assert windows[0].local_latents == 5
    assert windows[1].local_latents == 10
    assert windows[1].prior_chunks == (0,)


# --------------------------------------------------------------------------
# raw KV cache contract (cheap, no model)
# --------------------------------------------------------------------------


def _kv(rows: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(5)
    key = torch.randn(1, rows, HEADS, HEAD_DIM, generator=generator)
    return key, key.clone()


def test_raw_kv_cache_retains_only_the_window() -> None:
    cache = sgf_attention.H3RawKVCache()
    # Commit the first seven chunks. While generating a chunk, ``history`` is
    # asked for that chunk's index, so chunk 7 must see 2..6.
    for chunk_index in range(7):
        key, value = _kv()
        with torch.no_grad():
            cache.commit(0, chunk_index, key, value)
    assert [entry[0] for entry in cache.history(0, 7)] == [2, 3, 4, 5, 6]
    assert len(cache.layers[0]) == 5, "the cache must not grow past its window"

    # Committing chunk 7 slides the window and evicts chunk 2 outright; the
    # window never grows past five chunks.
    key, value = _kv()
    with torch.no_grad():
        cache.commit(0, 7, key, value)
    assert [entry[0] for entry in cache.layers[0]] == [3, 4, 5, 6, 7]
    assert len(cache.layers[0]) == 5
    assert [entry[0] for entry in cache.history(0, 8)] == [3, 4, 5, 6, 7]


def test_raw_kv_cache_rejects_a_history_gap() -> None:
    cache = sgf_attention.H3RawKVCache()
    key, value = _kv()
    with torch.no_grad():
        cache.commit(0, 0, key, value)
    try:
        cache.history(0, 4)
    except RuntimeError as exc:
        assert "history differs" in str(exc)
    else:
        raise AssertionError("a non-contiguous history must be rejected")


def test_raw_kv_commit_requires_no_grad() -> None:
    cache = sgf_attention.H3RawKVCache()
    key, value = _kv()
    try:
        cache.commit(0, 0, key, value)
    except RuntimeError as exc:
        assert "no-grad" in str(exc)
    else:
        raise AssertionError("committing outside no_grad must be rejected")


def test_raw_kv_history_is_empty_for_the_first_chunk() -> None:
    cache = sgf_attention.H3RawKVCache()
    assert cache.history(0, 0) == []


# --------------------------------------------------------------------------
# full rollout (slow: one 10-chunk rollout on CPU)
# --------------------------------------------------------------------------


def test_rollout_contract_call_pattern_and_golden_values() -> None:
    result, student = _run_rollout(torch.device("cpu"))

    # --- output contract -------------------------------------------------
    assert tuple(result.cache_target.shape) == (1, 24, 50, 48, 84)
    assert result.exit_index == EXIT_INDEX
    assert torch.isfinite(result.cache_target).all()
    assert float(result.cache_target.abs().max()) > 0
    supervised = result.cache_target[:, :, :47]
    assert tuple(supervised.shape) == (1, 24, 47, 48, 84)

    # --- four probes plus one commit per chunk ---------------------------
    assert len(student.calls) == CHUNKS * (EXIT_INDEX + 2)
    for chunk_index in range(CHUNKS):
        calls = student.calls[chunk_index * 5 : (chunk_index + 1) * 5]
        assert [call[0] for call in calls] == [chunk_index] * 5
        assert [call[2] for call in calls] == [False, False, False, False, True], (
            "only the fifth forward of a chunk may commit K/V"
        )
        assert [call[3] for call in calls] == [False, False, False, False, False], (
            "the whole rollout runs under no_grad"
        )

    # --- causality: no forward ever sees its own or a later chunk --------
    for chunk_index, history, _, _ in student.calls:
        assert all(seen < chunk_index for seen in history), (chunk_index, history)

    # --- the window is the previous five chunks --------------------------
    for chunk_index in range(6, CHUNKS):
        histories = [call[1] for call in student.calls if call[0] == chunk_index]
        expected = tuple(range(chunk_index - 5, chunk_index))
        assert histories[0] == expected, (chunk_index, histories[0], expected)

    # --- golden values from the reference implementation -----------------
    # The port reproduces SolarWM's rollout bit-for-bit, so these are exact
    # comparisons with no tolerance to hide drift behind.
    fixture = _fixture()
    subsample = result.cache_target[0][(slice(0, 2), slice(None, None, 10), slice(None, None, 24), slice(None, None, 42))]
    assert np.array_equal(subsample.numpy(), fixture["cache_target_subsample"]), (
        "rollout output drifted from the reference-derived golden values: max diff "
        f"{np.abs(subsample.numpy() - fixture['cache_target_subsample']).max()}"
    )
    assert np.array_equal(result.native_exit_timestep.numpy(), fixture["native_exit_timestep"])


# --------------------------------------------------------------------------
# port boundaries
# --------------------------------------------------------------------------


def test_sequence_parallel_is_rejected() -> None:
    try:
        sgf_rollout.broadcast_sequence_parallel_tensor(torch.zeros(1))
    except NotImplementedError as exc:
        assert "sequence parallelism" in str(exc)
    else:
        raise AssertionError("the SP shim must fail loudly")

    inputs = _inputs(torch.device("cpu"))
    assert inputs.sp_enabled is False


def test_vendored_modules_declare_their_provenance() -> None:
    package = Path(sgf_attention.__file__).parent
    vendored = [
        "geometry.py",
        "layout.py",
        "flow.py",
        "camera.py",
        "torch_geometry.py",
        "prope.py",
        "sgf.py",
        "sgf_attention.py",
        "sgf_rollout.py",
    ]
    for name in vendored:
        text = (package / name).read_text(encoding="utf-8")
        assert "SPDX-License-Identifier" in text, name
        assert "Vendored from SolarWM" in text, name
        assert "ce1da4e7705391eda8eeda6016c0fd3f614b975e" in text, name

    # torch_prope.py carries PRoPE's MIT notice as well as SolarWM's Apache one.
    prope_text = (package / "prope.py").read_text(encoding="utf-8")
    assert "MIT, Copyright (c) the PRoPE authors" in prope_text

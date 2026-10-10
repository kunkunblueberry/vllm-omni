# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Contract tests for SolarWM-H3's rectified-flow arithmetic.

The expectations here are the mathematical definitions SolarWM's SGF student is
trained against, not values read back from the implementation under test:

* forward flow is ``x_t = t*x0 + (1-t)*noise`` with ``t=1`` **clean**;
* the velocity target is data-ward, ``v = x0 - noise`` (the opposite sign from
  Wan) and inverts as ``x0 = x_t + (1-t)*v``;
* the sigma shift is ``s*sigma / (1 + (s-1)*sigma)``;
* SGF runs four denoiser evaluations, which is five *grid points* in H3's
  upstream convention, so four timesteps come back for ``shift`` 12 and 3.

Plain ``assert`` functions, so they run with or without pytest.
"""

from __future__ import annotations

import torch

from vllm_omni.diffusion.models.solarwm_h3.flow import (
    data_velocity_target,
    euler_step,
    make_shifted_schedule,
    predict_clean_sample,
    scale_noise,
    shift_sigma,
)

VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0
SGF_EVALUATIONS = 4


def _samples() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    shape = (1, 24, 5, 4, 6)
    clean = torch.randn(shape, dtype=torch.float32, generator=generator)
    noise = torch.randn(shape, dtype=torch.float32, generator=generator)
    return clean, noise


def test_t_equals_one_is_clean_and_t_zero_is_noise() -> None:
    clean, noise = _samples()
    assert torch.equal(scale_noise(clean, noise, 1.0), clean)
    assert torch.equal(scale_noise(clean, noise, 0.0), noise)


def test_velocity_target_is_data_ward() -> None:
    clean, noise = _samples()
    velocity = data_velocity_target(clean, noise)
    # H3 predicts x0 - noise, the opposite sign from Wan's noise - x0.
    assert torch.equal(velocity, clean - noise)
    assert not torch.equal(velocity, noise - clean)


def test_noising_inverts_to_the_clean_sample() -> None:
    clean, noise = _samples()
    velocity = data_velocity_target(clean, noise)
    for timestep in (0.0, 0.25, 0.5, 0.9, 1.0):
        noisy = scale_noise(clean, noise, timestep)
        recovered = predict_clean_sample(noisy, velocity, timestep)
        assert torch.allclose(recovered, clean, atol=1e-5), timestep


def test_shift_sigma_endpoints_and_monotonicity() -> None:
    sigmas = torch.linspace(0.0, 1.0, 11, dtype=torch.float32)
    for shift in (AUDIO_SHIFT, VIDEO_SHIFT):
        shifted = shift_sigma(sigmas, shift)
        expected = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
        assert torch.allclose(shifted, expected, atol=1e-6)
        assert float(shifted[0]) == 0.0
        assert abs(float(shifted[-1]) - 1.0) < 1e-6
        assert torch.all(torch.diff(shifted) > 0), "shift must be monotone"


def test_schedule_gives_four_evaluations_from_five_grid_points() -> None:
    for shift in (VIDEO_SHIFT, AUDIO_SHIFT):
        schedule = make_shifted_schedule(SGF_EVALUATIONS + 1, shift=shift)
        assert schedule.sigmas.numel() == SGF_EVALUATIONS + 1
        assert schedule.timesteps.numel() == SGF_EVALUATIONS
        assert torch.all(torch.diff(schedule.sigmas) < 0), "sigmas must descend"
        assert float(schedule.sigmas[-1]) == 0.0, "terminal sigma must be zero"
        # The model timestep is the complement of the sigma at that grid point.
        assert torch.allclose(schedule.timesteps, 1.0 - schedule.sigmas[:-1], atol=1e-6)


def test_shift_moves_mass_toward_high_sigma() -> None:
    # A larger shift must place the same base sigma higher on the shifted grid,
    # which is what makes the video schedule (12) noisier than audio's (3).
    base = torch.tensor([0.5], dtype=torch.float32)
    assert float(shift_sigma(base, VIDEO_SHIFT)) > float(shift_sigma(base, AUDIO_SHIFT))


def test_euler_step_blends_toward_the_denoised_estimate() -> None:
    clean, noise = _samples()
    velocity = data_velocity_target(clean, noise)
    timestep, sigma, sigma_next = 0.6, 0.4, 0.1
    sample = scale_noise(clean, noise, timestep)

    stepped = euler_step(sample, velocity, timestep, sigma, sigma_next)
    denoised = predict_clean_sample(sample, velocity, timestep)
    ratio = sigma_next / sigma
    assert torch.allclose(stepped, ratio * sample + (1.0 - ratio) * denoised, atol=1e-6)

    # At the terminal sigma the step returns the clean estimate outright.
    terminal = euler_step(sample, velocity, timestep, sigma, 0.0)
    assert torch.allclose(terminal, denoised, atol=1e-6)
    assert torch.allclose(terminal, clean, atol=1e-5)


def test_half_precision_inputs_keep_their_dtype() -> None:
    clean, noise = _samples()
    sample = scale_noise(clean, noise, 0.6).to(torch.bfloat16)
    velocity = data_velocity_target(clean, noise).to(torch.bfloat16)
    stepped = euler_step(sample, velocity, 0.6, 0.4, 0.2)

    assert stepped.dtype == torch.bfloat16
    assert torch.isfinite(stepped).all()

    # ``euler_step`` recovers x0 through ``predict_clean_sample`` in the input
    # dtype and only upcasts the final blend, so a bf16 call is not the fp32
    # result rounded once. Bound the disagreement by bf16's unit roundoff
    # (2**-8) with one bit of margin, scaled to the value range.
    reference = euler_step(
        sample.to(torch.float32), velocity.to(torch.float32), 0.6, 0.4, 0.2
    )
    assert reference.dtype == torch.float32
    scale = float(reference.abs().max())
    assert torch.allclose(stepped.to(torch.float32), reference, atol=(2**-7) * scale)


def test_predict_clean_sample_keeps_the_input_dtype() -> None:
    # Documents the property above: the x0 recovery is not upcast internally,
    # which is why a half-precision step carries half-precision error.
    clean, noise = _samples()
    noisy = scale_noise(clean, noise, 0.6).to(torch.bfloat16)
    recovered = predict_clean_sample(noisy, (clean - noise).to(torch.bfloat16), 0.6)
    assert recovered.dtype == torch.bfloat16

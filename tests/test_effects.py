"""Time-varying low-pass sweep tests."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_tone
from dcp.models import AudioBuffer, PlanValidationError
from dcp.rendering.effects import apply_lowpass_sweep
from dcp.transitions import EffectSpec


def _band_energy(samples: np.ndarray, sample_rate: int, low: float, high: float) -> float:
    spectrum = np.abs(np.fft.rfft(samples))
    frequencies = np.fft.rfftfreq(samples.shape[0], 1.0 / sample_rate)
    mask = (frequencies >= low) & (frequencies <= high)
    return float(np.sum(spectrum[mask] ** 2))


def _two_tone(duration_s: float = 2.0, sample_rate: int = 22050) -> AudioBuffer:
    frames = int(duration_s * sample_rate)
    time = np.arange(frames) / sample_rate
    mono = 0.35 * np.sin(2 * np.pi * 440.0 * time) + 0.35 * np.sin(
        2 * np.pi * 8000.0 * time
    )
    stereo = np.repeat(mono[np.newaxis, :], 2, axis=0)
    return AudioBuffer(np.ascontiguousarray(stereo), sample_rate)


def test_lowpass_sweep_attenuates_high_frequencies():
    dry = _two_tone()
    effect = EffectSpec(
        kind="lowpass_sweep",
        start_hz=10000.0,
        end_hz=500.0,
        start_s=0.0,
        end_s=2.0,
    )
    wet = apply_lowpass_sweep(
        dry, effect, transition_start_s=0.0, transition_duration_s=2.0
    )
    before_high = _band_energy(dry.mono(), dry.sample_rate, 6000.0, 10000.0)
    after_high = _band_energy(wet.mono(), wet.sample_rate, 6000.0, 10000.0)
    assert after_high < 0.25 * before_high

    before_low = _band_energy(dry.mono(), dry.sample_rate, 300.0, 600.0)
    after_low = _band_energy(wet.mono(), wet.sample_rate, 300.0, 600.0)
    assert after_low > 0.2 * before_low
    assert np.all(np.isfinite(wet.samples))
    assert wet.sample_rate == dry.sample_rate


def test_lowpass_sweep_does_not_mutate_input():
    dry = _two_tone(duration_s=1.0)
    original = dry.samples.copy()
    effect = EffectSpec(start_hz=10000.0, end_hz=800.0, start_s=0.0, end_s=1.0)
    apply_lowpass_sweep(
        dry, effect, transition_start_s=0.0, transition_duration_s=1.0
    )
    assert np.array_equal(dry.samples, original)


def test_lowpass_sweep_leaves_audio_before_effect_start_untouched():
    dry = _two_tone(duration_s=2.0)
    effect = EffectSpec(
        start_hz=10000.0, end_hz=800.0, start_s=0.5, end_s=1.5
    )
    wet = apply_lowpass_sweep(
        dry, effect, transition_start_s=0.0, transition_duration_s=2.0
    )
    boundary = int(0.5 * dry.sample_rate)
    assert np.array_equal(wet.samples[:, :boundary], dry.samples[:, :boundary])
    assert not np.array_equal(wet.samples[:, boundary:], dry.samples[:, boundary:])


def test_lowpass_sweep_rejects_unsupported_effect_kind():
    with pytest.raises(PlanValidationError, match="unsupported effect"):
        apply_lowpass_sweep(
            _two_tone(duration_s=0.5),
            EffectSpec(kind="reverb"),
            transition_start_s=0.0,
            transition_duration_s=0.5,
        )


def test_lowpass_sweep_rejects_cutoff_above_nyquist():
    with pytest.raises(PlanValidationError, match="Nyquist"):
        apply_lowpass_sweep(
            _two_tone(duration_s=0.5),
            EffectSpec(start_hz=10000.0, end_hz=12000.0),
            transition_start_s=0.0,
            transition_duration_s=0.5,
        )


def test_lowpass_sweep_rejects_invalid_cutoffs():
    with pytest.raises(PlanValidationError, match="greater than 20"):
        apply_lowpass_sweep(
            _two_tone(duration_s=0.5),
            EffectSpec(start_hz=10.0, end_hz=500.0),
            transition_start_s=0.0,
            transition_duration_s=0.5,
        )


def test_lowpass_sweep_outside_audio_returns_copy():
    dry = _two_tone(duration_s=1.0)
    effect = EffectSpec(
        start_hz=10000.0, end_hz=800.0, start_s=0.0, end_s=1.0
    )
    wet = apply_lowpass_sweep(
        dry,
        effect,
        transition_start_s=5.0,
        transition_duration_s=1.0,
    )
    assert np.array_equal(wet.samples, dry.samples)

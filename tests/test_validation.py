"""Objective validation metrics and failure detection."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_constant, make_silence, make_tone
from dcp.models import AudioBuffer
from dcp.validation import (
    compare_audio,
    longest_interior_silence_s,
    measure_audio,
    validate_audio,
)


def _check(report, name):
    return next(check for check in report.checks if check.name == name)


def test_valid_tone_passes_all_checks():
    tone = make_tone(duration_s=2.0, amplitude=0.5)
    report = validate_audio(
        tone,
        expected_duration_s=2.0,
        expected_sample_rate=22050,
        expected_channels=2,
    )
    assert report.passed is True
    assert all(check.passed for check in report.checks)
    assert report.metrics["duration_s"] == pytest.approx(2.0)
    assert "Validation: PASS" in report.format_text()


def test_duration_mismatch_fails():
    report = validate_audio(
        make_tone(duration_s=2.0), expected_duration_s=3.0
    )
    assert report.passed is False
    assert _check(report, "duration").passed is False


def test_sample_rate_and_channel_mismatch_fail():
    report = validate_audio(
        make_tone(duration_s=1.0, channels=1),
        expected_sample_rate=44100,
        expected_channels=2,
    )
    assert _check(report, "sample_rate").passed is False
    assert _check(report, "channels").passed is False


def test_non_finite_samples_fail_finite_check():
    samples = make_tone(duration_s=1.0).samples.copy()
    samples[0, 100] = np.nan
    report = validate_audio(AudioBuffer(samples, 22050))
    assert report.passed is False
    assert _check(report, "finite").passed is False


def test_clipping_is_detected():
    report = validate_audio(make_constant(1.0, duration_s=1.0))
    assert report.passed is False
    assert _check(report, "clipping").passed is False
    assert report.metrics["near_full_scale_ratio"] == pytest.approx(1.0)


def test_near_full_scale_below_threshold_is_allowed():
    report = validate_audio(make_constant(0.9998, duration_s=1.0))
    assert _check(report, "clipping").passed is True


def test_interior_silence_is_measured_and_limited():
    tone = make_tone(duration_s=1.0, amplitude=0.4)
    gap = np.zeros((2, 22050), dtype=np.float64)
    buffer = AudioBuffer(
        np.concatenate([tone.samples, gap, tone.samples], axis=1), 22050
    )
    assert longest_interior_silence_s(buffer) == pytest.approx(1.0, abs=0.06)

    failing = validate_audio(buffer, max_silence_s=0.5)
    assert _check(failing, "silence_gaps").passed is False

    passing = validate_audio(buffer, max_silence_s=1.5)
    assert _check(passing, "silence_gaps").passed is True


def test_edge_silence_is_not_an_interior_gap():
    tone = make_tone(duration_s=1.0, amplitude=0.4)
    zeros = np.zeros((2, 11025), dtype=np.float64)
    buffer = AudioBuffer(
        np.concatenate([zeros, tone.samples, zeros], axis=1), 22050
    )
    assert longest_interior_silence_s(buffer) == 0.0
    report = validate_audio(buffer, max_silence_s=0.0)
    assert _check(report, "silence_gaps").passed is True


def test_all_silent_file_reports_full_duration():
    silence = make_silence(duration_s=2.0)
    assert longest_interior_silence_s(silence) == pytest.approx(2.0)


def test_discontinuity_heuristic_detects_alternating_samples():
    samples = np.empty((1, 1000), dtype=np.float64)
    samples[0, ::2] = 1.0
    samples[0, 1::2] = -1.0
    report = validate_audio(AudioBuffer(samples, 22050))
    assert _check(report, "discontinuity").passed is False
    assert report.metrics["max_sample_step"] == pytest.approx(2.0)


def test_measure_audio_reports_expected_metrics():
    metrics = measure_audio(make_constant(0.5, duration_s=1.0))
    assert metrics["duration_s"] == pytest.approx(1.0)
    assert metrics["peak_dbfs"] == pytest.approx(-6.0206, abs=1e-3)
    assert metrics["rms_dbfs"] == pytest.approx(-6.0206, abs=1e-3)
    assert metrics["crest_factor_db"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["dc_offset_max"] == pytest.approx(0.5)
    assert metrics["max_sample_step"] == pytest.approx(0.0)
    assert metrics["longest_silence_s"] == 0.0


def test_compare_reports_deltas_and_disclaimer():
    baseline = make_tone(duration_s=1.0, amplitude=0.3)
    candidate = make_tone(duration_s=1.0, amplitude=0.15)
    comparison = compare_audio(baseline, candidate)
    assert "baseline_peak_dbfs" in comparison
    assert "dcp_peak_dbfs" in comparison
    assert comparison["delta_peak_dbfs"] < 0
    assert "not establish musical quality" in str(comparison["note"])


def test_duration_tolerance_accepts_small_rounding_difference():
    tone = make_tone(duration_s=2.0)
    report = validate_audio(tone, expected_duration_s=2.04)
    assert _check(report, "duration").passed is True
    report = validate_audio(tone, expected_duration_s=2.1)
    assert _check(report, "duration").passed is False

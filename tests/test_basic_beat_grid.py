"""Acceptance tests for the dependency-free basic analyzer beat/downbeat grids and LUFS proxy."""

from __future__ import annotations

import numpy as np
import pytest

import dcp.analysis.basic as basic_mod
from dcp.analysis import analyze_track_with_backend
from dcp.analysis.basic import analyze_track
from dcp.models import AudioBuffer


def _make_click_track(
    *,
    bpm: float,
    duration_s: float = 20.0,
    sr: int = 44100,
    pickup_s: float = 0.0,
    downbeat_boost_db: float = 10.0,
    backbeat_boost_db: float = 0.0,
) -> tuple[AudioBuffer, list[float]]:
    """Synthesize a deterministic 4/4 click track with accented downbeats (low + high burst)."""
    n_samples = int(round(duration_s * sr))
    samples = np.zeros(n_samples, dtype=np.float32)
    beat_period_s = 60.0 / bpm
    click_len = int(round(0.012 * sr))
    t_click = np.arange(click_len, dtype=np.float64) / sr
    env = np.exp(-t_click / 0.0025)

    downbeat_gain = 10.0 ** (downbeat_boost_db / 20.0)
    backbeat_gain = 10.0 ** (backbeat_boost_db / 20.0)
    base_amp = 0.25

    true_downbeats: list[float] = []
    beat_idx = 0
    t = pickup_s
    while t < duration_s - 0.02:
        start = int(round(t * sr))
        end = min(n_samples, start + click_len)
        L = end - start
        if L > 0:
            bar_pos = beat_idx % 4
            if bar_pos == 0:
                true_downbeats.append(t)
                # Downbeat has strong low-frequency kick thump + click (+10 dB) for estimate_downbeats
                burst = base_amp * downbeat_gain * (
                    0.7 * np.sin(2.0 * np.pi * 80.0 * t_click[:L])
                    + 0.3 * np.sin(2.0 * np.pi * 1600.0 * t_click[:L])
                ) * env[:L]
            elif bar_pos in (1, 3) and backbeat_boost_db != 0.0:
                burst = base_amp * backbeat_gain * np.sin(2.0 * np.pi * 1200.0 * t_click[:L]) * env[:L]
            else:
                burst = base_amp * np.sin(2.0 * np.pi * 1000.0 * t_click[:L]) * env[:L]
            samples[start:end] += burst.astype(np.float32)
        beat_idx += 1
        t = pickup_s + beat_idx * beat_period_s

    peak = float(np.max(np.abs(samples)))
    if peak > 0.98:
        samples *= 0.95 / peak

    return AudioBuffer(samples=samples[np.newaxis, :], sample_rate=sr), true_downbeats


def test_click_track_beat_grid_placed_correctly() -> None:
    buf, true_downbeats = _make_click_track(bpm=120.0, duration_s=20.0, sr=44100, pickup_s=0.0)
    result = analyze_track(buf)

    assert result.tempo_bpm is not None
    assert 85.0 <= result.tempo_bpm <= 125.0
    assert result.tempo_confidence >= 0.60
    assert len(result.beat_times_s) >= 38
    assert len(result.downbeat_times_s) > 0

    for db in result.downbeat_times_s:
        min_err = min(abs(db - t_true) for t_true in true_downbeats)
        assert min_err <= 0.035, f"Downbeat {db:.4f}s off by {min_err*1000:.1f}ms"


def test_pickup_phase_locks_not_to_zero() -> None:
    buf, true_downbeats = _make_click_track(bpm=120.0, duration_s=20.0, sr=44100, pickup_s=0.25)
    result = analyze_track(buf)

    assert len(result.downbeat_times_s) > 0
    for db in result.downbeat_times_s:
        min_err = min(abs(db - t_true) for t_true in true_downbeats)
        assert min_err <= 0.035, f"Pickup downbeat {db:.4f}s failed phase lock (err={min_err*1000:.1f}ms)"
        assert abs(db % 2.0) > 0.15, "Grid falsely locked to t=0.0 instead of 0.25s pickup"


def test_octave_resolution_no_double_time() -> None:
    buf, true_downbeats = _make_click_track(
        bpm=93.0,
        duration_s=22.0,
        sr=44100,
        downbeat_boost_db=10.0,
        backbeat_boost_db=4.0,
    )
    result = analyze_track(buf)

    assert result.tempo_bpm is not None
    assert 83.0 <= result.tempo_bpm <= 103.0, f"Octave error: got {result.tempo_bpm:.2f} BPM"
    assert len(result.downbeat_times_s) >= 4

    intervals = np.diff(result.downbeat_times_s)
    expected_bar_s = 4.0 * (60.0 / 93.0)  # ~2.5806 s
    for iv in intervals:
        assert abs(iv - expected_bar_s) <= 0.040, f"Bar interval {iv:.4f}s != {expected_bar_s:.4f}s"


def test_noise_and_silence_degrade_honestly() -> None:
    sr = 44100
    rng = np.random.default_rng(42)
    noise = 0.05 * rng.standard_normal(int(10.0 * sr)).astype(np.float32)
    silence = np.zeros(int(3.0 * sr), dtype=np.float32)
    combined = np.concatenate([noise, silence])

    res_noise = analyze_track(AudioBuffer(samples=combined[np.newaxis, :], sample_rate=sr))
    assert res_noise.tempo_confidence < 0.30
    assert res_noise.beat_times_s == ()
    assert res_noise.downbeat_times_s == ()

    res_silence = analyze_track(AudioBuffer(samples=silence[np.newaxis, :], sample_rate=sr))
    assert res_silence.tempo_confidence < 0.30
    assert res_silence.beat_times_s == ()
    assert res_silence.downbeat_times_s == ()


def test_lufs_proxy_bounded_and_ordered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(basic_mod, "_PYLOUDNORM_AVAILABLE", False)
    sr = 44100
    t = np.arange(int(5.0 * sr), dtype=np.float64) / sr
    carrier = np.sqrt(2.0) * np.sin(2.0 * np.pi * 440.0 * t)  # unit RMS sine

    sig_minus_20 = (carrier * (10.0 ** (-20.0 / 20.0))).astype(np.float32)
    sig_minus_6 = (carrier * (10.0 ** (-6.0 / 20.0))).astype(np.float32)

    res_20 = analyze_track(AudioBuffer(samples=sig_minus_20[np.newaxis, :], sample_rate=sr))
    res_6 = analyze_track(AudioBuffer(samples=sig_minus_6[np.newaxis, :], sample_rate=sr))

    assert res_20.loudness_lufs is not None
    assert res_6.loudness_lufs is not None
    assert res_20.loudness_lra is not None
    assert res_6.loudness_lra is not None
    assert -70.0 <= res_20.loudness_lufs <= 0.0
    assert -70.0 <= res_6.loudness_lufs <= 0.0
    delta_lu = res_6.loudness_lufs - res_20.loudness_lufs
    assert abs(delta_lu - 14.0) <= 2.5


def test_trackanalysis_grids_populated_via_backend() -> None:
    buf, _ = _make_click_track(bpm=120.0, duration_s=16.0, sr=44100)
    result = analyze_track_with_backend(buf, backend="basic")

    assert result.analyzer == "basic"
    assert len(result.beat_times_s) >= 28
    assert len(result.downbeat_times_s) >= 6


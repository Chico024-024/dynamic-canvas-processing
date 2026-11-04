"""Comprehensive verification of Step 3 DSP Engine & Bijective Warping."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_analysis, make_tone
from dcp.models import AudioBuffer
from dcp.rendering.dsp_fallback import (
    process_fdn_freeze_numpy,
    process_sidechain_compressor_numpy,
    process_zdf_svf_sweep_numpy,
    time_stretch_phase_vocoder_numpy,
)
from dcp.transitions.bijective_warp import (
    compute_safeguarded_bijective_warp,
    robust_quantile_normalize,
)


def test_bijective_warp_boundary_precision():
    """Verify analytical boundary error at phi=0 and phi=N_b is strictly < 1e-12 sec."""
    sr = 48000
    # Song A @ 120 BPM (0.5s = 24000 samples per beat)
    # Song B @ 128 BPM (0.46875s = 22500 samples per beat)
    beat_samples_a = np.array([0, 24000, 48000, 72000, 96000], dtype=np.float64)
    beat_samples_b = np.array([0, 22500, 45000, 67500, 90000], dtype=np.float64)

    warp = compute_safeguarded_bijective_warp(
        beat_samples_a,
        beat_samples_b,
        sr=sr,
        control_hop=256,
        tol_sec=1e-12,
    )

    # 1. Exact boundary phase
    assert warp.beat_phase_phi[0] == 0.0
    assert warp.beat_phase_phi[-1] == 4.0

    # 2. Exact source sample coincidence
    assert abs(warp.src_sample_a[0] - beat_samples_a[0]) < 1e-12
    assert abs(warp.src_sample_a[-1] - beat_samples_a[-1]) < 1e-12
    assert abs(warp.src_sample_b[0] - beat_samples_b[0]) < 1e-12
    assert abs(warp.src_sample_b[-1] - beat_samples_b[-1]) < 1e-12

    # 3. Monotonic phase advance
    assert np.all(np.diff(warp.beat_phase_phi) >= 0.0)

    # 4. Bounded positive stretch ratios
    assert np.all(warp.stretch_ratio_a > 0.8)
    assert np.all(warp.stretch_ratio_a < 1.2)
    assert np.all(warp.stretch_ratio_b > 0.8)
    assert np.all(warp.stretch_ratio_b < 1.2)


def test_robust_quantile_normalize():
    """Verify outlier resistance and zero-division protection."""
    # Data with a single massive outlier spike
    x = np.array([1.0, 1.1, 1.2, 1.05, 0.95, 1.15, 1.0, 1000.0], dtype=np.float64)
    norm = robust_quantile_normalize(x, p_low=10.0, p_high=90.0)

    # Outlier clamped to 1.0; normal cluster nicely mapped across [0, 1]
    assert norm[-1] == 1.0
    assert norm[4] == 0.0  # lowest point in cluster
    assert np.all(np.isfinite(norm))
    assert np.all((norm >= 0.0) & (norm <= 1.0))

    # Flat array should not NaN
    flat = np.array([5.0, 5.0, 5.0], dtype=np.float64)
    norm_flat = robust_quantile_normalize(flat)
    assert np.all(norm_flat == 0.0)

    # Empty array
    assert robust_quantile_normalize(np.array([])).size == 0


def test_fdn_freeze_energy_conservation():
    """Verify lossless unitary state lock during FDN freeze."""
    sr = 48000
    n_samples = 48000  # 1.0s

    # Inject brief burst of audio into FDN, then freeze for remaining time
    audio = np.zeros((2, n_samples), dtype=np.float32)
    burst_len = 2000
    audio[:, :burst_len] = 0.5 * np.sin(2.0 * np.pi * 440.0 * np.arange(burst_len) / sr)

    # Freeze engaged at sample 5000 until end
    freeze_start = 5000
    wet = process_fdn_freeze_numpy(
        audio,
        sr=sr,
        freeze_start_sample=freeze_start,
        freeze_end_sample=n_samples,
        feedback_gain=0.88,
        wet_mix=1.0,
        dry_mix=0.0,
        ramp_samples=256,
    )

    # During the frozen plateau (e.g., sample 15000 to 45000), audio should sustain without decaying to 0
    plateau_rms_early = float(np.sqrt(np.mean(wet[:, 15000:20000] ** 2)))
    plateau_rms_late = float(np.sqrt(np.mean(wet[:, 40000:45000] ** 2)))

    assert plateau_rms_early > 0.01
    assert plateau_rms_late > 0.01
    # Because of lossless unitary feedback, late plateau RMS should remain close to early plateau RMS
    ratio = plateau_rms_late / plateau_rms_early
    assert 0.70 < ratio < 1.30


def test_sidechain_ducking_gain_reduction():
    """Verify measurable ducking attenuation when key audio bursts occur."""
    sr = 48000
    duration_s = 2.0
    n_samples = int(duration_s * sr)

    # Target: continuous steady 100 Hz bass tone
    time = np.arange(n_samples) / sr
    bass_mono = 0.5 * np.sin(2.0 * np.pi * 100.0 * time).astype(np.float32)
    target = np.vstack([bass_mono, bass_mono])

    # Key: silence except for a loud kick drum burst from 0.5s to 0.7s
    key = np.zeros_like(target)
    kick_s = int(0.5 * sr)
    kick_e = int(0.7 * sr)
    kick_time = np.arange(kick_e - kick_s) / sr
    key[:, kick_s:kick_e] = 0.9 * np.sin(2.0 * np.pi * 60.0 * kick_time)

    ducked = process_sidechain_compressor_numpy(
        target,
        key,
        sr=sr,
        bpm=120.0,
        attack_ms=2.0,
        release_beat_fraction=0.25,
        threshold_db=-20.0,
        ratio=8.0,
    )

    # RMS during kick burst should be significantly lower than during un-ducked segment (0.1s to 0.3s)
    unducked_rms = float(np.sqrt(np.mean(ducked[:, int(0.1 * sr) : int(0.3 * sr)] ** 2)))
    ducked_rms = float(np.sqrt(np.mean(ducked[:, int(0.52 * sr) : int(0.68 * sr)] ** 2)))

    # Attenuation should be at least 4 dB (ratio < 0.65)
    assert ducked_rms < 0.65 * unducked_rms


def test_zdf_svf_sweep_attenuation():
    """Verify ZDF SVF lowpass and highpass frequency attenuation."""
    sr = 48000
    n_samples = 48000
    time = np.arange(n_samples) / sr

    # Two-tone: 100 Hz (low) + 5000 Hz (high)
    low_tone = 0.4 * np.sin(2.0 * np.pi * 100.0 * time)
    high_tone = 0.4 * np.sin(2.0 * np.pi * 5000.0 * time)
    signal_in = np.vstack([low_tone + high_tone, low_tone + high_tone]).astype(np.float32)

    # HPF sweeping from 1000 Hz to 2000 Hz should attenuate the 100 Hz tone
    hpf_out = process_zdf_svf_sweep_numpy(
        signal_in,
        sr=sr,
        start_cutoff_hz=1000.0,
        end_cutoff_hz=2000.0,
        resonance_q=1.0,
        mode="hpf",
    )
    # Spectral energy check
    spec_in = np.abs(np.fft.rfft(signal_in[0]))
    spec_hpf = np.abs(np.fft.rfft(hpf_out[0]))
    freqs = np.fft.rfftfreq(n_samples, 1.0 / sr)

    low_mask = (freqs >= 80.0) & (freqs <= 120.0)
    high_mask = (freqs >= 4900.0) & (freqs <= 5100.0)

    # 100 Hz should be heavily attenuated (> 20 dB reduction)
    assert np.sum(spec_hpf[low_mask]) < 0.1 * np.sum(spec_in[low_mask])
    # 5000 Hz should pass through largely intact
    assert np.sum(spec_hpf[high_mask]) > 0.7 * np.sum(spec_in[high_mask])


def test_time_stretch_phase_vocoder():
    """Verify STFT phase vocoder time-stretching."""
    sr = 22050
    duration_s = 1.0
    n_samples = int(duration_s * sr)
    time = np.arange(n_samples) / sr
    audio = 0.5 * np.sin(2.0 * np.pi * 440.0 * time).astype(np.float32)
    stereo = np.vstack([audio, audio])

    # Stretch by 1.05
    stretched = time_stretch_phase_vocoder_numpy(stereo, stretch_ratio=1.05)
    assert stretched.shape[0] == 2
    assert np.all(np.isfinite(stretched))
    assert stretched.shape[1] > 0


"""Unit tests for Drum Re-grooving, Vocal Chops, Master Bus Glue, and Autonomous Duration."""

from __future__ import annotations
import numpy as np
import pytest

from dcp.rendering.drum_regroove import detect_transient_onsets, regroove_drums_half_time
from dcp.rendering.vocal_chop import generate_vocal_chops
from dcp.rendering.dsp_fallback import process_master_bus_glue_numpy


def test_drum_regroove_detects_onsets_and_builds_half_time_pocket():
    sr = 48000
    n = sr * 4
    # Create synthetic drum track with distinct transients at 120 BPM (every 0.5s)
    drums = np.zeros(n, dtype=np.float32)
    for beat in range(8):
        pos = int(beat * 0.5 * sr)
        drums[pos : pos + 1000] = np.sin(np.linspace(0, 100, 1000)) * np.exp(-np.linspace(0, 5, 1000))

    onsets = detect_transient_onsets(drums, sr)
    assert len(onsets) >= 4

    # Regroove to half-time pocket over 4.0s
    regrooved = regroove_drums_half_time(drums, sr=sr, bpm=120.0, duration_s=4.0)
    assert regrooved.shape == (2, 4 * sr)
    assert np.all(np.isfinite(regrooved))
    assert np.max(np.abs(regrooved)) <= 1.0


def test_vocal_chop_rhythmic_sampler():
    sr = 48000
    n = sr * 4
    time = np.arange(n) / sr
    # Vocal audio with vocal-like tone
    vocal = (0.5 * np.sin(2 * np.pi * 440 * time) * (1.0 + 0.3 * np.sin(2 * np.pi * 5 * time))).astype(np.float32)

    chops = generate_vocal_chops(vocal, sr=sr, bpm=120.0, duration_s=3.0, level_db=-6.0)
    assert chops.shape == (2, int(3.0 * sr))
    assert np.all(np.isfinite(chops))
    assert np.max(np.abs(chops)) <= 1.0
    # Must contain audible slices
    assert np.sqrt(np.mean(chops ** 2)) > 0.001


def test_master_bus_glue_compressor():
    sr = 48000
    n = sr * 2
    time = np.arange(n) / sr
    hot_audio = (1.4 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
    stereo_hot = np.vstack([hot_audio, hot_audio])

    # Glue compressor with -0.5 dBFS ceiling (~0.944 linear)
    glued = process_master_bus_glue_numpy(stereo_hot, sr=sr, threshold_db=-6.0, ceiling_dbfs=-0.5)
    assert glued.shape == stereo_hot.shape
    assert np.all(np.isfinite(glued))
    # True peak must not exceed ceiling
    assert np.max(np.abs(glued)) <= 0.95


def test_studio_remix_derives_duration_autonomously():
    """Verify Studio Remix ignores requested duration and derives bar-aligned duration."""
    from dcp.models import AudioBuffer
    from dcp.transitions.planner import plan_transition
    from tests.test_composer_and_telemetry import _make_test_analysis

    sr = 48000
    n = sr * 36
    time = np.arange(n) / sr

    drums = AudioBuffer(np.vstack([0.3 * np.sin(2 * np.pi * 60 * time)] * 2).astype(np.float32), sr)
    bass = AudioBuffer(np.vstack([0.25 * np.sin(2 * np.pi * 120 * time)] * 2).astype(np.float32), sr)
    vocals = AudioBuffer(np.vstack([0.2 * np.sin(2 * np.pi * 440 * time)] * 2).astype(np.float32), sr)
    other = AudioBuffer(np.vstack([0.2 * np.sin(2 * np.pi * 880 * time)] * 2).astype(np.float32), sr)
    mix = AudioBuffer((drums.samples + bass.samples + vocals.samples + other.samples) * 0.5, sr)

    stems = {"drums": drums, "bass": bass, "vocals": vocals, "other": other}
    downbeats = tuple(float(i * 2.0) for i in range(18))  # 120 BPM: bar = 2.0s

    analysis = _make_test_analysis(mix, tempo_bpm=120.0, downbeats=downbeats)

    # Request an arbitrary 5.0 seconds
    plan = plan_transition(
        mix,
        mix,
        stems_a=stems,
        stems_b=stems,
        strategy="studio_remix",
        duration_s=5.0,  # Arbitrary slider value
        analysis_a=analysis,
        analysis_b=analysis,
    )

    # The AI Producer MUST decide musical bar length (12 or 16 bars @ 2.0s = 24.0s or 32.0s), NOT 5.0s!
    assert plan.duration_s >= 20.0, f"Expected autonomous bar-aligned duration, got {plan.duration_s}s"
    assert "total_bridge_bars" in plan.metadata
    assert plan.metadata["total_bridge_bars"] in (12, 16)


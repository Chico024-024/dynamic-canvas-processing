"""Tests for harmonic key detection, Camelot compatibility, and downbeat estimation."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_tone
from dcp.analysis import (
    camelot_distance,
    detect_key,
    estimate_downbeats,
    key_compatibility_score,
)
from dcp.analysis.basic import analyze_track
from dcp.transitions import plan_transition


def test_camelot_distance_identical():
    assert camelot_distance("8A", "8A") == 0
    assert camelot_distance("1B", "1B") == 0
    assert camelot_distance("12A", "12A") == 0


def test_camelot_distance_adjacent_and_circular():
    # Adjacent on wheel (±1)
    assert camelot_distance("8A", "9A") == 1
    assert camelot_distance("8A", "7A") == 1
    # Circular 12 <-> 1
    assert camelot_distance("12A", "1A") == 1
    assert camelot_distance("1A", "12A") == 1
    assert camelot_distance("12B", "1B") == 1


def test_camelot_distance_relative_major_minor():
    # Same number, different mode letter (e.g. Am = 8A, C = 8B)
    assert camelot_distance("8A", "8B") == 1
    assert camelot_distance("1A", "1B") == 1
    assert camelot_distance("12B", "12A") == 1


def test_camelot_distance_distant_and_invalid():
    assert camelot_distance("8A", "10A") == 2
    assert camelot_distance("8A", "2A") == 6
    assert camelot_distance("?", "8A") == -1
    assert camelot_distance("8A", "?") == -1
    assert camelot_distance("", "8A") == -1
    assert camelot_distance("invalid", "8A") == -1


def test_key_compatibility_score():
    assert key_compatibility_score("8A", "8A") == 1.0
    assert key_compatibility_score("8A", "9A") == 0.9
    assert key_compatibility_score("8A", "8B") == 0.9
    assert key_compatibility_score("8A", "10A") == 0.6
    assert key_compatibility_score("?", "8A") == 0.5
    assert key_compatibility_score("8A", "?") == 0.5


def test_detect_key_silence_or_short():
    silent = np.zeros(1000, dtype=np.float32)
    key, camelot, mode, conf = detect_key(silent, 11025)
    assert key == "unknown"
    assert camelot == "?"
    assert conf == 0.0


def test_detect_key_on_synthetic_tone():
    # 440 Hz is A4 (A major / minor component)
    audio = make_tone(frequency=440.0, duration_s=1.5, sample_rate=22050)
    key, camelot, mode, conf = detect_key(audio.mono(), 22050)
    assert isinstance(key, str) and key != "unknown"
    assert isinstance(camelot, str) and camelot != "?"
    assert mode in ("major", "minor")
    assert 0.0 <= conf <= 1.0


def test_analyze_track_populates_key_and_downbeats():
    audio = make_tone(frequency=440.0, duration_s=2.0, sample_rate=22050)
    analysis = analyze_track(audio)
    assert analysis.key is not None
    assert analysis.camelot_code is not None
    assert isinstance(analysis.key_confidence, float)
    assert isinstance(analysis.downbeat_times_s, tuple)
    # Check to_dict serialization
    d = analysis.to_dict()
    assert "key" in d
    assert "camelot_code" in d
    assert "downbeat_times_s" in d


def test_estimate_downbeats_identifies_best_phase():
    sr = 11025
    # Create 8 beats at 0.5s intervals (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5)
    beats = [0.1 + 0.5 * i for i in range(8)]
    # Create an audio buffer where beats at indices 0 and 4 have higher energy
    audio = np.zeros(int(4.5 * sr), dtype=np.float32)
    for i in (0, 4):
        center = int(beats[i] * sr)
        audio[center - 100 : center + 100] = 0.9

    downbeats = estimate_downbeats(audio, sr, beats, beats_per_bar=4)
    assert len(downbeats) == 2
    assert downbeats[0] == pytest.approx(beats[0])
    assert downbeats[1] == pytest.approx(beats[4])


def test_estimate_downbeats_too_few_beats():
    audio = np.zeros(1000, dtype=np.float32)
    downbeats = estimate_downbeats(audio, 11025, [0.1, 0.5])
    assert downbeats == ()


def test_rhythmic_strategy_aligns_to_downbeats(synthetic_pair):
    song_a, song_b = synthetic_pair
    beat_s = 0.5
    # 32 beats: indices 0, 4, 8, 12, 16, 20, 24, 28 are downbeats
    beats = tuple(0.13 + beat_s * i for i in range(32))
    downbeats = tuple(beats[i] for i in range(0, 32, 4))

    from conftest import make_analysis

    analysis_a = make_analysis(
        song_a.mix,
        tempo_bpm=120.0,
        beat_times_s=beats,
        downbeat_times_s=downbeats,
        analyzer="librosa",
    )
    analysis_b = make_analysis(song_b.mix, tempo_bpm=120.0)

    plan = plan_transition(
        song_a.mix,
        song_b.mix,
        stems_a=song_a.stems,
        stems_b=song_b.stems,
        strategy="rhythmic",
        duration_s=7.0,
        analysis_a=analysis_a,
        analysis_b=analysis_b,
    )

    assert plan.strategy == "rhythmic"
    assert plan.metadata.get("downbeat_aligned") is True
    # Anchor beat must be in downbeats
    anchor = plan.metadata["anchor_beat_s"]
    assert any(abs(anchor - db) < 0.05 for db in downbeats)
    assert any("downbeat" in reason.lower() for reason in plan.reasoning)

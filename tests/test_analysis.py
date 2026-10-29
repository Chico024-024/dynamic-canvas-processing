"""Basic analysis tests, including confidence-gated tempo behavior."""

from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import make_analysis, make_constant, make_silence, make_tone
from dcp.analysis import analyze_track, amplitude_to_dbfs
from dcp.config import DEFAULT_CONFIG
from dcp.models import AnalysisError, AudioBuffer
from dcp.transitions import TransitionContext, tempo_is_usable


def test_amplitude_to_dbfs_floor():
    assert amplitude_to_dbfs(0.0) == -120.0
    assert amplitude_to_dbfs(1.0) == 0.0
    assert amplitude_to_dbfs(0.5) == pytest.approx(-6.0206, abs=1e-3)


def test_sine_peak_and_rms_are_accurate():
    analysis = analyze_track(make_tone(duration_s=2.0, amplitude=0.5))
    assert analysis.peak_dbfs == pytest.approx(-6.0206, abs=0.02)
    assert analysis.rms_dbfs == pytest.approx(-9.0309, abs=0.02)
    assert analysis.silence_ratio == 0.0
    assert analysis.sample_rate == 22050
    assert analysis.channels == 2


def test_silence_reports_no_tempo_and_full_silence_ratio():
    analysis = analyze_track(make_silence(duration_s=2.0))
    assert analysis.tempo_bpm is None
    assert analysis.tempo_confidence == 0.0
    assert analysis.silence_ratio == 1.0
    assert analysis.peak_dbfs == -120.0
    assert any("Tempo estimate unavailable" in note for note in analysis.notes)


def test_silence_ratio_tracks_half_silent_track():
    tone = make_tone(duration_s=2.0, amplitude=0.3)
    samples = np.concatenate(
        [tone.samples, np.zeros_like(tone.samples)], axis=1
    )
    buffer = AudioBuffer(samples, tone.sample_rate)
    analysis = analyze_track(buffer)
    assert analysis.silence_ratio == pytest.approx(0.5, abs=0.03)


def test_short_track_skips_tempo_estimation():
    analysis = analyze_track(make_tone(duration_s=0.5))
    assert analysis.tempo_bpm is None
    assert any("Tempo estimate unavailable" in note for note in analysis.notes)


def test_tempo_estimate_for_120_bpm_synthetic_mix(synthetic_pair):
    song_a, _ = synthetic_pair
    analysis = analyze_track(song_a.mix)
    assert analysis.tempo_bpm is not None
    assert analysis.tempo_bpm == pytest.approx(120.0, abs=3.0)
    assert analysis.tempo_confidence >= 0.3
    assert analysis.onset_count > 0
    assert any("not a full beat tracker" in note for note in analysis.notes)


def test_analysis_rejects_non_finite_audio():
    samples = np.zeros((2, 1000), dtype=np.float64)
    samples[1, 10] = np.inf
    with pytest.raises(AnalysisError, match="NaN or infinite"):
        analyze_track(AudioBuffer(samples, 22050))


def test_analysis_to_dict_is_json_serializable():
    analysis = analyze_track(make_tone(duration_s=2.0))
    payload = analysis.to_dict()
    encoded = json.dumps(payload)
    assert "tempo_bpm" in encoded
    assert payload["duration_s"] == 2.0


def test_constant_signal_has_zero_crest_in_metrics():
    analysis = analyze_track(make_constant(0.25, duration_s=1.0))
    assert analysis.peak_dbfs == pytest.approx(-12.0412, abs=1e-3)
    assert analysis.rms_dbfs == pytest.approx(-12.0412, abs=1e-3)


def _context(
    tempo_a: float | None,
    confidence_a: float,
    tempo_b: float | None,
    confidence_b: float,
) -> TransitionContext:
    song = make_tone(duration_s=4.0)
    return TransitionContext(
        song_a=song,
        song_b=song,
        stems_a=None,
        stems_b=None,
        analysis_a=make_analysis(
            song, tempo_bpm=tempo_a, confidence=confidence_a
        ),
        analysis_b=make_analysis(
            song, tempo_bpm=tempo_b, confidence=confidence_b
        ),
        requested_duration_s=4.0,
        requested_strategy="rhythmic",
        config=DEFAULT_CONFIG,
    )


def test_tempo_is_usable_for_matching_tempos():
    usable, reason = tempo_is_usable(_context(120.0, 0.9, 120.0, 0.8))
    assert usable is True
    assert reason == ""


def test_tempo_is_usable_for_double_and_half_time():
    assert tempo_is_usable(_context(120.0, 0.9, 240.0, 0.9))[0] is True
    assert tempo_is_usable(_context(120.0, 0.9, 60.0, 0.9))[0] is True


def test_tempo_rejects_mismatch_and_low_confidence():
    usable, reason = tempo_is_usable(_context(120.0, 0.9, 130.0, 0.9))
    assert usable is False
    assert "tempo ratio" in reason

    usable, reason = tempo_is_usable(_context(120.0, 0.1, 120.0, 0.9))
    assert usable is False
    assert "confidence" in reason

    usable, reason = tempo_is_usable(_context(None, 0.0, 120.0, 0.9))
    assert usable is False
    assert "Song A" in reason

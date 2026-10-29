"""Optional librosa analysis backend and its fallback behavior."""

from __future__ import annotations

import json

import pytest

from conftest import make_tone
from dcp.analysis import (
    analyze_track,
    analyze_track_librosa,
    analyze_track_with_backend,
    librosa_available,
)
from dcp.config import DEFAULT_CONFIG
from dcp.models import AnalysisError

requires_librosa = pytest.mark.skipif(
    not librosa_available(), reason="optional librosa backend is not installed"
)


def test_basic_backend_has_no_beats_and_reports_analyzer():
    analysis = analyze_track(make_tone(duration_s=2.0))
    assert analysis.analyzer == "basic"
    assert analysis.beat_times_s == ()
    payload = analysis.to_dict()
    assert payload["analyzer"] == "basic"
    assert payload["beat_times_s"] == []


def test_explicit_basic_backend_never_returns_beats():
    analysis = analyze_track_with_backend(
        make_tone(duration_s=2.0), backend="basic"
    )
    assert analysis.analyzer == "basic"
    assert analysis.beat_times_s == ()


def test_unknown_backend_is_rejected():
    with pytest.raises(AnalysisError, match="unknown analysis backend"):
        analyze_track_with_backend(
            make_tone(duration_s=2.0), backend="definitely-not-a-backend"
        )


def test_librosa_request_falls_back_for_very_short_audio():
    analysis = analyze_track_librosa(make_tone(duration_s=0.5))
    assert analysis.analyzer == "basic"
    assert analysis.beat_times_s == ()


def test_auto_backend_falls_back_when_librosa_fails(monkeypatch):
    monkeypatch.setattr(
        "dcp.analysis.librosa_backend.librosa_available", lambda: True
    )

    def fail(*_args, **_kwargs):
        raise AnalysisError("simulated librosa failure")

    monkeypatch.setattr(
        "dcp.analysis.librosa_backend.analyze_track_librosa", fail
    )
    analysis = analyze_track_with_backend(
        make_tone(duration_s=2.0), backend="auto"
    )
    assert analysis.analyzer == "basic"
    assert any("librosa analysis failed" in note for note in analysis.notes)

    with pytest.raises(AnalysisError, match="simulated librosa failure"):
        analyze_track_with_backend(
            make_tone(duration_s=2.0), backend="librosa"
        )


@requires_librosa
def test_librosa_backend_returns_checked_beats(synthetic_pair):
    song_a, _ = synthetic_pair
    analysis = analyze_track_librosa(song_a.mix)
    assert analysis.analyzer == "librosa"
    assert analysis.tempo_bpm == pytest.approx(120.0, abs=3.0)
    assert (
        analysis.tempo_confidence
        >= DEFAULT_CONFIG.analysis.tempo_confidence_threshold
    )
    assert len(analysis.beat_times_s) >= 8
    assert list(analysis.beat_times_s) == sorted(analysis.beat_times_s)
    assert all(0.0 <= t <= song_a.mix.duration_s for t in analysis.beat_times_s)
    assert any("librosa" in note for note in analysis.notes)
    payload = analysis.to_dict()
    assert payload["beat_times_s"]
    json.dumps(payload, allow_nan=False)


@requires_librosa
def test_auto_backend_prefers_librosa_when_installed(synthetic_pair):
    song_a, _ = synthetic_pair
    analysis = analyze_track_with_backend(song_a.mix, backend="auto")
    assert analysis.analyzer == "librosa"
    assert analysis.beat_times_s

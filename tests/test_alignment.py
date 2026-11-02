"""Tests for downbeat-to-downbeat phase alignment."""

from __future__ import annotations

import numpy as np
import pytest

from dcp.analysis import TrackAnalysis
from dcp.models import AudioBuffer
from dcp.transitions.alignment import (
    PhaseLockResult,
    calculate_phase_locked_cue_offset,
)
from dcp.transitions.strategies import RhythmicStrategy
from dcp.transitions.context import TransitionContext
from dcp.config import DEFAULT_CONFIG


def _make_mock_analysis(
    duration_s: float,
    tempo_bpm: float,
    tempo_confidence: float,
    downbeat_times: list[float],
    sample_rate: int = 44100,
) -> TrackAnalysis:
    """Create a mock TrackAnalysis for testing."""
    return TrackAnalysis(
        duration_s=duration_s,
        sample_rate=sample_rate,
        channels=2,
        peak_dbfs=-3.0,
        rms_dbfs=-12.0,
        silence_ratio=0.05,
        tempo_bpm=tempo_bpm,
        tempo_confidence=tempo_confidence,
        onset_count=100,
        tempo_ambiguous=False,
        notes=(),
        beat_times_s=tuple(downbeat_times),
        analyzer="test",
        key=None,
        key_confidence=0.0,
        camelot_code=None,
        downbeat_times_s=tuple(downbeat_times),
    )


def _make_mock_context(
    analysis_a: TrackAnalysis,
    analysis_b: TrackAnalysis,
    duration_a: float = 30.0,
    duration_b: float = 30.0,
    requested_strategy: str = "rhythmic",
    requested_duration_s: float = 8.0,
) -> TransitionContext:
    """Create a mock TransitionContext for testing."""
    # Create dummy audio buffers
    song_a = AudioBuffer(
        samples=np.zeros((2, int(duration_a * 44100)), dtype=np.float64),
        sample_rate=44100,
    )
    song_b = AudioBuffer(
        samples=np.zeros((2, int(duration_b * 44100)), dtype=np.float64),
        sample_rate=44100,
    )

    return TransitionContext(
        song_a=song_a,
        song_b=song_b,
        stems_a=None,
        stems_b=None,
        analysis_a=analysis_a,
        analysis_b=analysis_b,
        requested_duration_s=requested_duration_s,
        requested_strategy=requested_strategy,
        config=DEFAULT_CONFIG,
    )


def test_phase_lock_basic():
    """Test basic phase lock calculation with matching downbeats."""
    # Song A: 120 BPM, downbeats every 2 seconds (4 beats at 120 BPM = 2s per bar)
    downbeats_a = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0]
    analysis_a = _make_mock_analysis(
        duration_s=30.0,
        tempo_bpm=120.0,
        tempo_confidence=0.9,
        downbeat_times=downbeats_a,
    )

    # Song B: same tempo, downbeats at same positions
    downbeats_b = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0]
    analysis_b = _make_mock_analysis(
        duration_s=30.0,
        tempo_bpm=120.0,
        tempo_confidence=0.9,
        downbeat_times=downbeats_b,
    )

    context = _make_mock_context(analysis_a, analysis_b)

    # With anchor_bar_a=3, anchor_bar_b=3, lead_in_bars=2:
    # anchor_downbeat_a = downbeats_a[2] = 6.0s
    # anchor_downbeat_b = downbeats_b[2] = 6.0s
    # bar_duration_b = 60/120 * 4 = 2.0s
    # lead_in_duration = 2 bars * 2.0s = 4.0s
    # transition_start_a = anchor_downbeat_a - lead_in_duration = 6.0 - 4.0 = 2.0s
    # cue_offset_b = anchor_downbeat_b - lead_in_duration = 6.0 - 4.0 = 2.0s
    # In output timeline: Song B enters at 2.0s, reaches downbeat at 2.0 + (6.0 - 2.0) = 6.0s.
    # Song A reaches downbeat at 6.0s. Output downbeats coincide at 6.0s!
    from dcp.transitions.plan import TransitionParams
    params = TransitionParams(anchor_bar_a=3, anchor_bar_b=3)

    result = calculate_phase_locked_cue_offset(context, params=params, lead_in_bars=2, min_confidence=0.5)

    assert result is not None, "Phase lock should succeed with matching downbeats"
    assert isinstance(result, PhaseLockResult)
    assert abs(result.cue_offset_b_s - 2.0) < 0.001
    assert abs(result.transition_start_a_s - 2.0) < 0.001
    # Verify coincidence in output timeline:
    assert abs(result.transition_start_a_s + (result.anchor_downbeat_b_s - result.cue_offset_b_s) - result.anchor_downbeat_a_s) < 1e-9


def test_phase_lock_with_valid_cue_offset():
    """Test phase lock with valid cue offset (anchor_bar_b > lead_in_bars)."""
    # Song A: 120 BPM, downbeats every 2 seconds
    downbeats_a = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0]
    analysis_a = _make_mock_analysis(
        duration_s=30.0,
        tempo_bpm=120.0,
        tempo_confidence=0.9,
        downbeat_times=downbeats_a,
    )

    downbeats_b = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0]
    analysis_b = _make_mock_analysis(
        duration_s=30.0,
        tempo_bpm=120.0,
        tempo_confidence=0.9,
        downbeat_times=downbeats_b,
    )

    from dcp.transitions.plan import TransitionParams
    params = TransitionParams(anchor_bar_a=3, anchor_bar_b=3)

    context = _make_mock_context(analysis_a, analysis_b)

    result = calculate_phase_locked_cue_offset(
        context,
        params=params,
        lead_in_bars=2,
        min_confidence=0.5,
    )

    assert result is not None, "Phase lock should succeed"
    assert result.cue_offset_b_s >= 0, "Cue offset should be non-negative"

    assert abs(result.cue_offset_b_s - 2.0) < 0.001
    assert abs(result.anchor_downbeat_a_s - 6.0) < 0.001
    assert abs(result.anchor_downbeat_b_s - 6.0) < 0.001
    assert abs(result.lead_in_duration_s - 4.0) < 0.001
    assert abs(result.transition_start_a_s - 2.0) < 0.001
    assert result.confidence > 0.8
    # Exact output coincidence:
    assert abs(result.transition_start_a_s + (result.anchor_downbeat_b_s - result.cue_offset_b_s) - result.anchor_downbeat_a_s) < 1e-9


def test_phase_lock_fails_lead_in_exceeds_song_a():
    """Test that phase lock safely fails when requested lead-in exceeds Song A start."""
    downbeats_a = [2.0, 4.0, 6.0, 8.0, 10.0]
    analysis_a = _make_mock_analysis(
        duration_s=30.0,
        tempo_bpm=120.0,
        tempo_confidence=0.9,
        downbeat_times=downbeats_a,
    )
    downbeats_b = [2.0, 4.0, 6.0, 8.0, 10.0]
    analysis_b = _make_mock_analysis(
        duration_s=30.0,
        tempo_bpm=120.0,
        tempo_confidence=0.9,
        downbeat_times=downbeats_b,
    )
    from dcp.transitions.plan import TransitionParams
    # anchor_bar_a=1 (2.0s) with lead_in_bars=2 (4.0s) -> 2.0 - 4.0 = -2.0s
    params = TransitionParams(anchor_bar_a=1, anchor_bar_b=3)
    context = _make_mock_context(analysis_a, analysis_b)

    result = calculate_phase_locked_cue_offset(context, params=params, lead_in_bars=2, min_confidence=0.5)
    assert not result.success
    assert result.fallback_reason == "transition_start_out_of_bounds"


def test_phase_lock_fails_low_confidence():
    """Test phase lock fails when confidence is below threshold."""
    downbeats_a = [2.0, 4.0, 6.0, 8.0, 10.0]
    analysis_a = _make_mock_analysis(
        duration_s=30.0,
        tempo_bpm=120.0,
        tempo_confidence=0.9,
        downbeat_times=downbeats_a,
    )

    downbeats_b = [2.0, 4.0, 6.0, 8.0, 10.0]
    analysis_b = _make_mock_analysis(
        duration_s=30.0,
        tempo_bpm=120.0,
        tempo_confidence=0.2,  # Below threshold
        downbeat_times=downbeats_b,
    )

    context = _make_mock_context(analysis_a, analysis_b)

    result = calculate_phase_locked_cue_offset(context, min_confidence=0.5)

    assert not result.success, "Phase lock should fail with low confidence"
    assert result.fallback_reason == "low_confidence"


def test_phase_lock_fails_no_downbeats():
    """Test phase lock fails when downbeats are missing."""
    analysis_a = _make_mock_analysis(
        duration_s=30.0,
        tempo_bpm=120.0,
        tempo_confidence=0.9,
        downbeat_times=[],  # No downbeats
    )

    downbeats_b = [2.0, 4.0, 6.0, 8.0, 10.0]
    analysis_b = _make_mock_analysis(
        duration_s=30.0,
        tempo_bpm=120.0,
        tempo_confidence=0.9,
        downbeat_times=downbeats_b,
    )

    context = _make_mock_context(analysis_a, analysis_b)

    result = calculate_phase_locked_cue_offset(context, min_confidence=0.5)

    assert not result.success, "Phase lock should fail with missing downbeats"
    assert result.fallback_reason == "insufficient_downbeats"


def test_phase_lock_fails_cue_offset_negative():
    """Test phase lock fails when cue offset would be negative."""
    # Song A: anchor at bar 1 (2.0s)
    downbeats_a = [2.0, 4.0, 6.0, 8.0, 10.0]
    analysis_a = _make_mock_analysis(
        duration_s=30.0,
        tempo_bpm=120.0,
        tempo_confidence=0.9,
        downbeat_times=downbeats_a,
    )

    # Song B: same downbeats
    downbeats_b = [2.0, 4.0, 6.0, 8.0, 10.0]
    analysis_b = _make_mock_analysis(
        duration_s=30.0,
        tempo_bpm=120.0,
        tempo_confidence=0.9,
        downbeat_times=downbeats_b,
    )

    from dcp.transitions.plan import TransitionParams
    params = TransitionParams(anchor_bar_a=1, anchor_bar_b=1)

    context = _make_mock_context(analysis_a, analysis_b)

    # With anchor_bar_b=1 and lead_in_bars=2:
    # anchor_downbeat_b = 2.0, lead_in = 4.0s
    # cue_offset_b = 2.0 - 4.0 = -2.0 (negative -> should fail)
    result = calculate_phase_locked_cue_offset(
        context,
        params=params,
        lead_in_bars=2,
        min_confidence=0.5,
    )

    assert not result.success, "Phase lock should fail with negative cue offset"
    assert result.fallback_reason == "negative_cue_offset"


def test_rhythmic_strategy_phase_lock_integration():
    """Test that RhythmicStrategy uses phase lock when available."""
    # Create context with matching downbeats that allow phase lock
    downbeats_a = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
    analysis_a = _make_mock_analysis(
        duration_s=30.0,
        tempo_bpm=120.0,
        tempo_confidence=0.9,
        downbeat_times=downbeats_a,
    )

    # Song B needs anchor_bar_b > lead_in_bars for valid cue offset
    # With lead_in_bars=2, anchor_bar_b=3 gives downbeat at 6.0s
    downbeats_b = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
    analysis_b = _make_mock_analysis(
        duration_s=30.0,
        tempo_bpm=120.0,
        tempo_confidence=0.9,
        downbeat_times=downbeats_b,
    )

    context = _make_mock_context(analysis_a, analysis_b, requested_strategy="rhythmic", requested_duration_s=8.0)

    strategy = RhythmicStrategy()
    plan = strategy.plan(context)

    # Check that plan was created successfully
    assert plan is not None
    assert plan.strategy == "rhythmic"
    assert plan.duration_s > 0

    # Check that phase lock was achieved (should be True with our setup)
    assert plan.metadata["phase_locked"] is True
    assert plan.metadata["phase_lock_confidence"] > 0.5
    assert "cue_offset_b_s" in plan.metadata
    assert "anchor_downbeat_a_s" in plan.metadata
    assert "anchor_downbeat_b_s" in plan.metadata
    assert "transition_start_a_s" in plan.metadata
    assert plan.metadata["alignment"] == "phase_locked"

    # Verify timeline coincidence invariant:
    # Output downbeat B time = transition_start_a + (anchor_b - cue_b) == anchor_a
    t_start = plan.start_time_s
    anchor_a = plan.metadata["anchor_downbeat_a_s"]
    anchor_b = plan.metadata["anchor_downbeat_b_s"]
    cue_b = plan.metadata["cue_offset_b_s"]
    output_arrival_b = t_start + (anchor_b - cue_b)
    assert abs(output_arrival_b - anchor_a) < 1e-6, (
        f"Downbeat arrival {output_arrival_b:.6f}s does not match Song A anchor {anchor_a:.6f}s"
    )

    # Verify the phase lock reasoning is in the plan
    phase_lock_reasoning = [
        r for r in plan.reasoning if "phase lock" in r.lower()
    ]
    assert len(phase_lock_reasoning) > 0, "Phase lock reasoning should be in plan"


def test_rhythmic_strategy_fallback_without_phase_lock():
    """Test RhythmicStrategy falls back to beat alignment when phase lock fails."""
    # Good tempo confidence but missing downbeats -> tempo usable but no phase lock
    downbeats_a = [2.0, 4.0, 6.0, 8.0, 10.0]
    analysis_a = _make_mock_analysis(
        duration_s=30.0,
        tempo_bpm=120.0,
        tempo_confidence=0.9,
        downbeat_times=downbeats_a,
    )

    # Song B has no downbeats -> phase lock fails
    analysis_b = _make_mock_analysis(
        duration_s=30.0,
        tempo_bpm=120.0,
        tempo_confidence=0.9,
        downbeat_times=[],  # No downbeats -> phase lock fails
    )

    context = _make_mock_context(analysis_a, analysis_b, requested_strategy="rhythmic", requested_duration_s=8.0)

    strategy = RhythmicStrategy()
    plan = strategy.plan(context)

    assert plan is not None
    assert plan.strategy == "rhythmic"

    # Should NOT have phase lock metadata
    assert not plan.metadata.get("phase_locked", False)
    # Should have standard alignment
    assert plan.metadata.get("alignment") in ("detected_beats", "tempo_fractions")


def test_rendered_audio_downbeat_coincidence():
    """Verify that Song A and Song B downbeats coincide at the exact same output sample."""
    from dcp.rendering.stem_mix import render_plan
    sr = 44100
    duration_s = 30.0

    # 120 BPM: 0.5s per beat, 2.0s per bar (4/4 meter)
    downbeats_a = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0]
    downbeats_b = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0]

    analysis_a = _make_mock_analysis(
        duration_s=duration_s,
        tempo_bpm=120.0,
        tempo_confidence=0.9,
        downbeat_times=downbeats_a,
        sample_rate=sr,
    )
    analysis_b = _make_mock_analysis(
        duration_s=duration_s,
        tempo_bpm=120.0,
        tempo_confidence=0.9,
        downbeat_times=downbeats_b,
        sample_rate=sr,
    )

    samples_a = np.zeros((2, int(duration_s * sr)), dtype=np.float64)
    samples_b = np.zeros((2, int(duration_s * sr)), dtype=np.float64)

    song_a = AudioBuffer(samples=samples_a, sample_rate=sr)
    song_b = AudioBuffer(samples=samples_b, sample_rate=sr)

    context = TransitionContext(
        song_a=song_a,
        song_b=song_b,
        stems_a=None,
        stems_b=None,
        analysis_a=analysis_a,
        analysis_b=analysis_b,
        requested_duration_s=14.0,
        requested_strategy="rhythmic",
        config=DEFAULT_CONFIG,
    )

    strategy = RhythmicStrategy()
    plan = strategy.plan(context)
    assert plan.metadata.get("phase_locked") is True

    anchor_downbeat_a = plan.metadata["anchor_downbeat_a_s"]
    anchor_downbeat_b = plan.metadata["anchor_downbeat_b_s"]

    # Place unit impulses on the exact anchor downbeat samples:
    # Left channel has Song A impulse
    sample_a_idx = int(round(anchor_downbeat_a * sr))
    samples_a[0, sample_a_idx] = 1.0

    # Right channel has Song B impulse
    sample_b_idx = int(round(anchor_downbeat_b * sr))
    samples_b[1, sample_b_idx] = 1.0

    render_result = render_plan(song_a, song_b, plan)
    rendered = render_result.audio

    # In rendered audio:
    # Left channel contains Song A
    # Right channel contains Song B
    peak_sample_a = int(np.argmax(np.abs(rendered.samples[0])))
    peak_sample_b = int(np.argmax(np.abs(rendered.samples[1])))

    # Assert exact coincidence in the rendered WAV output!
    assert peak_sample_a == peak_sample_b, (
        f"Downbeats do not coincide in rendered audio: "
        f"Song A peak at sample {peak_sample_a} ({peak_sample_a / sr:.6f}s), "
        f"Song B peak at sample {peak_sample_b} ({peak_sample_b / sr:.6f}s)"
    )
    assert peak_sample_a == sample_a_idx, (
        f"Output downbeat should land at {sample_a_idx}, got {peak_sample_a}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
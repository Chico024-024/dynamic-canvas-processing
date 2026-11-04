"""Unit tests for Layer 2 Musical IR, Layer 3 Multi-Hypothesis Composer, and Layer 5 Telemetry."""

from __future__ import annotations

import json
import numpy as np
import pytest

from dcp.rendering.telemetry import evaluate_bridge_telemetry
from dcp.transitions.composer import compose_optimal_bridge
from dcp.transitions.musical_ir import (
    PhraseIR,
    TrackMusicalIR,
    build_track_musical_ir,
)


def _make_dummy_stems(duration_s: float = 32.0, sr: int = 48000) -> dict[str, np.ndarray]:
    n_samples = int(duration_s * sr)
    time = np.arange(n_samples) / sr
    drums = 0.4 * np.sin(2.0 * np.pi * 60.0 * time).astype(np.float32)
    bass = 0.3 * np.sin(2.0 * np.pi * 120.0 * time).astype(np.float32)
    vocals = 0.2 * np.sin(2.0 * np.pi * 440.0 * time).astype(np.float32)
    other = 0.2 * np.sin(2.0 * np.pi * 880.0 * time).astype(np.float32)
    return {
        "drums": np.vstack([drums, drums]),
        "bass": np.vstack([bass, bass]),
        "vocals": np.vstack([vocals, vocals]),
        "other": np.vstack([other, other]),
    }


def test_musical_ir_construction_and_json_serialization():
    """Verify Layer 2 TrackMusicalIR construction and JSON serialization."""
    sr = 48000
    stems = _make_dummy_stems(duration_s=32.0, sr=sr)
    # 16 bars at 2.0s per bar (120 BPM)
    downbeats = np.arange(0.0, 32.1, 2.0)

    ir = build_track_musical_ir(
        track_id="track_test",
        stems=stems,
        downbeats_s=downbeats,
        sr=sr,
        base_confidence=0.90,
    )

    assert ir.track_id == "track_test"
    assert abs(ir.base_bpm - 120.0) < 1.0
    assert len(ir.metrical_hypotheses) >= 2  # primary + metric_4_3 (and half_time if >= 128)
    assert len(ir.phrases) >= 3

    # Verify JSON serialization works without type errors
    json_str = ir.to_json()
    data = json.loads(json_str)
    assert data["track_id"] == "track_test"
    assert len(data["phrases"]) == len(ir.phrases)


def test_starboy_to_coming_down_multi_hypothesis_pivot():
    """
    Test the Starboy (93 BPM) -> Coming Down (118 BPM) benchmark:
    A direct tempo stretch of 27% (118/93 = 1.27) violates bounded phase vocoder (+/- 8%).
    The composer MUST evaluate multi-hypothesis options (4:3 metric modulation or FDN ghost wash)
    and NEVER select an out-of-bounds direct tempo warp.
    """
    sr = 48000
    stems_a = _make_dummy_stems(duration_s=40.0, sr=sr)
    stems_b = _make_dummy_stems(duration_s=40.0, sr=sr)

    # Track A @ 93 BPM (bar = 4 * 60 / 93 ≈ 2.5806s)
    bar_dur_a = 4.0 * 60.0 / 93.0
    downbeats_a = np.arange(0.0, 40.0, bar_dur_a)

    # Track B @ 118 BPM (bar = 4 * 60 / 118 ≈ 2.0339s)
    bar_dur_b = 4.0 * 60.0 / 118.0
    downbeats_b = np.arange(0.0, 40.0, bar_dur_b)

    ir_a = build_track_musical_ir("starboy_93", stems_a, downbeats_a, sr=sr)
    ir_b = build_track_musical_ir("coming_down_118", stems_b, downbeats_b, sr=sr)

    # Hypotheses check
    hypo_ids_a = {h.hypothesis_id for h in ir_a.metrical_hypotheses}
    assert "metric_4_3" in hypo_ids_a

    plan = compose_optimal_bridge(ir_a, ir_b, min_feasibility_threshold=0.50)

    # The plan must be viable and use either 4:3 pivot or atemporal wash
    assert plan.is_viable is True
    assert plan.transition_mode in ("METRIC_MODULATION_PIVOT", "ATEMPORAL_WASH_HANDOFF")
    assert plan.archetype in ("POLYRHYTHMIC_4_3_PIVOT", "GHOST_WASH_SUSPENSION")
    # Must NOT force a 27% direct tempo warp
    assert plan.transition_mode != "CONTINUOUS_TEMPO_WARP"


def test_rejection_gate_on_unresolvable_clash():
    """Verify that unresolvable dissonance / clashing tracks trigger the Rejection Gate."""
    sr = 48000
    stems_a = _make_dummy_stems(duration_s=32.0, sr=sr)
    stems_b = _make_dummy_stems(duration_s=32.0, sr=sr)

    # 85 BPM vs 145 BPM: ratio 1.70, no 4:3 alignment possible, threshold set strictly
    downbeats_a = np.arange(0.0, 32.0, 4.0 * 60.0 / 85.0)
    downbeats_b = np.arange(0.0, 32.0, 4.0 * 60.0 / 145.0)

    # Orthogonal, clashing chroma profiles (C major vs F# major)
    chroma_a = np.zeros((16, 12), dtype=np.float64)
    chroma_a[:, 0] = 1.0  # C
    chroma_b = np.zeros((16, 12), dtype=np.float64)
    chroma_b[:, 6] = 1.0  # F# (tritone clash)

    ir_a = build_track_musical_ir("track_a", stems_a, downbeats_a, sr=sr, chroma_per_bar=chroma_a)
    ir_b = build_track_musical_ir("track_b", stems_b, downbeats_b, sr=sr, chroma_per_bar=chroma_b)

    # High threshold should reject the transition
    plan = compose_optimal_bridge(
        ir_a, ir_b, min_feasibility_threshold=0.88, allow_pitch_shift_semitones=0
    )

    assert plan.is_viable is False
    assert plan.transition_mode == "REJECTED"
    assert plan.archetype == "REJECTED_FALLBACK"
    assert plan.rejection_reason is not None
    assert "rejected" in plan.composer_rationale.lower()


def test_breath_window_suppression_when_vocals_active():
    """Verify that Breath silence is suppressed when vocals are active mid-phrase."""
    sr = 48000
    stems_a = _make_dummy_stems(duration_s=32.0, sr=sr)
    stems_b = _make_dummy_stems(duration_s=32.0, sr=sr)

    # Both tracks @ 120 BPM
    downbeats = np.arange(0.0, 32.1, 2.0)
    ir_a = build_track_musical_ir("track_a", stems_a, downbeats, sr=sr)
    ir_b = build_track_musical_ir("track_b", stems_b, downbeats, sr=sr)

    # Overwrite phrase vocal resolution to simulate singer holding a sustained lyric at phrase end
    phrases_modified = []
    for p in ir_a.phrases:
        phrases_modified.append(
            PhraseIR(
                phrase_index=p.phrase_index,
                bar_start=p.bar_start,
                bar_end=p.bar_end,
                start_time_s=p.start_time_s,
                end_time_s=p.end_time_s,
                structural_role=p.structural_role,
                boundary_salience=p.boundary_salience,
                chroma_12d=p.chroma_12d,
                bass_root_pc=p.bass_root_pc,
                vocal_activity=0.85,
                vocal_resolves_at_end=False,  # Mid-phrase lyric!
                low_end_weight=p.low_end_weight,
                drum_crest_factor=p.drum_crest_factor,
            )
        )
    ir_a_mid_lyric = TrackMusicalIR(
        track_id="mid_lyric_a",
        sample_rate=sr,
        duration_s=ir_a.duration_s,
        base_bpm=120.0,
        metrical_hypotheses=ir_a.metrical_hypotheses,
        phrases=tuple(phrases_modified),
    )

    plan = compose_optimal_bridge(ir_a_mid_lyric, ir_b, min_feasibility_threshold=0.45)

    if plan.is_viable:
        # Breath window MUST be suppressed (0 beats) to avoid cutting the vocal
        assert plan.breath_beats == 0
        assert plan.archetype == "CONTINUOUS_CROSS_FILTER"


def test_layer_5_audio_telemetry_qc_pass():
    """Verify Layer 5 telemetry passes studio QC on a clean rendered bridge."""
    sr = 48000
    duration_s = 4.0
    n_samples = int(duration_s * sr)
    time = np.arange(n_samples) / sr

    # Clean master: -6 dBFS peak, correlated stereo
    left = 0.5 * np.sin(2.0 * np.pi * 440.0 * time).astype(np.float32)
    right = 0.5 * np.sin(2.0 * np.pi * 440.0 * time + 0.1).astype(np.float32)
    master = np.vstack([left, right])

    report = evaluate_bridge_telemetry(master, sr=sr)

    assert report.passes_studio_qc is True
    assert report.true_peak_dbfs < 0.0
    assert report.min_stereo_phase_correlation > 0.80
    assert len(report.qc_warnings) == 0


def test_layer_5_audio_telemetry_flags_clipping_and_phase_cancellation():
    """Verify Layer 5 telemetry flags clipping and stereo phase cancellation."""
    sr = 48000
    n_samples = 48000

    # Digital clipping (> 1.0 linear = > 0.0 dBFS) and anti-phase stereo (L = -R)
    anti_phase = 1.2 * np.sin(np.linspace(0, 50, n_samples)).astype(np.float32)
    clipped_master = np.vstack([anti_phase, -anti_phase])

    report = evaluate_bridge_telemetry(clipped_master, sr=sr)

    assert report.passes_studio_qc is False
    assert report.true_peak_dbfs > 0.0
    assert report.min_stereo_phase_correlation < -0.90
    assert any("clipping" in w.lower() for w in report.qc_warnings)
    assert any("phase cancellation" in w.lower() for w in report.qc_warnings)


def _make_test_analysis(buffer, tempo_bpm: float, downbeats: tuple[float, ...]):
    from dcp.analysis.basic import TrackAnalysis
    return TrackAnalysis(
        duration_s=buffer.duration_s,
        sample_rate=buffer.sample_rate,
        channels=buffer.channels,
        peak_dbfs=-6.0,
        rms_dbfs=-12.0,
        silence_ratio=0.0,
        tempo_bpm=tempo_bpm,
        tempo_confidence=0.90,
        onset_count=32,
        tempo_ambiguous=False,
        notes=(),
        beat_times_s=downbeats,
        analyzer="basic",
        key="C major",
        camelot_code="8B",
        key_confidence=0.85,
        downbeat_times_s=downbeats,
    )


def test_studio_remix_strategy_uses_layer_3_composer():
    """Verify StudioRemixStrategy wires Layer 3 composer and populates archetype metadata."""
    from dcp.models import AudioBuffer
    from dcp.transitions.planner import plan_transition

    sr = 48000
    n = sr * 32
    time = np.arange(n) / sr

    # Create 32-second synthetic audio buffers
    drums = AudioBuffer(np.vstack([0.4 * np.sin(2 * np.pi * 60 * time)] * 2).astype(np.float32), sr)
    bass = AudioBuffer(np.vstack([0.3 * np.sin(2 * np.pi * 120 * time)] * 2).astype(np.float32), sr)
    vocals = AudioBuffer(np.vstack([0.2 * np.sin(2 * np.pi * 440 * time)] * 2).astype(np.float32), sr)
    other = AudioBuffer(np.vstack([0.2 * np.sin(2 * np.pi * 880 * time)] * 2).astype(np.float32), sr)
    mix = AudioBuffer((drums.samples + bass.samples + vocals.samples + other.samples) * 0.5, sr)

    stems_a = {"drums": drums, "bass": bass, "vocals": vocals, "other": other}
    stems_b = {"drums": drums, "bass": bass, "vocals": vocals, "other": other}

    downbeats_a = tuple(float(i * 2.0) for i in range(16))
    downbeats_b = tuple(float(i * 2.0) for i in range(16))

    analysis_a = _make_test_analysis(mix, tempo_bpm=120.0, downbeats=downbeats_a)
    analysis_b = _make_test_analysis(mix, tempo_bpm=120.0, downbeats=downbeats_b)

    plan = plan_transition(
        mix,
        mix,
        stems_a=stems_a,
        stems_b=stems_b,
        strategy="studio_remix",
        duration_s=8.0,
        analysis_a=analysis_a,
        analysis_b=analysis_b,
    )

    assert plan.strategy == "studio_remix"
    assert "archetype" in plan.metadata
    assert plan.metadata["archetype"] in (
        "VACUUM_BREATH_DROP",
        "CONTINUOUS_CROSS_FILTER",
        "POLYRHYTHMIC_4_3_PIVOT",
        "GHOST_WASH_SUSPENSION",
    )
    assert "feasibility_score" in plan.metadata
    assert plan.metadata["feasibility_score"] > 0.5
    assert "composer_rationale" in plan.metadata
    assert any("Layer 3 Composer" in r for r in plan.reasoning)


def test_studio_remix_handles_starboy_benchmark_with_polyrhythmic_pivot():
    """Verify StudioRemixStrategy adapts to Starboy (93 BPM) -> Coming Down (118 BPM)."""
    from dcp.models import AudioBuffer
    from dcp.transitions.planner import plan_transition

    sr = 48000
    n = sr * 40
    time = np.arange(n) / sr

    drums = AudioBuffer(np.vstack([0.4 * np.sin(2 * np.pi * 60 * time)] * 2).astype(np.float32), sr)
    bass = AudioBuffer(np.vstack([0.3 * np.sin(2 * np.pi * 120 * time)] * 2).astype(np.float32), sr)
    vocals = AudioBuffer(np.vstack([0.2 * np.sin(2 * np.pi * 440 * time)] * 2).astype(np.float32), sr)
    other = AudioBuffer(np.vstack([0.2 * np.sin(2 * np.pi * 880 * time)] * 2).astype(np.float32), sr)
    mix = AudioBuffer((drums.samples + bass.samples + vocals.samples + other.samples) * 0.5, sr)

    stems_a = {"drums": drums, "bass": bass, "vocals": vocals, "other": other}
    stems_b = {"drums": drums, "bass": bass, "vocals": vocals, "other": other}

    # Track A: 93 BPM (bar ~ 2.5806s)
    bar_a = 60.0 * 4.0 / 93.0
    downbeats_a = tuple(float(i * bar_a) for i in range(15))

    # Track B: 118 BPM (bar ~ 2.0339s)
    bar_b = 60.0 * 4.0 / 118.0
    downbeats_b = tuple(float(i * bar_b) for i in range(15))

    analysis_a = _make_test_analysis(mix, tempo_bpm=93.0, downbeats=downbeats_a)
    analysis_b = _make_test_analysis(mix, tempo_bpm=118.0, downbeats=downbeats_b)

    plan = plan_transition(
        mix,
        mix,
        stems_a=stems_a,
        stems_b=stems_b,
        strategy="studio_remix",
        duration_s=12.0,
        analysis_a=analysis_a,
        analysis_b=analysis_b,
    )

    assert plan.strategy == "studio_remix"
    # Should use POLYRHYTHMIC_4_3_PIVOT or GHOST_WASH_SUSPENSION, NOT naive continuous tempo warp
    assert plan.metadata["archetype"] in ("POLYRHYTHMIC_4_3_PIVOT", "GHOST_WASH_SUSPENSION")


def test_studio_remix_renders_and_passes_layer_5_telemetry():
    """Verify end-to-end render of Studio Remix bridge passes Layer 5 telemetry QC."""
    from dcp.models import AudioBuffer
    from dcp.rendering import render_plan
    from dcp.transitions.planner import plan_transition

    sr = 48000
    n = sr * 32
    time = np.arange(n) / sr

    drums = AudioBuffer(np.vstack([0.3 * np.sin(2 * np.pi * 60 * time)] * 2).astype(np.float32), sr)
    bass = AudioBuffer(np.vstack([0.25 * np.sin(2 * np.pi * 120 * time)] * 2).astype(np.float32), sr)
    vocals = AudioBuffer(np.vstack([0.2 * np.sin(2 * np.pi * 440 * time)] * 2).astype(np.float32), sr)
    other = AudioBuffer(np.vstack([0.2 * np.sin(2 * np.pi * 880 * time)] * 2).astype(np.float32), sr)
    mix = AudioBuffer((drums.samples + bass.samples + vocals.samples + other.samples) * 0.5, sr)

    stems_a = {"drums": drums, "bass": bass, "vocals": vocals, "other": other}
    stems_b = {"drums": drums, "bass": bass, "vocals": vocals, "other": other}

    downbeats_a = tuple(float(i * 2.0) for i in range(16))
    downbeats_b = tuple(float(i * 2.0) for i in range(16))

    analysis_a = _make_test_analysis(mix, tempo_bpm=120.0, downbeats=downbeats_a)
    analysis_b = _make_test_analysis(mix, tempo_bpm=120.0, downbeats=downbeats_b)

    plan = plan_transition(
        mix,
        mix,
        stems_a=stems_a,
        stems_b=stems_b,
        strategy="studio_remix",
        duration_s=8.0,
        analysis_a=analysis_a,
        analysis_b=analysis_b,
    )

    rendered = render_plan(
        mix,
        mix,
        plan,
        stems_a=stems_a,
        stems_b=stems_b,
    )

    assert rendered.audio is not None
    assert rendered.audio.samples.shape[0] == 2
    assert rendered.audio.samples.shape[1] > 0

    report = evaluate_bridge_telemetry(rendered.audio.samples, sr=sr)
    assert report.passes_studio_qc is True
    assert report.true_peak_dbfs <= 0.0




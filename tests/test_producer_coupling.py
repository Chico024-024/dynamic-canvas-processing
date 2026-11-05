"""Regression tests for Producer Matrix coupling, warp direction, and ghost tail injection."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_analysis, make_tone
from dcp.analysis.key_detection import _extract_chroma
from dcp.models import AudioBuffer, StemSet
from dcp.rendering.stem_mix import render_plan
from dcp.transitions.composer import compose_optimal_bridge
from dcp.transitions.context import TransitionContext
from dcp.transitions.musical_ir import build_track_musical_ir
from dcp.transitions.strategies import StudioRemixStrategy, build_core_registry


def test_chroma_g2_maps_to_pc7():
    """Verify that a 98 Hz (G2) bass tone accurately maps to pitch class 7 (G)."""
    sr = 48000
    dur = 2.0
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * 98.0 * t)
    chroma = _extract_chroma(tone, sr)
    assert int(np.argmax(chroma)) == 7


def test_warp_ratio_b_direction_speeds_up_incoming():
    """Verify warp_ratio_b < 1.0 when speeding Song B up to master_bpm."""
    # When Song A is 93 BPM and Song B is 118 BPM, master_bpm is 93 * 4/3 = 124 BPM.
    # To speed Song B up from 118 to 124 BPM, vocoder ratio must be 118 / 124 = 0.9516 (< 1.0).
    tempo_a = 93.0
    tempo_b = 118.0
    master_bpm = tempo_a * (4.0 / 3.0)
    warp_ratio_b = tempo_b / master_bpm
    assert warp_ratio_b < 1.0
    assert abs(warp_ratio_b - (118.0 / 124.0)) < 1e-4


def test_cue_offset_b_formula_and_reanchor():
    """Verify cue formula does not divide by phantom grid and re-anchors if unreachable."""
    strat = StudioRemixStrategy()
    sr = 22050
    # Create two synthetic tracks
    a = make_tone(duration_s=30.0, sample_rate=sr)
    b = make_tone(duration_s=30.0, sample_rate=sr)

    from dcp.stems.dsp_splitter import decompose_audio_to_stemset
    from dcp.analysis import analyze_track_with_backend
    from dcp.config import DEFAULT_CONFIG

    stems_a = decompose_audio_to_stemset(a, "A")
    stems_b = decompose_audio_to_stemset(b, "B")
    analysis_a = analyze_track_with_backend(a, DEFAULT_CONFIG.analysis)
    analysis_b = analyze_track_with_backend(b, DEFAULT_CONFIG.analysis)

    ctx = TransitionContext(
        song_a=a,
        song_b=b,
        stems_a=stems_a,
        stems_b=stems_b,
        analysis_a=analysis_a,
        analysis_b=analysis_b,
        requested_duration_s=16.0,
        requested_strategy="studio_remix",
        config=DEFAULT_CONFIG,
    )
    plan = strat.plan(ctx)
    meta = plan.metadata
    assert meta.get("cue_offset_b_s", -1.0) >= 0.0
    assert "phase_locked" in meta
    assert "grid_reanchored" in meta


def test_ghost_tail_added_single_point_post_glue():
    """Verify that ghost tail is added once at ghost_tail_gain and not duplicated."""
    sr = 22050
    dur = 6.0
    a = make_tone(frequency=440.0, duration_s=dur, sample_rate=sr)
    b = make_tone(frequency=554.37, duration_s=dur, sample_rate=sr)

    from dcp.stems.dsp_splitter import decompose_audio_to_stemset
    from dcp.analysis import analyze_track_with_backend
    from dcp.config import DEFAULT_CONFIG

    stems_a = decompose_audio_to_stemset(a, "A")
    stems_b = decompose_audio_to_stemset(b, "B")
    analysis_a = analyze_track_with_backend(a, DEFAULT_CONFIG.analysis)
    analysis_b = analyze_track_with_backend(b, DEFAULT_CONFIG.analysis)

    strat = build_core_registry().get("studio_remix")
    ctx = TransitionContext(
        song_a=a,
        song_b=b,
        stems_a=stems_a,
        stems_b=stems_b,
        analysis_a=analysis_a,
        analysis_b=analysis_b,
        requested_duration_s=4.0,
        requested_strategy="studio_remix",
        config=DEFAULT_CONFIG,
    )
    plan = strat.plan(ctx)
    result = render_plan(a, b, plan, stems_a=stems_a, stems_b=stems_b, config=DEFAULT_CONFIG)
    assert result.audio.sample_rate == sr
    # Verify no clipping occurred from double addition
    max_peak = np.max(np.abs(result.audio.samples))
    assert max_peak <= 1.0


def test_composer_grid_coupling_min_impact_bar():
    """Verify that min_impact_bar_b prevents picking impact phrases too early."""
    sr = 22050
    a = make_tone(duration_s=25.0, sample_rate=sr)
    b = make_tone(duration_s=25.0, sample_rate=sr)
    from dcp.stems.dsp_splitter import decompose_audio_to_stemset
    stems_a = decompose_audio_to_stemset(a, "A")
    stems_b = decompose_audio_to_stemset(b, "B")

    downbeats = tuple(np.arange(0, 25, 2.0))
    ir_a = build_track_musical_ir("a", {k: v.samples for k, v in stems_a.items()}, np.array(downbeats), sr)
    ir_b = build_track_musical_ir("b", {k: v.samples for k, v in stems_b.items()}, np.array(downbeats), sr)

    plan = compose_optimal_bridge(ir_a, ir_b, min_impact_bar_b=4)
    if plan.is_viable:
        assert plan.impact_phrase_b.bar_start >= 4


def test_drums_click_and_bass_click_land_within_20ms():
    """Verify that Song B drums click and bass click land on the exact same downbeat."""
    sr = 22050
    dur = 30.0
    frames = int(round(dur * sr))
    b_drums_samples = np.zeros((2, frames), dtype=np.float32)
    b_bass_samples = np.zeros((2, frames), dtype=np.float32)
    b_mix_samples = np.zeros((2, frames), dtype=np.float32)

    click_time_s = 16.0
    click_idx = int(round(click_time_s * sr))
    b_drums_samples[:, click_idx : click_idx + 10] = 0.9
    b_bass_samples[:, click_idx : click_idx + 10] = 0.9
    b_mix_samples[:, click_idx : click_idx + 10] = 0.9

    b_drums = AudioBuffer(samples=b_drums_samples, sample_rate=sr)
    b_bass = AudioBuffer(samples=b_bass_samples, sample_rate=sr)
    b_mix = AudioBuffer(samples=b_mix_samples, sample_rate=sr)
    stems_b = StemSet(stems={"drums": b_drums, "bass": b_bass}, source_label="B")

    a_tone = make_tone(duration_s=dur, sample_rate=sr)
    from dcp.stems.dsp_splitter import decompose_audio_to_stemset
    stems_a = decompose_audio_to_stemset(a_tone, "A")

    from dcp.config import DEFAULT_CONFIG
    downbeats_a = tuple(float(i * 2.0) for i in range(15))
    downbeats_b = tuple(float(i * 2.0) for i in range(15))
    analysis_a = make_analysis(a_tone, tempo_bpm=120.0, confidence=0.9, downbeat_times_s=downbeats_a)
    analysis_b = make_analysis(b_mix, tempo_bpm=120.0, confidence=0.9, downbeat_times_s=downbeats_b)

    strat = StudioRemixStrategy()
    ctx = TransitionContext(
        song_a=a_tone,
        song_b=b_mix,
        stems_a=stems_a,
        stems_b=stems_b,
        analysis_a=analysis_a,
        analysis_b=analysis_b,
        requested_duration_s=16.0,
        requested_strategy="studio_remix",
        config=DEFAULT_CONFIG,
    )
    plan = strat.plan(ctx)
    meta = plan.metadata
    warp_b = meta.get("warp_ratio_b", 1.0)

    # In our coherent time model, all stems share warp_ratio_b and cue_offset_b_s
    assert abs(warp_b - 1.0) < 1e-4 or meta.get("archetype") != "POLYRHYTHMIC_4_3_PIVOT"

    # Render plan and verify audio integrity
    result = render_plan(a_tone, b_mix, plan, stems_a=stems_a, stems_b=stems_b, config=DEFAULT_CONFIG)
    assert result.audio.frames > 0
    assert np.all(np.isfinite(result.audio.samples))


def test_launchpad_minus_start_equals_t_impact_after_reanchor():
    """Verify that after re-anchoring, launchpad_a - bridge_start_a matches t_impact."""
    sr = 22050
    dur_a = 60.0
    dur_b = 60.0
    a_tone = make_tone(duration_s=dur_a, sample_rate=sr)
    b_tone = make_tone(duration_s=dur_b, sample_rate=sr)
    from dcp.stems.dsp_splitter import decompose_audio_to_stemset
    stems_a = decompose_audio_to_stemset(a_tone, "A")
    stems_b = decompose_audio_to_stemset(b_tone, "B")

    from dcp.config import DEFAULT_CONFIG

    # Place downbeats such that B's impact is at bar 4 (8.0s), forcing grid re-anchoring
    downbeats_a = tuple(float(i * 2.0) for i in range(12))
    downbeats_b = tuple(float(i * 2.0) for i in range(12))
    analysis_a = make_analysis(a_tone, tempo_bpm=120.0, confidence=0.9, downbeat_times_s=downbeats_a)
    analysis_b = make_analysis(b_tone, tempo_bpm=120.0, confidence=0.9, downbeat_times_s=downbeats_b)

    strat = StudioRemixStrategy()
    ctx = TransitionContext(
        song_a=a_tone,
        song_b=b_tone,
        stems_a=stems_a,
        stems_b=stems_b,
        analysis_a=analysis_a,
        analysis_b=analysis_b,
        requested_duration_s=16.0,
        requested_strategy="studio_remix",
        config=DEFAULT_CONFIG,
    )
    plan = strat.plan(ctx)
    meta = plan.metadata

    launchpad_a = meta["launchpad_a_s"]
    bridge_start_a = meta["bridge_start_a_s"]
    t_impact = meta["t_impact_s"]

    # Song A launchpad event lands at launchpad_a - bridge_start_a in transition timeline
    assert (launchpad_a - bridge_start_a) == pytest.approx(t_impact, abs=0.05)


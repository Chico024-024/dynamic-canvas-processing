"""Strategy behavior, selection, quantization, and plan validation."""

from __future__ import annotations

from dataclasses import replace
import json
import math

import pytest

from conftest import make_analysis
from dcp.models import DCPError, MIX_STEM
from dcp.transitions import (
    StrategyRegistry,
    TransitionPlanner,
    build_core_registry,
    plan_transition,
    validate_plan,
)


def _planner() -> TransitionPlanner:
    return TransitionPlanner()


def _analyses(song_a, song_b, tempo_a=120.0, tempo_b=120.0, confidence=0.9):
    return (
        make_analysis(song_a.mix, tempo_bpm=tempo_a, confidence=confidence),
        make_analysis(song_b.mix, tempo_bpm=tempo_b, confidence=confidence),
    )


def test_all_seven_core_strategies_are_registered():
    registry = build_core_registry()
    assert registry.names() == (
        "atmospheric",
        "conservative",
        "crossfade",
        "optimized",
        "rhythmic",
        "stem_level",
        "studio_remix",
    )
    for name in registry.names():
        strategy = registry.get(name)
        assert strategy.name == name
        assert strategy.description


def test_registry_rejects_duplicate_and_invalid_strategies():
    registry = build_core_registry()
    with pytest.raises(DCPError, match="already registered"):
        registry.register(registry.get("crossfade"))

    class Blank:
        name = ""
        description = "blank"
        def plan(self, context):
            raise AssertionError("unused")

    with pytest.raises(DCPError, match="non-empty name"):
        registry.register(Blank())

    with pytest.raises(DCPError, match="unknown transition strategy"):
        registry.get("missing")


def test_crossfade_strategy_uses_full_mix_and_linear_curve(synthetic_pair):
    song_a, song_b = synthetic_pair
    analysis_a, analysis_b = _analyses(song_a, song_b)
    plan = plan_transition(
        song_a.mix,
        song_b.mix,
        strategy="crossfade",
        duration_s=8.0,
        analysis_a=analysis_a,
        analysis_b=analysis_b,
    )
    assert plan.strategy == "crossfade"
    assert plan.duration_s == pytest.approx(8.0)
    assert len(plan.automations) == 2
    assert all(automation.stem == MIX_STEM for automation in plan.automations)
    assert all(automation.curve == "linear" for automation in plan.automations)
    assert any("stems are intentionally ignored" in reason for reason in plan.reasoning)
    validate_plan(plan, sample_rate=song_a.mix.sample_rate)


def test_stem_level_strategy_builds_independent_automation(synthetic_pair):
    song_a, song_b = synthetic_pair
    analysis_a, analysis_b = _analyses(song_a, song_b)
    plan = plan_transition(
        song_a.mix,
        song_b.mix,
        stems_a=song_a.stems,
        stems_b=song_b.stems,
        strategy="stem_level",
        duration_s=8.0,
        analysis_a=analysis_a,
        analysis_b=analysis_b,
    )
    assert plan.strategy == "stem_level"
    assert len(plan.automations) == 8
    vocals_a = plan.automation_for("a", "vocals")
    vocals_b = plan.automation_for("b", "vocals")
    bass_a = plan.automation_for("a", "bass")
    assert vocals_a is not None and vocals_b is not None and bass_a is not None
    assert any(
        point.time_s == pytest.approx(0.45 * 8.0) and point.gain_db < -100
        for point in vocals_a.points
    )
    assert any(
        point.time_s == pytest.approx(0.55 * 8.0) and point.gain_db < -100
        for point in vocals_b.points
    )
    assert vocals_b.points[-1].time_s == pytest.approx(8.0)
    assert vocals_b.points[-1].gain_db == pytest.approx(0.0)
    assert any(
        point.time_s == pytest.approx(0.55 * 8.0) and point.gain_db < -100
        for point in bass_a.points
    )
    assert plan.metadata["stem_aware"] is True
    assert any("independent" in reason for reason in plan.reasoning)


def test_stem_level_without_stems_falls_back_to_conservative(synthetic_pair):
    song_a, song_b = synthetic_pair
    analysis_a, analysis_b = _analyses(song_a, song_b)
    plan = plan_transition(
        song_a.mix,
        song_b.mix,
        strategy="stem_level",
        duration_s=8.0,
        analysis_a=analysis_a,
        analysis_b=analysis_b,
    )
    assert plan.strategy == "conservative"
    assert plan.metadata["fallback"] is True
    assert any("without stems" in warning for warning in plan.warnings)


def test_rhythmic_strategy_quantizes_to_whole_bars(synthetic_pair):
    song_a, song_b = synthetic_pair
    analysis_a, analysis_b = _analyses(song_a, song_b, 120.0, 122.0)
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
    beat_s = 60.0 / 120.0
    assert plan.duration_s == pytest.approx(8.0)
    assert plan.metadata["bars"] == 4
    assert plan.metadata["total_beats"] == 16
    for automation in plan.automations:
        for point in automation.points:
            beats = point.time_s / beat_s
            assert abs(beats - round(beats)) < 1e-6
    assert any("whole beat" in reason for reason in plan.reasoning)


def test_rhythmic_strategy_uses_detected_beat_times(synthetic_pair):
    song_a, song_b = synthetic_pair
    beat_s = 0.5
    detected = tuple(0.13 + beat_s * index for index in range(32))
    analysis_a = make_analysis(
        song_a.mix,
        tempo_bpm=120.0,
        beat_times_s=detected,
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
    assert plan.metadata["alignment"] == "detected_beats"
    assert plan.metadata["anchor_beat_s"] == pytest.approx(9.13)
    assert plan.metadata["bars"] == 3
    assert plan.metadata["total_beats"] == 12
    assert plan.start_time_s == pytest.approx(9.13)
    assert plan.duration_s == pytest.approx(16.0 - 9.13)
    assert plan.start_time_s + plan.duration_s == pytest.approx(16.0)

    allowed = {round(index * beat_s, 6) for index in range(13)}
    allowed.add(round(plan.duration_s, 6))
    for automation in plan.automations:
        for point in automation.points:
            assert min(abs(point.time_s - t) for t in allowed) < 1e-6

    vocals_a = plan.automation_for("a", "vocals")
    drums_a = plan.automation_for("a", "drums")
    assert vocals_a is not None and drums_a is not None
    assert any(
        point.time_s == pytest.approx(1.0) and point.gain_db < -100
        for point in vocals_a.points
    )
    assert any(
        point.time_s == pytest.approx(3.0) and point.gain_db == pytest.approx(0.0)
        for point in drums_a.points
    )
    assert any("detected Song A beat" in reason for reason in plan.reasoning)
    assert any("downbeats are not detected" in warning for warning in plan.warnings)


def test_rhythmic_strategy_uses_tempo_fractions_without_enough_beats(synthetic_pair):
    song_a, song_b = synthetic_pair
    analysis_a = make_analysis(
        song_a.mix, tempo_bpm=120.0, beat_times_s=(1.0, 1.5, 2.0)
    )
    analysis_b = make_analysis(song_b.mix, tempo_bpm=120.0)
    plan = plan_transition(
        song_a.mix,
        song_b.mix,
        strategy="rhythmic",
        duration_s=7.0,
        analysis_a=analysis_a,
        analysis_b=analysis_b,
    )
    assert plan.metadata["alignment"] == "tempo_fractions"
    assert plan.metadata["bars"] == 4
    assert any("whole beat" in reason for reason in plan.reasoning)


def test_rhythmic_strategy_falls_back_on_tempo_mismatch(synthetic_pair):
    song_a, song_b = synthetic_pair
    analysis_a, analysis_b = _analyses(song_a, song_b, 120.0, 130.0)
    plan = plan_transition(
        song_a.mix,
        song_b.mix,
        stems_a=song_a.stems,
        stems_b=song_b.stems,
        strategy="rhythmic",
        duration_s=8.0,
        analysis_a=analysis_a,
        analysis_b=analysis_b,
    )
    assert plan.strategy == "conservative"
    assert plan.metadata["fallback"] is True
    assert any("tempo ratio" in warning for warning in plan.warnings)
    assert any("conservative fallback" in warning for warning in plan.warnings)


def test_atmospheric_strategy_extends_overlap_and_adds_sweeps(synthetic_pair):
    song_a, song_b = synthetic_pair
    analysis_a, analysis_b = _analyses(song_a, song_b)
    plan = plan_transition(
        song_a.mix,
        song_b.mix,
        stems_a=song_a.stems,
        stems_b=song_b.stems,
        strategy="atmospheric",
        duration_s=8.0,
        analysis_a=analysis_a,
        analysis_b=analysis_b,
    )
    assert plan.strategy == "atmospheric"
    assert plan.duration_s == pytest.approx(11.2)
    assert plan.start_time_s == pytest.approx(4.8)
    assert plan.metadata["filter_sweep"] is True
    other_a = plan.automation_for("a", "other")
    other_b = plan.automation_for("b", "other")
    assert other_a is not None and other_b is not None
    assert len(other_a.effects) == 1 and len(other_b.effects) == 1
    assert other_a.effects[0].kind == "lowpass_sweep"
    assert other_a.effects[0].start_hz > other_a.effects[0].end_hz
    assert other_b.effects[0].start_hz < other_b.effects[0].end_hz
    assert plan.start_time_s + plan.duration_s == pytest.approx(16.0)
    validate_plan(plan, sample_rate=song_a.mix.sample_rate)


def test_atmospheric_without_stems_falls_back(synthetic_pair):
    song_a, song_b = synthetic_pair
    analysis_a, analysis_b = _analyses(song_a, song_b)
    plan = plan_transition(
        song_a.mix,
        song_b.mix,
        strategy="atmospheric",
        duration_s=8.0,
        analysis_a=analysis_a,
        analysis_b=analysis_b,
    )
    assert plan.strategy == "conservative"
    assert any("without stems" in warning for warning in plan.warnings)


def test_conservative_strategy_is_predictable(synthetic_pair):
    song_a, song_b = synthetic_pair
    analysis_a, analysis_b = _analyses(song_a, song_b)
    plan = plan_transition(
        song_a.mix,
        song_b.mix,
        strategy="conservative",
        duration_s=8.0,
        analysis_a=analysis_a,
        analysis_b=analysis_b,
    )
    assert plan.strategy == "conservative"
    assert len(plan.automations) == 2
    assert all(automation.stem == MIX_STEM for automation in plan.automations)
    assert all(automation.curve == "smoothstep" for automation in plan.automations)
    assert plan.metadata["fallback"] is False


def test_auto_selects_rhythmic_with_usable_tempo(synthetic_pair):
    song_a, song_b = synthetic_pair
    analysis_a, analysis_b = _analyses(song_a, song_b, 120.0, 122.0)
    plan = plan_transition(
        song_a.mix,
        song_b.mix,
        stems_a=song_a.stems,
        stems_b=song_b.stems,
        strategy="auto",
        duration_s=8.0,
        analysis_a=analysis_a,
        analysis_b=analysis_b,
    )
    assert plan.strategy == "rhythmic"
    assert plan.requested_strategy == "auto"


def test_auto_selects_stem_level_when_tempo_is_unsure(synthetic_pair):
    song_a, song_b = synthetic_pair
    analysis_a, analysis_b = _analyses(
        song_a, song_b, 120.0, 130.0, confidence=0.2
    )
    plan = plan_transition(
        song_a.mix,
        song_b.mix,
        stems_a=song_a.stems,
        stems_b=song_b.stems,
        strategy="auto",
        duration_s=8.0,
        analysis_a=analysis_a,
        analysis_b=analysis_b,
    )
    assert plan.strategy == "stem_level"


def test_auto_selects_conservative_without_stems_or_tempo(synthetic_pair):
    song_a, song_b = synthetic_pair
    analysis_a, analysis_b = _analyses(
        song_a, song_b, 120.0, 130.0, confidence=0.2
    )
    plan = plan_transition(
        song_a.mix,
        song_b.mix,
        strategy="auto",
        duration_s=8.0,
        analysis_a=analysis_a,
        analysis_b=analysis_b,
    )
    assert plan.strategy == "conservative"


def test_auto_uses_rhythmic_with_full_mix_when_tempo_is_usable(synthetic_pair):
    song_a, song_b = synthetic_pair
    analysis_a, analysis_b = _analyses(song_a, song_b, 120.0, 120.0)
    plan = plan_transition(
        song_a.mix,
        song_b.mix,
        strategy="auto",
        duration_s=8.0,
        analysis_a=analysis_a,
        analysis_b=analysis_b,
    )
    assert plan.strategy == "rhythmic"
    assert all(automation.stem == MIX_STEM for automation in plan.automations)
    assert any("No stems were supplied" in warning for warning in plan.warnings)


def test_planner_clamps_transition_to_shorter_track(synthetic_pair):
    song_a, song_b = synthetic_pair
    analysis_a, analysis_b = _analyses(song_a, song_b)
    plan = plan_transition(
        song_a.mix,
        song_b.mix,
        strategy="crossfade",
        duration_s=20.0,
        analysis_a=analysis_a,
        analysis_b=analysis_b,
    )
    assert plan.duration_s == pytest.approx(16.0)
    assert any("exceeds the shorter track" in warning for warning in plan.warnings)


def test_planner_raises_minimum_transition_to_configured_floor(synthetic_pair):
    song_a, song_b = synthetic_pair
    analysis_a, analysis_b = _analyses(song_a, song_b)
    plan = plan_transition(
        song_a.mix,
        song_b.mix,
        strategy="crossfade",
        duration_s=0.1,
        analysis_a=analysis_a,
        analysis_b=analysis_b,
    )
    assert plan.duration_s == pytest.approx(2.0)
    assert any("below the configured minimum" in warning for warning in plan.warnings)


@pytest.mark.parametrize("duration", [0.0, -1.0, math.nan, math.inf])
def test_planner_rejects_invalid_duration(synthetic_pair, duration):
    song_a, song_b = synthetic_pair
    analysis_a, analysis_b = _analyses(song_a, song_b)
    with pytest.raises(DCPError, match="positive finite"):
        plan_transition(
            song_a.mix,
            song_b.mix,
            strategy="crossfade",
            duration_s=duration,
            analysis_a=analysis_a,
            analysis_b=analysis_b,
        )


def test_planner_rejects_unknown_strategy(synthetic_pair):
    song_a, song_b = synthetic_pair
    analysis_a, analysis_b = _analyses(song_a, song_b)
    with pytest.raises(DCPError, match="unknown transition strategy"):
        plan_transition(
            song_a.mix,
            song_b.mix,
            strategy="magical",
            duration_s=8.0,
            analysis_a=analysis_a,
            analysis_b=analysis_b,
        )


def test_plan_serializes_to_json(synthetic_pair):
    song_a, song_b = synthetic_pair
    analysis_a, analysis_b = _analyses(song_a, song_b)
    plan = plan_transition(
        song_a.mix,
        song_b.mix,
        stems_a=song_a.stems,
        stems_b=song_b.stems,
        strategy="stem_level",
        duration_s=8.0,
        analysis_a=analysis_a,
        analysis_b=analysis_b,
    )
    payload = plan.to_dict()
    encoded = json.dumps(payload, allow_nan=False)
    decoded = json.loads(encoded)
    assert decoded["schema_version"] == 1
    assert decoded["strategy"] == "stem_level"
    assert len(decoded["automations"]) == 8
    assert decoded["automations"][0]["points"]


def test_studio_remix_strategy_falls_back_without_stems(synthetic_pair):
    song_a, song_b = synthetic_pair
    analysis_a, analysis_b = _analyses(song_a, song_b)
    plan = plan_transition(
        song_a.mix,
        song_b.mix,
        strategy="studio_remix",
        duration_s=8.0,
        analysis_a=analysis_a,
        analysis_b=analysis_b,
    )
    assert plan.strategy == "conservative"
    assert any("Studio Remix requires stems" in w for w in plan.warnings)


def test_studio_remix_strategy_plans_with_stems(synthetic_pair):
    song_a, song_b = synthetic_pair
    downbeats_a = tuple(float(i * 2.0) for i in range(16))
    downbeats_b = tuple(float(i * 2.0) for i in range(16))
    analysis_a = replace(
        make_analysis(song_a.mix, tempo_bpm=120.0, confidence=0.9),
        downbeat_times_s=downbeats_a,
    )
    analysis_b = replace(
        make_analysis(song_b.mix, tempo_bpm=120.0, confidence=0.9),
        downbeat_times_s=downbeats_b,
    )
    plan = plan_transition(
        song_a.mix,
        song_b.mix,
        stems_a=song_a.stems,
        stems_b=song_b.stems,
        strategy="studio_remix",
        duration_s=8.0,
        analysis_a=analysis_a,
        analysis_b=analysis_b,
    )
    assert plan.strategy == "studio_remix"
    assert plan.metadata.get("studio_remix") is True
    assert "launchpad_a_s" in plan.metadata
    assert "impact_b_s" in plan.metadata
    assert len(plan.automations) == 8
    validate_plan(plan, sample_rate=song_a.mix.sample_rate)


def test_studio_remix_strategy_renders_safely(synthetic_pair):
    import numpy as np
    from dcp.rendering import render_plan

    song_a, song_b = synthetic_pair
    downbeats_a = tuple(float(i * 2.0) for i in range(16))
    downbeats_b = tuple(float(i * 2.0) for i in range(16))
    analysis_a = replace(
        make_analysis(song_a.mix, tempo_bpm=120.0, confidence=0.9),
        downbeat_times_s=downbeats_a,
    )
    analysis_b = replace(
        make_analysis(song_b.mix, tempo_bpm=120.0, confidence=0.9),
        downbeat_times_s=downbeats_b,
    )
    plan = plan_transition(
        song_a.mix,
        song_b.mix,
        stems_a=song_a.stems,
        stems_b=song_b.stems,
        strategy="studio_remix",
        duration_s=8.0,
        analysis_a=analysis_a,
        analysis_b=analysis_b,
    )
    rendered = render_plan(
        song_a.mix,
        song_b.mix,
        plan,
        stems_a=song_a.stems,
        stems_b=song_b.stems,
    )
    assert rendered.audio.channels == 2
    assert rendered.audio.frames > 0
    assert np.all(np.isfinite(rendered.audio.samples))
    assert np.max(np.abs(rendered.audio.samples)) <= 1.0



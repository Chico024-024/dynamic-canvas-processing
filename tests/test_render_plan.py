"""Plan rendering: coverage rules, independent gains, duration, and safety."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_analysis, make_constant, make_stems, make_tone
from dcp.models import MIX_STEM, RenderError, StemSet
from dcp.rendering import render_plan, rendered_duration_s
from dcp.transitions import (
    EffectSpec,
    GainPoint,
    StemAutomation,
    TransitionPlan,
    TransitionPlanner,
)


def _plan(
    automations: tuple[StemAutomation, ...],
    *,
    duration_s: float = 0.5,
    start_time_s: float = 1.5,
    strategy: str = "test",
) -> TransitionPlan:
    return TransitionPlan(
        strategy=strategy,
        requested_strategy=strategy,
        duration_s=duration_s,
        start_time_s=start_time_s,
        end_time_s=start_time_s + duration_s,
        automations=automations,
    )


def _hold(source: str, stem: str, gain_db: float, duration_s: float = 0.5) -> StemAutomation:
    return StemAutomation(
        source=source,
        stem=stem,
        points=(
            GainPoint(0.0, gain_db),
            GainPoint(duration_s, gain_db),
        ),
    )


def test_stem_render_is_independent_for_each_stem():
    song_a = make_tone(duration_s=2.0)
    song_b = make_tone(duration_s=2.0)
    stems_a = make_stems(duration_s=2.0)
    plan = _plan(
        (
            _hold("a", "vocals", 0.0),
            _hold("a", "drums", -120.0),
            _hold("a", "bass", -120.0),
            _hold("a", "other", -120.0),
            _hold("b", MIX_STEM, -120.0),
        )
    )
    result = render_plan(song_a, song_b, plan, stems_a=stems_a)
    vocals = stems_a["vocals"].samples

    assert result.audio.duration_s == pytest.approx(3.5)
    assert np.allclose(result.audio.samples[:, : vocals.shape[1]], vocals, atol=1e-12)
    assert np.allclose(result.audio.samples[:, vocals.shape[1] :], 0.0, atol=1e-12)
    assert any("Stem-level rendering" in note for note in result.notes)


def test_stem_render_requires_supplied_stems():
    song_a = make_tone(duration_s=2.0)
    song_b = make_tone(duration_s=2.0)
    plan = _plan(
        (
            _hold("a", "vocals", 0.0),
            _hold("b", MIX_STEM, -120.0),
        )
    )
    with pytest.raises(RenderError, match="but no stems were provided"):
        render_plan(song_a, song_b, plan)


def test_stem_render_requires_full_coverage_of_provided_stems():
    song_a = make_tone(duration_s=2.0)
    song_b = make_tone(duration_s=2.0)
    stems_a = make_stems(duration_s=2.0)
    plan = _plan(
        (
            _hold("a", "vocals", 0.0),
            _hold("b", MIX_STEM, -120.0),
        )
    )
    with pytest.raises(RenderError, match="does not cover provided stem"):
        render_plan(song_a, song_b, plan, stems_a=stems_a)


def test_stem_render_reports_unknown_stem():
    song_a = make_tone(duration_s=2.0)
    song_b = make_tone(duration_s=2.0)
    stems_a = StemSet({"vocals": make_tone(duration_s=2.0)})
    plan = _plan(
        (
            _hold("a", "vocals", -120.0),
            _hold("a", "guitar", -120.0),
            _hold("b", MIX_STEM, -120.0),
        )
    )
    with pytest.raises(RenderError, match="only \\['vocals'\\]"):
        render_plan(song_a, song_b, plan, stems_a=stems_a)


def test_full_mix_plan_ignores_supplied_stems_with_note():
    song_a = make_tone(duration_s=2.0, amplitude=0.2)
    song_b = make_tone(
        duration_s=2.0, frequency=550.0, amplitude=0.2, phase=0.4
    )
    stems_a = make_stems(duration_s=2.0)
    stems_b = make_stems(duration_s=2.0)
    analyses = (make_analysis(song_a), make_analysis(song_b))
    plan = TransitionPlanner().plan(
        song_a,
        song_b,
        stems_a=stems_a,
        stems_b=stems_b,
        strategy="crossfade",
        duration_s=0.5,
        analysis_a=analyses[0],
        analysis_b=analyses[1],
    )
    with_stems = render_plan(
        song_a, song_b, plan, stems_a=stems_a, stems_b=stems_b
    )
    without_stems = render_plan(song_a, song_b, plan)

    assert np.allclose(
        with_stems.audio.samples, without_stems.audio.samples, atol=1e-12
    )
    assert any("stems were provided" in note for note in with_stems.notes)


def test_render_uses_both_sources_and_expected_duration():
    song_a = make_constant(0.2, duration_s=2.0)
    song_b = make_constant(0.1, duration_s=3.0)
    plan = _plan(
        (
            _hold("a", MIX_STEM, 0.0),
            _hold("b", MIX_STEM, 0.0),
        )
    )
    result = render_plan(song_a, song_b, plan)
    assert rendered_duration_s(song_a, song_b, plan) == pytest.approx(4.5)
    assert result.audio.duration_s == pytest.approx(4.5)
    assert result.audio.samples[0, 0] == pytest.approx(0.2)
    assert result.audio.samples[0, -1] == pytest.approx(0.1)


def test_render_prevents_clipping_with_master_gain():
    song_a = make_constant(0.8, duration_s=2.0)
    song_b = make_constant(0.8, duration_s=2.0)
    plan = _plan(
        (
            _hold("a", MIX_STEM, 0.0),
            _hold("b", MIX_STEM, 0.0),
        )
    )
    result = render_plan(song_a, song_b, plan)
    assert result.audio.peak <= 0.999 + 1e-9
    assert result.master_gain_db < 0
    assert any("prevent clipping" in note for note in result.notes)


def test_render_rejects_transition_past_song_a_end():
    song_a = make_tone(duration_s=2.0)
    song_b = make_tone(duration_s=2.0)
    plan = _plan(
        (
            _hold("a", MIX_STEM, 0.0),
            _hold("b", MIX_STEM, 0.0),
        ),
        duration_s=1.0,
        start_time_s=1.5,
    )
    with pytest.raises(RenderError, match="after Song A ends"):
        render_plan(song_a, song_b, plan)


def test_render_rejects_start_after_song_a_end():
    song_a = make_tone(duration_s=2.0)
    song_b = make_tone(duration_s=2.0)
    plan = _plan(
        (
            _hold("a", MIX_STEM, 0.0),
            _hold("b", MIX_STEM, 0.0),
        ),
        duration_s=1.0,
        start_time_s=2.5,
    )
    with pytest.raises(RenderError, match="starts at 2.5"):
        render_plan(song_a, song_b, plan)


def test_render_adapts_stem_format_with_notes():
    song_a = make_tone(duration_s=2.0, sample_rate=22050)
    song_b = make_tone(duration_s=2.0, sample_rate=22050)
    stems_a = StemSet(
        {"vocals": make_tone(duration_s=2.0, sample_rate=11025, channels=1)}
    )
    plan = _plan(
        (
            _hold("a", "vocals", 0.0),
            _hold("b", MIX_STEM, -120.0),
        )
    )
    result = render_plan(song_a, song_b, plan, stems_a=stems_a)
    assert result.audio.sample_rate == 22050
    assert result.audio.channels == 2
    assert any("resampled" in note for note in result.notes)
    assert any("channel" in note for note in result.notes)


def test_cue_offset_changes_output_duration():
    song_a = make_constant(0.2, duration_s=2.0)
    song_b = make_constant(0.1, duration_s=5.0)

    # Without cue offset: duration = max(2.0, 1.0 + 5.0) = 6.0s
    plan_no_cue = TransitionPlan(
        strategy="test",
        requested_strategy="test",
        duration_s=1.0,
        start_time_s=1.0,
        end_time_s=2.0,
        automations=(
            _hold("a", MIX_STEM, 0.0, duration_s=1.0),
            _hold("b", MIX_STEM, 0.0, duration_s=1.0),
        ),
    )
    assert rendered_duration_s(song_a, song_b, plan_no_cue) == pytest.approx(6.0)
    result_no_cue = render_plan(song_a, song_b, plan_no_cue)
    assert result_no_cue.audio.duration_s == pytest.approx(6.0)

    # With cue offset = 2.0s: duration = max(2.0, 1.0 + (5.0 - 2.0)) = 4.0s
    plan_cued = TransitionPlan(
        strategy="test",
        requested_strategy="test",
        duration_s=1.0,
        start_time_s=1.0,
        end_time_s=2.0,
        automations=(
            _hold("a", MIX_STEM, 0.0, duration_s=1.0),
            _hold("b", MIX_STEM, 0.0, duration_s=1.0),
        ),
        metadata={"cue_offset_b_s": 2.0},
    )
    assert rendered_duration_s(song_a, song_b, plan_cued) == pytest.approx(4.0)
    result_cued = render_plan(song_a, song_b, plan_cued)
    assert result_cued.audio.duration_s == pytest.approx(4.0)


def test_automation_timing_and_source_positioning():
    sr = 22050
    dur_a = 4.0
    dur_b = 6.0
    song_a = make_constant(0.0, duration_s=dur_a, sample_rate=sr)

    # Song B is a normalized time ramp peaking at 0.5 (below 0.999 ceiling to avoid limiter scaling)
    t_b = 0.5 * (np.arange(int(dur_b * sr), dtype=np.float64) / sr) / dur_b
    from dcp.models import AudioBuffer
    song_b = AudioBuffer(np.tile(t_b, (2, 1)), sample_rate=sr)

    cue_offset_s = 2.0
    start_time_s = 2.5
    trans_dur_s = 1.0

    plan = TransitionPlan(
        strategy="test",
        requested_strategy="test",
        duration_s=trans_dur_s,
        start_time_s=start_time_s,
        end_time_s=start_time_s + trans_dur_s,
        automations=(
            _hold("a", MIX_STEM, -120.0, duration_s=trans_dur_s),
            # Unity gain for Song B across the transition
            _hold("b", MIX_STEM, 0.0, duration_s=trans_dur_s),
        ),
        metadata={"cue_offset_b_s": cue_offset_s},
    )

    result = render_plan(song_a, song_b, plan)
    out_samples = result.audio.samples[0]

    # At output time = start_time_s, Song B should begin reading at cue_offset_s
    entry_frame = int(round(start_time_s * sr))
    expected_entry_val = 0.5 * (cue_offset_s / dur_b)
    assert out_samples[entry_frame] == pytest.approx(expected_entry_val, abs=1e-5)

    # At 0.5s into the transition, Song B should be at cue_offset_s + 0.5s
    mid_frame = int(round((start_time_s + 0.5) * sr))
    expected_mid_val = 0.5 * ((cue_offset_s + 0.5) / dur_b)
    assert out_samples[mid_frame] == pytest.approx(expected_mid_val, abs=1e-5)


def test_effect_timing_lowpass_sweep_bounds():
    song_a = make_tone(duration_s=3.0)
    song_b = make_tone(duration_s=4.0)

    sweep_effect = EffectSpec(
        kind="lowpass_sweep",
        start_hz=8000.0,
        end_hz=400.0,
        start_s=0.2,
        end_s=0.8,
    )
    plan = TransitionPlan(
        strategy="test",
        requested_strategy="test",
        duration_s=1.0,
        start_time_s=2.0,
        end_time_s=3.0,
        automations=(
            _hold("a", MIX_STEM, 0.0, duration_s=1.0),
            StemAutomation(
                source="b",
                stem=MIX_STEM,
                points=(GainPoint(0.0, 0.0), GainPoint(1.0, 0.0)),
                effects=(sweep_effect,),
            ),
        ),
        metadata={"cue_offset_b_s": 1.0},
    )

    result = render_plan(song_a, song_b, plan)
    assert result.audio.duration_s == pytest.approx(5.0)
    assert np.all(np.isfinite(result.audio.samples))


def test_render_rejects_cue_offset_beyond_track_duration():
    song_a = make_tone(duration_s=2.0)
    song_b = make_tone(duration_s=3.0)
    plan = TransitionPlan(
        strategy="test",
        requested_strategy="test",
        duration_s=1.0,
        start_time_s=1.0,
        end_time_s=2.0,
        automations=(
            _hold("a", MIX_STEM, 0.0, duration_s=1.0),
            _hold("b", MIX_STEM, 0.0, duration_s=1.0),
        ),
        metadata={"cue_offset_b_s": 3.5},
    )
    with pytest.raises(RenderError, match="is beyond track duration"):
        render_plan(song_a, song_b, plan)


def test_render_rejects_insufficient_remaining_audio_for_transition():
    song_a = make_tone(duration_s=2.0)
    song_b = make_tone(duration_s=3.0)
    # Remaining = 3.0 - 2.5 = 0.5s, but transition requires 1.0s
    plan = TransitionPlan(
        strategy="test",
        requested_strategy="test",
        duration_s=1.0,
        start_time_s=1.0,
        end_time_s=2.0,
        automations=(
            _hold("a", MIX_STEM, 0.0, duration_s=1.0),
            _hold("b", MIX_STEM, 0.0, duration_s=1.0),
        ),
        metadata={"cue_offset_b_s": 2.5},
    )
    with pytest.raises(RenderError, match="cannot cover the planned transition duration"):
        render_plan(song_a, song_b, plan)


def test_render_rejects_negative_cue_offset():
    song_a = make_tone(duration_s=2.0)
    song_b = make_tone(duration_s=3.0)
    plan = TransitionPlan(
        strategy="test",
        requested_strategy="test",
        duration_s=1.0,
        start_time_s=1.0,
        end_time_s=2.0,
        automations=(
            _hold("a", MIX_STEM, 0.0, duration_s=1.0),
            _hold("b", MIX_STEM, 0.0, duration_s=1.0),
        ),
        metadata={"cue_offset_b_s": -0.5},
    )
    with pytest.raises(RenderError, match="cannot be negative"):
        render_plan(song_a, song_b, plan)


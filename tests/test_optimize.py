"""Tests for the RATO objective and the objective-based optimizer."""

from __future__ import annotations

from dataclasses import replace

import pytest

from conftest import make_analysis
from dcp.config import DEFAULT_CONFIG, OptimizationConfig
from dcp.transitions import (
    CandidateFeatures,
    QualityScoringError,
    TransitionContext,
    evaluate_candidate,
    generate_candidate_plans,
    optimize_transition,
    plan_transition,
)
from dcp.transitions.optimize import (
    complexity_loss,
    continuity_loss,
    distortion_loss,
    musical_loss,
    objective,
    rhythm_loss,
    vocal_overlap_loss,
)
from dcp.transitions.plan import validate_plan


def _stem_plan(synthetic_pair):
    song_a, song_b = synthetic_pair
    return plan_transition(
        song_a.mix,
        song_b.mix,
        stems_a=song_a.stems,
        stems_b=song_b.stems,
        strategy="stem_level",
        duration_s=8.0,
    )


def _context(synthetic_pair, duration_s: float = 4.0) -> TransitionContext:
    song_a, song_b = synthetic_pair
    return TransitionContext(
        song_a=song_a.mix,
        song_b=song_b.mix,
        stems_a=song_a.stems,
        stems_b=song_b.stems,
        analysis_a=make_analysis(song_a.mix),
        analysis_b=make_analysis(song_b.mix),
        requested_duration_s=duration_s,
        requested_strategy="optimized",
        config=DEFAULT_CONFIG,
    )


def test_distortion_loss_zero_at_unity_and_saturates():
    assert distortion_loss(0.0) == 0.0
    assert distortion_loss(6.0) == 0.0  # a positive master gain is not distortion
    assert distortion_loss(-6.0) == pytest.approx(1.0)
    assert distortion_loss(-12.0) == pytest.approx(1.0)  # clipped at 1.0


def test_musical_loss_is_inert_placeholder(synthetic_pair):
    plan = _stem_plan(synthetic_pair)
    assert musical_loss(plan) == 0.0


def test_objective_combines_weighted_terms():
    config = replace(
        DEFAULT_CONFIG,
        optimization=OptimizationConfig(w_loudness=2.0, w_vocal=0.5),
    )
    losses = {
        "loudness": 0.5,
        "continuity": 0.0,
        "musical": 0.0,
        "vocal": 0.4,
        "rhythm": 0.0,
        "distortion": 0.0,
        "complexity": 0.0,
    }
    assert objective(losses, config) == pytest.approx(2.0 * 0.5 + 0.5 * 0.4)


def test_plan_level_losses_are_normalized(synthetic_pair):
    plan = _stem_plan(synthetic_pair)
    values = (
        continuity_loss(plan),
        vocal_overlap_loss(plan),
        rhythm_loss(plan, tempo_bpm=120.0),
        complexity_loss(plan),
    )
    for value in values:
        assert 0.0 <= value <= 1.0


def test_rhythm_loss_zero_when_tempo_unknown(synthetic_pair):
    plan = _stem_plan(synthetic_pair)
    assert rhythm_loss(plan, tempo_bpm=None) == 0.0


def test_optimize_transition_returns_valid_plan(synthetic_pair):
    song_a, song_b = synthetic_pair
    plan = optimize_transition(
        song_a.mix,
        song_b.mix,
        stems_a=song_a.stems,
        stems_b=song_b.stems,
        duration_s=8.0,
    )
    assert plan.strategy == "optimized"
    validate_plan(plan, sample_rate=song_a.mix.sample_rate)

    assert "candidate" in plan.metadata
    assert "objective" in plan.metadata
    assert "losses" in plan.metadata
    assert "weights" in plan.metadata
    assert plan.metadata["scorer"] == "weighted_loss"
    assert plan.metadata["weights"]["vocal"] == 1.0
    assert plan.metadata["candidates_evaluated"] >= 1
    assert (
        plan.metadata["candidates_generated"]
        >= plan.metadata["candidates_evaluated"]
    )
    for name, value in plan.metadata["losses"].items():
        assert 0.0 <= value <= 1.0, name


def test_optimize_transition_falls_back_without_stems(synthetic_pair):
    song_a, song_b = synthetic_pair
    plan = optimize_transition(song_a.mix, song_b.mix, duration_s=8.0)
    assert plan.strategy == "conservative"


def test_generate_candidate_plans_are_valid_and_labeled(synthetic_pair):
    context = _context(synthetic_pair)
    plans = generate_candidate_plans(context)
    assert [plan.metadata["candidate"] for plan in plans] == [
        "default",
        "wide_vocal_handoff",
        "early_drums_exit",
        "late_bass_entry",
        "atmosphere_blend",
    ]
    for plan in plans:
        validate_plan(plan, sample_rate=context.song_a.sample_rate)
        assert plan.strategy == "optimized"


def test_generate_candidate_plans_empty_without_stems(synthetic_pair):
    context = replace(_context(synthetic_pair), stems_a=None, stems_b=None)
    assert generate_candidate_plans(context) == []


def test_default_scorer_matches_the_weighted_objective(synthetic_pair):
    context = _context(synthetic_pair)
    plan = generate_candidate_plans(context)[0]
    evaluation = evaluate_candidate(context, plan)
    assert isinstance(evaluation.features, CandidateFeatures)
    assert evaluation.features.candidate == "default"
    assert evaluation.features.automation_count == len(plan.automations)
    assert evaluation.score == pytest.approx(
        objective(evaluation.features.losses)
    )


def test_optimize_transition_accepts_a_custom_scorer(synthetic_pair):
    class PreferLateBass:
        name = "prefer_late_bass"

        def score(self, features):
            return 0.0 if features.candidate == "late_bass_entry" else 1.0

    song_a, song_b = synthetic_pair
    plan = optimize_transition(
        song_a.mix,
        song_b.mix,
        stems_a=song_a.stems,
        stems_b=song_b.stems,
        duration_s=4.0,
        scorer=PreferLateBass(),
    )
    assert plan.metadata["candidate"] == "late_bass_entry"
    assert plan.metadata["scorer"] == "prefer_late_bass"
    assert plan.metadata["weights"] is None
    assert plan.metadata["objective"] == 0.0
    assert plan.metadata["candidates_evaluated"] == 5
    assert "prefer_late_bass" in " ".join(plan.reasoning)


def test_scorer_contract_is_validated_before_generation(synthetic_pair):
    class Unnamed:
        def score(self, features):
            return 0.0

    song_a, song_b = synthetic_pair
    with pytest.raises(QualityScoringError, match="non-empty string 'name'"):
        optimize_transition(
            song_a.mix,
            song_b.mix,
            stems_a=song_a.stems,
            stems_b=song_b.stems,
            duration_s=2.0,
            scorer=Unnamed(),
        )


def test_non_finite_scorer_output_is_rejected(synthetic_pair):
    class Broken:
        name = "broken"

        def score(self, features):
            return float("nan")

    song_a, song_b = synthetic_pair
    with pytest.raises(QualityScoringError, match="non-finite"):
        optimize_transition(
            song_a.mix,
            song_b.mix,
            stems_a=song_a.stems,
            stems_b=song_b.stems,
            duration_s=2.0,
            scorer=Broken(),
        )

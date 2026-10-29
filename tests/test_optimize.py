"""Tests for the RATO objective and the objective-based optimizer."""

from __future__ import annotations

from dataclasses import replace

import pytest

from dcp.config import DEFAULT_CONFIG, OptimizationConfig
from dcp.transitions import optimize_transition, plan_transition
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
    assert plan.metadata["candidates_evaluated"] >= 1
    for name, value in plan.metadata["losses"].items():
        assert 0.0 <= value <= 1.0, name


def test_optimize_transition_falls_back_without_stems(synthetic_pair):
    song_a, song_b = synthetic_pair
    plan = optimize_transition(song_a.mix, song_b.mix, duration_s=8.0)
    assert plan.strategy == "conservative"

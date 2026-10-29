"""Structural validation of transition plans."""

from __future__ import annotations

from dataclasses import replace

import pytest

from dcp.models import MIX_STEM, PlanValidationError
from dcp.transitions import (
    EffectSpec,
    GainPoint,
    StemAutomation,
    TransitionPlan,
    validate_plan,
)


def _automation(
    source: str = "a",
    stem: str = MIX_STEM,
    *,
    points: tuple[GainPoint, ...] | None = None,
    curve: str = "smoothstep",
    effects: tuple[EffectSpec, ...] = (),
) -> StemAutomation:
    return StemAutomation(
        source=source,
        stem=stem,
        points=points
        or (GainPoint(0.0, 0.0), GainPoint(2.0, -120.0)),
        curve=curve,
        effects=effects,
    )


def _base_plan() -> TransitionPlan:
    return TransitionPlan(
        strategy="conservative",
        requested_strategy="conservative",
        duration_s=2.0,
        start_time_s=3.0,
        end_time_s=5.0,
        automations=(
            _automation("a"),
            _automation(
                "b",
                points=(GainPoint(0.0, -120.0), GainPoint(2.0, 0.0)),
            ),
        ),
        reasoning=("test",),
    )


def test_valid_plan_passes():
    validate_plan(_base_plan())


def test_empty_strategy_is_rejected():
    plan = replace(_base_plan(), strategy="")
    with pytest.raises(PlanValidationError, match="strategy"):
        validate_plan(plan)


def test_duration_must_be_positive_and_finite():
    with pytest.raises(PlanValidationError, match="duration_s must be a positive"):
        validate_plan(replace(_base_plan(), duration_s=0.0))
    with pytest.raises(PlanValidationError, match="duration_s must be a positive"):
        validate_plan(replace(_base_plan(), duration_s=float("nan")))


def test_end_time_must_match_start_plus_duration():
    plan = replace(_base_plan(), end_time_s=5.5)
    with pytest.raises(PlanValidationError, match="end_time_s must equal"):
        validate_plan(plan)


def test_negative_start_is_rejected():
    plan = replace(_base_plan(), start_time_s=-1.0, end_time_s=1.0)
    with pytest.raises(PlanValidationError, match="non-negative"):
        validate_plan(plan)


def test_plan_requires_automation_for_both_sources():
    plan = replace(_base_plan(), automations=(_automation("a"),))
    with pytest.raises(PlanValidationError, match="source 'b'"):
        validate_plan(plan)

    plan = replace(_base_plan(), automations=())
    with pytest.raises(PlanValidationError, match="at least one stem automation"):
        validate_plan(plan)


def test_duplicate_automation_is_rejected():
    plan = replace(
        _base_plan(), automations=(_automation("a"), _automation("a"))
    )
    with pytest.raises(PlanValidationError, match="duplicate automation"):
        validate_plan(plan)


def test_invalid_source_is_rejected():
    plan = replace(
        _base_plan(),
        automations=(
            _automation("c"),
            _automation(
                "b",
                points=(GainPoint(0.0, -120.0), GainPoint(2.0, 0.0)),
            ),
        ),
    )
    with pytest.raises(PlanValidationError, match="source must be 'a' or 'b'"):
        validate_plan(plan)


def test_gain_points_must_be_at_least_two_and_ordered():
    single = _automation(points=(GainPoint(0.0, 0.0),))
    with pytest.raises(PlanValidationError, match="at least two"):
        validate_plan(replace(_base_plan(), automations=(single, _automation("b"))))

    repeated = _automation(
        points=(GainPoint(0.0, 0.0), GainPoint(1.0, -6.0), GainPoint(1.0, -12.0))
    )
    with pytest.raises(PlanValidationError, match="strictly increasing"):
        validate_plan(
            replace(_base_plan(), automations=(repeated, _automation("b")))
        )


def test_gain_points_must_stay_inside_window():
    outside = _automation(
        points=(GainPoint(0.0, 0.0), GainPoint(2.5, -120.0))
    )
    with pytest.raises(PlanValidationError, match="outside the transition window"):
        validate_plan(
            replace(_base_plan(), automations=(outside, _automation("b")))
        )


def test_gain_limits_are_enforced():
    too_loud = _automation(
        points=(GainPoint(0.0, 10.0), GainPoint(2.0, -120.0))
    )
    with pytest.raises(PlanValidationError, match="exceeds the maximum"):
        validate_plan(
            replace(_base_plan(), automations=(too_loud, _automation("b")))
        )

    too_quiet = _automation(
        points=(GainPoint(0.0, -130.0), GainPoint(2.0, -120.0))
    )
    with pytest.raises(PlanValidationError, match="below the floor"):
        validate_plan(
            replace(_base_plan(), automations=(too_quiet, _automation("b")))
        )


def test_non_finite_gain_is_rejected():
    broken = _automation(
        points=(GainPoint(0.0, float("nan")), GainPoint(2.0, -120.0))
    )
    with pytest.raises(PlanValidationError, match="must be finite"):
        validate_plan(
            replace(_base_plan(), automations=(broken, _automation("b")))
        )


def test_unknown_curve_is_rejected():
    broken = _automation(curve="exponential")
    with pytest.raises(PlanValidationError, match="unsupported curve"):
        validate_plan(
            replace(_base_plan(), automations=(broken, _automation("b")))
        )


def test_effect_fields_are_validated():
    plan = _base_plan()
    bad_kind = replace(
        plan,
        automations=(
            _automation(effects=(EffectSpec(kind="reverb"),)),
            _automation("b"),
        ),
    )
    with pytest.raises(PlanValidationError, match="unsupported effect kind"):
        validate_plan(bad_kind)

    bad_cutoff = replace(
        plan,
        automations=(
            _automation(effects=(EffectSpec(start_hz=10.0, end_hz=1000.0),)),
            _automation("b"),
        ),
    )
    with pytest.raises(PlanValidationError, match="greater than 20"):
        validate_plan(bad_cutoff)

    late_effect = replace(
        plan,
        automations=(
            _automation(effects=(EffectSpec(start_s=1.0, end_s=3.0),)),
            _automation("b"),
        ),
    )
    with pytest.raises(PlanValidationError, match="within the transition"):
        validate_plan(late_effect)

    reversed_effect = replace(
        plan,
        automations=(
            _automation(effects=(EffectSpec(start_s=1.5, end_s=0.5),)),
            _automation("b"),
        ),
    )
    with pytest.raises(PlanValidationError, match="after start_s"):
        validate_plan(reversed_effect)


def test_effect_cutoff_must_stay_below_nyquist():
    plan = replace(
        _base_plan(),
        automations=(
            _automation(effects=(EffectSpec(start_hz=16000.0, end_hz=800.0),)),
            _automation("b"),
        ),
    )
    with pytest.raises(PlanValidationError, match="Nyquist"):
        validate_plan(plan, sample_rate=22050)

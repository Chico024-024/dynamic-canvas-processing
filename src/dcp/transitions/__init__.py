"""Transition planning: models, curves, strategies, and orchestration."""

from .context import TransitionContext
from .curves import db_to_gain, equal_power_fades, gain_to_db, interpolate_gain
from .plan import (
    EffectSpec,
    GainPoint,
    StemAutomation,
    TransitionPlan,
    validate_plan,
)
from .planner import TransitionPlanner, plan_transition
from .optimize import evaluate_losses, objective, optimize_transition
from .strategies import (
    StrategyRegistry,
    TransitionStrategy,
    build_core_registry,
    tempo_is_usable,
)

__all__ = [
    "EffectSpec",
    "GainPoint",
    "StemAutomation",
    "StrategyRegistry",
    "TransitionContext",
    "TransitionPlan",
    "TransitionPlanner",
    "TransitionStrategy",
    "build_core_registry",
    "db_to_gain",
    "equal_power_fades",
    "evaluate_losses",
    "gain_to_db",
    "interpolate_gain",
    "objective",
    "optimize_transition",
    "plan_transition",
    "tempo_is_usable",
    "validate_plan",
]

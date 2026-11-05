"""Transition planning: models, curves, strategies, and orchestration."""

from .context import TransitionContext
from .curves import crossfade_envelopes, db_to_gain, equal_power_fades, gain_to_db, interpolate_gain
from .plan import (
    EffectSpec,
    GainPoint,
    StemAutomation,
    TransitionPlan,
    TransitionParams,
    validate_plan,
)
from .planner import TransitionPlanner, plan_transition
from .optimize import (
    CandidateEvaluation,
    WeightedLossScorer,
    candidate_features,
    evaluate_candidate,
    evaluate_losses,
    generate_candidate_plans,
    objective,
    optimize_transition,
    optimize_transition_from_context,
)
from .quality import CandidateFeatures, QualityScorer, QualityScoringError
from .strategies import (
    StrategyRegistry,
    TransitionStrategy,
    build_core_registry,
    tempo_is_usable,
)
from .alignment import (
    PhaseLockResult,
    calculate_phase_locked_cue_offset,
    apply_phase_lock_to_context,
)
from .cache import (
    CacheConfig,
    CacheEntry,
    TransitionCache,
    build_cache_key,
    get_cached_transition,
    ENGINE_VERSION,
)
from .policy_network import (
    PolicyNetwork,
    PolicyFeatures,
    NeuralPlanStrategy,
    extract_policy_features,
    policy_features_to_tensor,
    neural_policy_available,
)

__all__ = [
    "CacheConfig",
    "CacheEntry",
    "CandidateEvaluation",
    "CandidateFeatures",
    "EffectSpec",
    "ENGINE_VERSION",
    "GainPoint",
    "PhaseLockResult",
    "PolicyFeatures",
    "PolicyNetwork",
    "QualityScorer",
    "QualityScoringError",
    "StemAutomation",
    "StrategyRegistry",
    "TransitionCache",
    "TransitionContext",
    "TransitionPlan",
    "TransitionParams",
    "TransitionPlanner",
    "TransitionStrategy",
    "WeightedLossScorer",
    "apply_phase_lock_to_context",
    "build_cache_key",
    "build_core_registry",
    "calculate_phase_locked_cue_offset",
    "candidate_features",
    "crossfade_envelopes",
    "db_to_gain",
    "equal_power_fades",
    "evaluate_candidate",
    "evaluate_losses",
    "gain_to_db",
    "generate_candidate_plans",
    "get_cached_transition",
    "interpolate_gain",
    "neural_policy_available",
    "objective",
    "optimize_transition",
    "optimize_transition_from_context",
    "plan_transition",
    "tempo_is_usable",
    "validate_plan",
]

"""Role-aware transition optimization (RATO).

This module implements the planner side of DCP's objective-based design. The
signal-generation equation (the sum over sources and roles of per-role gain
automations with optional effects) is already implemented in
:mod:`dcp.rendering.stem_mix`; this module adds the ``argmin J(theta)`` search
that *decides* those gains and timings.

The objective ``J`` is a weighted sum of seven normalized loss terms, each
roughly in ``[0, 1]`` so no single term dominates by scale. Weights live in
:class:`dcp.config.OptimizationConfig`.

Loss terms
----------

* ``loudness``     - short-window RMS fluctuation across the transition.
* ``continuity``   - abruptness of the linear gain curves (max second difference).
* ``musical``      - placeholder (``0.0``); needs phrase/section analysis, which
                     is out of scope for Core.
* ``vocal``        - amount of vocal overlap when a clean handoff is expected.
* ``rhythm``       - mean distance of fade boundaries from the beat grid.
* ``distortion``   - how much master gain the renderer had to apply to avoid
                     clipping (a proxy for how hard the transition is pushed).
* ``complexity``   - number of automation points and effects.

These are objective engineering proxies. They cannot establish musical quality;
listening tests remain required (see the README). The weights are hand-tuned
starting points, intentionally not claimed to be optimal.

Candidate ranking is delegated to a pluggable :class:`QualityScorer`. Core's
default is :class:`WeightedLossScorer`, which is exactly the hand-tuned weighted
sum above. Supplying a different scorer changes only how candidates are ranked;
it does not imply that any perceptual model exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Mapping

import numpy as np

from ..config import DCPConfig, DEFAULT_CONFIG
from ..models import MIX_STEM, DCPError, PlanValidationError
from .context import TransitionContext
from .curves import interpolate_gain
from .plan import StemAutomation, TransitionPlan, validate_plan
from .quality import CandidateFeatures, QualityScorer, QualityScoringError
from .strategies import (
    _A_WINDOWS,
    _B_WINDOWS,
    _clamp_requested_duration,
    _conservative_plan,
    _fade_points,
    _make_plan,
    _ordered_names,
    _views,
)

if TYPE_CHECKING:
    from ..rendering.common import RenderResult

_FLOOR_DB = -120.0

# Loss-term names, in canonical order, used by evaluate_losses and objective.
LOSS_NAMES: tuple[str, ...] = (
    "loudness",
    "continuity",
    "musical",
    "vocal",
    "rhythm",
    "distortion",
    "complexity",
)


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def _default_window(stem: str, source: str) -> tuple[float, float]:
    if stem == MIX_STEM:
        return (0.0, 0.85) if source == "a" else (0.1, 1.0)
    return (0.05, 0.85) if source == "a" else (0.15, 0.95)


def _candidate_specs() -> list[
    tuple[str, dict[str, tuple[float, float]], dict[str, tuple[float, float]]]
]:
    """Return a small, hand-designed set of role-timing candidates.

    Each candidate is ``(label, windows_a, windows_b)`` where the window dicts
    map a stem name to its ``(start_fraction, end_fraction)`` of the transition
    duration. Only the musically-meaningful axes are varied; the default
    candidate is the ``stem_level`` timing so the optimizer is never worse than
    the un-optimized strategy.
    """

    base_a = {k: v for k, v in _A_WINDOWS.items()}
    base_b = {k: v for k, v in _B_WINDOWS.items()}
    specs: list[
        tuple[str, dict[str, tuple[float, float]], dict[str, tuple[float, float]]]
    ] = [("default", base_a, base_b)]

    # Cleaner vocal handoff: outgoing vocals resolve earlier, incoming vocals
    # enter later.
    a, b = dict(base_a), dict(base_b)
    a["vocals"] = (a["vocals"][0], 0.40)
    b["vocals"] = (0.62, b["vocals"][1])
    specs.append(("wide_vocal_handoff", a, b))

    # Outgoing drums exit earlier, near a compatible rhythmic boundary.
    a, b = dict(base_a), dict(base_b)
    a["drums"] = (a["drums"][0], 0.55)
    specs.append(("early_drums_exit", a, b))

    # Incoming bass waits a little longer to reduce low-end overlap.
    a, b = dict(base_a), dict(base_b)
    b["bass"] = (0.20, b["bass"][1])
    specs.append(("late_bass_entry", a, b))

    # Atmospheric layers blend across the entire transition.
    a, b = dict(base_a), dict(base_b)
    a["other"] = (0.0, 1.0)
    b["other"] = (0.0, 1.0)
    specs.append(("atmosphere_blend", a, b))

    return specs


def _automations(
    view_a: dict,
    view_b: dict,
    windows_a: dict[str, tuple[float, float]],
    windows_b: dict[str, tuple[float, float]],
    duration_s: float,
    floor_db: float,
) -> list[StemAutomation]:
    automations: list[StemAutomation] = []
    for stem in _ordered_names(view_a):
        start_f, end_f = windows_a.get(stem, _default_window(stem, "a"))
        automations.append(
            StemAutomation(
                source="a",
                stem=stem,
                points=_fade_points(
                    duration_s,
                    start_f * duration_s,
                    end_f * duration_s,
                    fade_in=False,
                    floor_db=floor_db,
                ),
            )
        )
    for stem in _ordered_names(view_b):
        start_f, end_f = windows_b.get(stem, _default_window(stem, "b"))
        automations.append(
            StemAutomation(
                source="b",
                stem=stem,
                points=_fade_points(
                    duration_s,
                    start_f * duration_s,
                    end_f * duration_s,
                    fade_in=True,
                    floor_db=floor_db,
                ),
            )
        )
    return automations


# ---------------------------------------------------------------------------
# Loss terms
# ---------------------------------------------------------------------------

def loudness_loss(
    audio,
    start_s: float,
    duration_s: float,
    *,
    window_s: float = 0.05,
    ref_std_db: float = 12.0,
) -> float:
    """Penalize loudness fluctuation (std of short-window RMS in dB)."""

    mono = audio.mono()
    sample_rate = audio.sample_rate
    start = max(0, int(round(start_s * sample_rate)))
    end = min(mono.shape[0], int(round((start_s + duration_s) * sample_rate)))
    if end - start < 2:
        return 0.0
    region = mono[start:end]
    window = max(1, int(round(window_s * sample_rate)))
    count = region.shape[0] // window
    if count < 2:
        return 0.0
    trimmed = region[: count * window].reshape(count, window)
    rms = np.sqrt(np.mean(np.square(trimmed), axis=1))
    db = 20.0 * np.log10(np.maximum(rms, 1e-12))
    return float(np.clip(float(np.std(db)) / ref_std_db, 0.0, 1.0))


def continuity_loss(plan: TransitionPlan, *, samples: int = 200, ref_step: float = 0.02) -> float:
    """Penalize abrupt gain changes: max second difference of the linear gain."""

    times = np.linspace(0.0, plan.duration_s, samples)
    worst = 0.0
    for automation in plan.automations:
        gains = interpolate_gain(
            automation.points, times, curve=automation.curve, floor_db=_FLOOR_DB
        )
        second = np.abs(np.diff(gains, 2))
        if second.size:
            worst = max(worst, float(np.max(second)))
    return float(np.clip(worst / ref_step, 0.0, 1.0))


def musical_loss(plan: TransitionPlan) -> float:
    """Placeholder for arrangement-quality loss.

    A meaningful implementation requires phrase-boundary, section, and melodic
    resolution analysis, which is deliberately out of scope for Core. Returning
    zero keeps the term inert rather than pretending to measure something that
    is not yet measured.
    """

    return 0.0


def vocal_overlap_loss(plan: TransitionPlan) -> float:
    """Penalize overlap between outgoing and incoming vocals."""

    a_end: float | None = None
    b_start: float | None = None
    for automation in plan.automations:
        if automation.stem != "vocals":
            continue
        for point in automation.points:
            if automation.source == "a" and point.gain_db <= _FLOOR_DB + 1e-9:
                a_end = point.time_s
                break
            if automation.source == "b" and point.gain_db > _FLOOR_DB + 1e-9:
                b_start = point.time_s
                break
    if a_end is None or b_start is None or plan.duration_s <= 0:
        return 0.0
    overlap = a_end - b_start
    if overlap <= 0:
        return 0.0
    return float(np.clip(overlap / plan.duration_s, 0.0, 1.0))


def rhythm_loss(
    plan: TransitionPlan, *, tempo_bpm: float | None, beats_per_bar: int = 4
) -> float:
    """Penalize fade boundaries that miss the beat grid (when tempo is usable)."""

    if tempo_bpm is None or tempo_bpm <= 0 or plan.duration_s <= 0:
        return 0.0
    beat_s = 60.0 / float(tempo_bpm)
    errors: list[float] = []
    for automation in plan.automations:
        for point in automation.points:
            t = point.time_s
            if t <= 1e-9 or t >= plan.duration_s - 1e-9:
                continue
            nearest = round(t / beat_s) * beat_s
            errors.append(abs(t - nearest) / beat_s)
    if not errors:
        return 0.0
    return float(np.clip(float(np.mean(errors)) / 0.5, 0.0, 1.0))


def distortion_loss(master_gain_db: float, *, ref_db: float = 6.0) -> float:
    """Penalize how much master gain the renderer had to apply to avoid clipping."""

    reduction_db = max(0.0, -float(master_gain_db))
    return float(np.clip(reduction_db / ref_db, 0.0, 1.0))


def complexity_loss(plan: TransitionPlan, *, ref_points: float = 64.0) -> float:
    """Penalize needless automation points and effects."""

    n_points = sum(len(a.points) for a in plan.automations)
    n_effects = sum(len(a.effects) for a in plan.automations)
    return float(np.clip((n_points + 3.0 * n_effects) / ref_points, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def evaluate_losses(
    audio,
    master_gain_db: float,
    plan: TransitionPlan,
    *,
    tempo_bpm: float | None,
    config: DCPConfig | None = None,
) -> dict[str, float]:
    """Compute the seven normalized loss terms for one rendered candidate."""

    return {
        "loudness": loudness_loss(audio, plan.start_time_s, plan.duration_s),
        "continuity": continuity_loss(plan),
        "musical": musical_loss(plan),
        "vocal": vocal_overlap_loss(plan),
        "rhythm": rhythm_loss(plan, tempo_bpm=tempo_bpm),
        "distortion": distortion_loss(master_gain_db),
        "complexity": complexity_loss(plan),
    }


def objective(
    losses: Mapping[str, float], config: DCPConfig | None = None
) -> float:
    """Weighted sum of the normalized loss terms."""

    weights = (config or DEFAULT_CONFIG).optimization
    return (
        weights.w_loudness * losses["loudness"]
        + weights.w_continuity * losses["continuity"]
        + weights.w_musical * losses["musical"]
        + weights.w_vocal * losses["vocal"]
        + weights.w_rhythm * losses["rhythm"]
        + weights.w_distortion * losses["distortion"]
        + weights.w_complexity * losses["complexity"]
    )


def _weight_dict(config: DCPConfig) -> dict[str, float]:
    weights = config.optimization
    return {
        "loudness": weights.w_loudness,
        "continuity": weights.w_continuity,
        "musical": weights.w_musical,
        "vocal": weights.w_vocal,
        "rhythm": weights.w_rhythm,
        "distortion": weights.w_distortion,
        "complexity": weights.w_complexity,
    }


# ---------------------------------------------------------------------------
# Pluggable quality scoring
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WeightedLossScorer:
    """Core's default scorer: the hand-tuned weighted sum of loss terms.

    This is deliberate, deterministic arithmetic over measured engineering
    proxies. It is not trained on listening data and is not a perceptual model.
    """

    config: DCPConfig = DEFAULT_CONFIG
    name: ClassVar[str] = "weighted_loss"

    def score(self, features: CandidateFeatures) -> float:
        missing = [name for name in LOSS_NAMES if name not in features.losses]
        if missing:
            raise QualityScoringError(
                "weighted_loss scorer requires every loss term; missing: "
                + ", ".join(missing)
            )
        return objective(features.losses, self.config)


@dataclass(frozen=True)
class CandidateEvaluation:
    """One rendered candidate with its features, scorer output, and render."""

    plan: TransitionPlan
    features: CandidateFeatures
    score: float
    render: "RenderResult"


def scorer_name(scorer: QualityScorer) -> str:
    """Validate the scorer contract and return its name."""

    name = getattr(scorer, "name", None)
    if not isinstance(name, str) or not name.strip():
        raise QualityScoringError(
            "a quality scorer must define a non-empty string 'name'"
        )
    if not callable(getattr(scorer, "score", None)):
        raise QualityScoringError(
            f"quality scorer {name!r} must define a callable score(features)"
        )
    return name


def score_features(features: CandidateFeatures, scorer: QualityScorer) -> float:
    """Run a scorer and reject non-numeric or non-finite output."""

    name = scorer_name(scorer)
    try:
        value = scorer.score(features)
    except QualityScoringError:
        raise
    except Exception as exc:
        raise QualityScoringError(
            f"quality scorer {name!r} failed on candidate "
            f"{features.candidate!r}: {type(exc).__name__}: {exc}"
        ) from exc
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QualityScoringError(
            f"quality scorer {name!r} returned {type(value).__name__} for "
            f"candidate {features.candidate!r}; expected a number"
        )
    numeric = float(value)
    if not math.isfinite(numeric):
        raise QualityScoringError(
            f"quality scorer {name!r} returned a non-finite score for "
            f"candidate {features.candidate!r}"
        )
    return numeric


# ---------------------------------------------------------------------------
# Candidate generation and evaluation
# ---------------------------------------------------------------------------

def generate_candidate_plans(context: TransitionContext) -> list[TransitionPlan]:
    """Build every structurally valid role-timing candidate, without rendering.

    Returns an empty list when no stems are available. Invalid plans are
    skipped rather than returned, so every result can be rendered.
    """

    if context.stems_a is None and context.stems_b is None:
        return []

    config = context.config
    duration, duration_warnings = _clamp_requested_duration(
        context, context.requested_duration_s
    )
    view_a, view_b, warnings, _reasoning = _views(context)
    warnings = list(warnings) + list(duration_warnings)
    floor_db = config.transition.fade_floor_db

    specs = _candidate_specs()[: max(1, int(config.optimization.max_candidates))]
    plans: list[TransitionPlan] = []
    for label, windows_a, windows_b in specs:
        automations = _automations(
            view_a, view_b, windows_a, windows_b, duration, floor_db
        )
        candidate = _make_plan(
            context,
            strategy="optimized",
            duration_s=duration,
            automations=automations,
            reasoning=[],
            warnings=list(warnings),
            metadata={"candidate": label},
        )
        try:
            validate_plan(candidate, config, sample_rate=context.song_a.sample_rate)
        except PlanValidationError:
            continue
        plans.append(candidate)
    return plans


def candidate_features(
    context: TransitionContext,
    plan: TransitionPlan,
    losses: Mapping[str, float],
) -> CandidateFeatures:
    """Describe one candidate for scoring and preference logging."""

    return CandidateFeatures(
        candidate=str(plan.metadata.get("candidate", "")),
        losses=losses,
        duration_s=plan.duration_s,
        tempo_bpm=context.analysis_a.tempo_bpm,
        automation_count=len(plan.automations),
        gain_point_count=sum(len(a.points) for a in plan.automations),
        effect_count=sum(len(a.effects) for a in plan.automations),
        vocal_overlap_s=vocal_overlap_loss(plan) * plan.duration_s,
        stem_aware=context.stems_a is not None or context.stems_b is not None,
    )


def evaluate_candidate(
    context: TransitionContext,
    plan: TransitionPlan,
    *,
    scorer: QualityScorer | None = None,
) -> CandidateEvaluation:
    """Validate, render, measure, and score one candidate plan.

    Rendering or validation failures raise :class:`DCPError`; scorer failures
    raise :class:`QualityScoringError`, which callers should surface rather than
    treat as a rejected candidate.
    """

    # Imported lazily to avoid a transitions<->rendering import cycle.
    from ..rendering import render_plan

    config = context.config
    validate_plan(plan, config, sample_rate=context.song_a.sample_rate)
    result = render_plan(
        context.song_a,
        context.song_b,
        plan,
        stems_a=context.stems_a,
        stems_b=context.stems_b,
        config=config,
    )
    losses = evaluate_losses(
        result.audio,
        result.master_gain_db,
        plan,
        tempo_bpm=context.analysis_a.tempo_bpm,
        config=config,
    )
    features = candidate_features(context, plan, losses)
    active = scorer if scorer is not None else WeightedLossScorer(config)
    return CandidateEvaluation(
        plan=plan,
        features=features,
        score=score_features(features, active),
        render=result,
    )


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def optimize_transition_from_context(
    context: TransitionContext,
    *,
    scorer: QualityScorer | None = None,
) -> TransitionPlan:
    """Generate, render, and score candidate timing plans; return the best.

    Candidates are ranked by ``scorer`` (default: :class:`WeightedLossScorer`).
    Falls back to the conservative plan when no stems are available or when no
    candidate can be rendered.
    """

    config = context.config
    if context.stems_a is None and context.stems_b is None:
        return _conservative_plan(
            context,
            requested_strategy=context.requested_strategy,
            extra_warnings=[
                "Role-aware optimization requires stems for at least one song."
            ],
            prefix_reasoning=(
                "Without stems there are no roles to schedule, so the "
                "conservative full-mix plan is used."
            ),
        )

    active = scorer if scorer is not None else WeightedLossScorer(config)
    active_name = scorer_name(active)

    duration, duration_warnings = _clamp_requested_duration(
        context, context.requested_duration_s
    )
    _view_a, _view_b, warnings, _reasoning = _views(context)
    warnings = list(warnings) + list(duration_warnings)

    plans = generate_candidate_plans(context)
    evaluations: list[CandidateEvaluation] = []
    for plan in plans:
        try:
            evaluations.append(evaluate_candidate(context, plan, scorer=active))
        except QualityScoringError:
            raise
        except DCPError:
            continue

    if not evaluations:
        return _conservative_plan(
            context,
            requested_strategy=context.requested_strategy,
            extra_warnings=["No valid optimized candidate could be produced."],
            prefix_reasoning=(
                "Every candidate was rejected, so the conservative plan is used."
            ),
        )

    best = min(evaluations, key=lambda evaluation: evaluation.score)
    best_losses = {name: best.features.losses[name] for name in LOSS_NAMES}
    detail = ", ".join(f"{name}={value:.3f}" for name, value in best_losses.items())
    reasoning = [
        f"Evaluated {len(evaluations)} candidate timing plan(s) and selected "
        f"'{best.features.candidate}' with score {best.score:.3f}.",
    ]
    if isinstance(active, WeightedLossScorer):
        reasoning.append(f"Objective terms: {detail}.")
        reasoning.append(
            "The objective is a weighted sum of normalized engineering proxies "
            "(loudness, continuity, vocal overlap, rhythm, distortion, "
            "complexity); it is not a perceptual quality model."
        )
    else:
        reasoning.append(f"Diagnostic loss terms: {detail}.")
        reasoning.append(
            f"Candidates were ranked by the pluggable scorer {active_name!r} "
            "(lower is better); Core did not evaluate perceptual quality."
        )
    if context.stems_a is not None and context.stems_b is not None:
        reasoning.append(
            "Independent stem automation was scheduled for both songs."
        )
    else:
        reasoning.append(
            "Stem automation was scheduled where stems exist; the other side "
            "uses a full-mix fade."
        )

    return _make_plan(
        context,
        strategy="optimized",
        duration_s=duration,
        automations=best.plan.automations,
        reasoning=tuple(reasoning),
        warnings=tuple(warnings),
        metadata={
            "candidate": best.features.candidate,
            "objective": round(best.score, 6),
            "losses": {name: round(best_losses[name], 6) for name in LOSS_NAMES},
            "weights": (
                _weight_dict(config)
                if isinstance(active, WeightedLossScorer)
                else None
            ),
            "scorer": active_name,
            "candidates_generated": len(plans),
            "candidates_evaluated": len(evaluations),
            "stem_aware": True,
        },
    )


def optimize_transition(
    song_a,
    song_b,
    *,
    stems_a=None,
    stems_b=None,
    strategy: str = "optimized",
    duration_s: float | None = None,
    config: DCPConfig | None = None,
    analysis_a=None,
    analysis_b=None,
    scorer: QualityScorer | None = None,
) -> TransitionPlan:
    """Convenience wrapper around :func:`optimize_transition_from_context`."""

    from ..analysis import analyze_track

    cfg = config or DEFAULT_CONFIG
    requested = (
        cfg.transition.default_duration_s if duration_s is None else float(duration_s)
    )
    if not np.isfinite(requested) or requested <= 0:
        raise DCPError(
            f"transition duration must be a positive finite number, got {duration_s}"
        )
    context = TransitionContext(
        song_a=song_a,
        song_b=song_b,
        stems_a=stems_a,
        stems_b=stems_b,
        analysis_a=analysis_a or analyze_track(song_a, cfg.analysis),
        analysis_b=analysis_b or analyze_track(song_b, cfg.analysis),
        requested_duration_s=requested,
        requested_strategy=strategy,
        config=cfg,
    )
    return optimize_transition_from_context(context, scorer=scorer)


__all__ = [
    "LOSS_NAMES",
    "CandidateEvaluation",
    "WeightedLossScorer",
    "candidate_features",
    "complexity_loss",
    "continuity_loss",
    "distortion_loss",
    "evaluate_candidate",
    "evaluate_losses",
    "generate_candidate_plans",
    "loudness_loss",
    "musical_loss",
    "objective",
    "optimize_transition",
    "optimize_transition_from_context",
    "rhythm_loss",
    "score_features",
    "scorer_name",
    "vocal_overlap_loss",
]

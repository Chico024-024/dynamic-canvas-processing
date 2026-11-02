"""Transition-plan data model, serialization, and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from ..config import DEFAULT_CONFIG, DCPConfig
from ..models import PlanValidationError

_ALLOWED_SOURCES = ("a", "b")
_ALLOWED_CURVES = ("smoothstep", "linear")
_ALLOWED_EFFECTS = ("lowpass_sweep",)
_TIME_TOLERANCE = 1e-6


@dataclass(frozen=True)
class GainPoint:
    """A gain value in decibels at a time relative to the transition start."""

    time_s: float
    gain_db: float


@dataclass(frozen=True)
class EffectSpec:
    """A supported per-stem effect.

    ``lowpass_sweep`` applies two cascaded one-pole low-pass stages whose cutoff
    glides from ``start_hz`` to ``end_hz`` between ``start_s`` and ``end_s``
    (both relative to the transition start). The final cutoff is held after the
    sweep. Effects are optional metadata on a stem automation.
    """

    kind: str = "lowpass_sweep"
    start_hz: float = 16000.0
    end_hz: float = 1200.0
    start_s: float = 0.0
    end_s: float | None = None


@dataclass(frozen=True)
class StemAutomation:
    """Gain automation and optional effects for one stem from one source."""

    source: str
    stem: str
    points: tuple[GainPoint, ...]
    curve: str = "smoothstep"
    effects: tuple[EffectSpec, ...] = ()


@dataclass(frozen=True)
class TransitionPlan:
    """An explainable description of how two tracks should be combined.

    Times for gain points and effects are relative to ``start_time_s``.
    ``start_time_s`` is also where Song B begins in the rendered output.
    """

    strategy: str
    requested_strategy: str
    duration_s: float
    start_time_s: float
    end_time_s: float
    automations: tuple[StemAutomation, ...]
    reasoning: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def automation_for(self, source: str, stem: str) -> StemAutomation | None:
        for automation in self.automations:
            if automation.source == source and automation.stem == stem:
                return automation
        return None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "strategy": self.strategy,
            "requested_strategy": self.requested_strategy,
            "duration_s": round(self.duration_s, 6),
            "start_time_s": round(self.start_time_s, 6),
            "end_time_s": round(self.end_time_s, 6),
            "reasoning": list(self.reasoning),
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
            "automations": [
                {
                    "source": automation.source,
                    "stem": automation.stem,
                    "curve": automation.curve,
                    "points": [
                        {
                            "time_s": round(point.time_s, 6),
                            "gain_db": round(point.gain_db, 3),
                        }
                        for point in automation.points
                    ],
                    "effects": [
                        {
                            "kind": effect.kind,
                            "start_hz": effect.start_hz,
                            "end_hz": effect.end_hz,
                            "start_s": effect.start_s,
                            "end_s": effect.end_s,
                        }
                        for effect in automation.effects
                    ],
                }
                for automation in self.automations
            ],
        }


def validate_plan(
    plan: TransitionPlan,
    config: DCPConfig | None = None,
    *,
    sample_rate: int | None = None,
) -> None:
    """Validate structural and numeric invariants, raising PlanValidationError."""

    cfg = config or DEFAULT_CONFIG
    transition_cfg = cfg.transition
    errors: list[str] = []

    if not plan.strategy or not plan.requested_strategy:
        errors.append("strategy and requested_strategy must be non-empty")
    if plan.duration_s <= 0 or not _is_finite(plan.duration_s):
        errors.append("duration_s must be a positive finite number")
    if plan.start_time_s < 0 or not _is_finite(plan.start_time_s):
        errors.append("start_time_s must be finite and non-negative")
    if not _is_finite(plan.end_time_s):
        errors.append("end_time_s must be finite")
    elif abs(plan.end_time_s - (plan.start_time_s + plan.duration_s)) > 1e-4:
        errors.append(
            "end_time_s must equal start_time_s + duration_s "
            f"({plan.start_time_s + plan.duration_s:.6f})"
        )
    if not plan.automations:
        errors.append("a plan must contain at least one stem automation")

    seen: set[tuple[str, str]] = set()
    sources: set[str] = set()
    nyquist = None if sample_rate is None else sample_rate / 2.0

    for index, automation in enumerate(plan.automations):
        label = f"automation[{index}]"
        if automation.source not in _ALLOWED_SOURCES:
            errors.append(f"{label}: source must be 'a' or 'b'")
        else:
            sources.add(automation.source)
        if not automation.stem or not automation.stem.strip():
            errors.append(f"{label}: stem name must be non-empty")
        key = (automation.source, automation.stem)
        if key in seen:
            errors.append(f"{label}: duplicate automation for {automation.source}/{automation.stem}")
        seen.add(key)

        if automation.curve not in _ALLOWED_CURVES:
            errors.append(
                f"{label}: unsupported curve {automation.curve!r}; "
                f"expected one of {_ALLOWED_CURVES}"
            )
        if len(automation.points) < 2:
            errors.append(f"{label}: at least two gain points are required")

        previous_time = None
        for point_index, point in enumerate(automation.points):
            point_label = f"{label}.points[{point_index}]"
            if not _is_finite(point.time_s) or not _is_finite(point.gain_db):
                errors.append(f"{point_label}: time and gain must be finite")
                continue
            if point.time_s < -_TIME_TOLERANCE or point.time_s > plan.duration_s + _TIME_TOLERANCE:
                errors.append(
                    f"{point_label}: time {point.time_s:.6f}s is outside the "
                    f"transition window [0, {plan.duration_s:.6f}]"
                )
            if previous_time is not None and point.time_s <= previous_time + _TIME_TOLERANCE:
                errors.append(f"{point_label}: gain points must be strictly increasing")
            previous_time = point.time_s
            if point.gain_db > transition_cfg.max_automation_gain_db:
                errors.append(
                    f"{point_label}: gain {point.gain_db:.2f} dB exceeds the "
                    f"maximum {transition_cfg.max_automation_gain_db:.2f} dB"
                )
            if point.gain_db < transition_cfg.fade_floor_db:
                errors.append(
                    f"{point_label}: gain {point.gain_db:.2f} dB is below the "
                    f"floor {transition_cfg.fade_floor_db:.2f} dB"
                )

        for effect_index, effect in enumerate(automation.effects):
            effect_label = f"{label}.effects[{effect_index}]"
            if effect.kind not in _ALLOWED_EFFECTS:
                errors.append(f"{effect_label}: unsupported effect kind {effect.kind!r}")
            if not _is_finite(effect.start_hz) or effect.start_hz <= 20.0:
                errors.append(f"{effect_label}: start_hz must be greater than 20 Hz")
            if not _is_finite(effect.end_hz) or effect.end_hz <= 20.0:
                errors.append(f"{effect_label}: end_hz must be greater than 20 Hz")
            if nyquist is not None:
                limit = nyquist * 0.98
                if effect.start_hz >= limit or effect.end_hz >= limit:
                    errors.append(
                        f"{effect_label}: cutoff must stay below the Nyquist "
                        f"limit ({limit:.0f} Hz) for {sample_rate} Hz audio"
                    )
            end_s = plan.duration_s if effect.end_s is None else effect.end_s
            if not _is_finite(effect.start_s) or effect.start_s < -_TIME_TOLERANCE:
                errors.append(f"{effect_label}: start_s must be non-negative")
            if not _is_finite(end_s) or end_s > plan.duration_s + _TIME_TOLERANCE:
                errors.append(f"{effect_label}: end_s must be within the transition")
            if end_s <= effect.start_s + _TIME_TOLERANCE:
                errors.append(f"{effect_label}: end_s must be after start_s")

    if "a" not in sources:
        errors.append("plan must include at least one automation for source 'a'")
    if "b" not in sources:
        errors.append("plan must include at least one automation for source 'b'")

    if errors:
        raise PlanValidationError("invalid transition plan: " + "; ".join(errors))


def _is_finite(value: float) -> bool:
    return isinstance(value, (int, float)) and value == value and abs(value) != float("inf")


@dataclass(frozen=True)
class TransitionParams:
    """
    Structured high-level targets for a transition.

    These are the discrete parameters that a policy network or strategy
    decides on. The deterministic curve engine then builds the exact
    smoothstep automation curves from these targets.

    This replaces the old approach of generating raw (time_s, gain_db) tokens.
    """

    # Core strategy selection
    strategy_type: str = "stem_level"  # stem_level, rhythmic, atmospheric, conservative, crossfade

    # Timing structure
    duration_bars: int = 4              # Whole bars (e.g. 4, 8, 16)
    anchor_bar_a: int = 1               # Measure boundary in Song A where transition starts
    anchor_bar_b: int = 1               # Measure boundary in Song B where Song B enters

    # Phase locking
    cue_offset_b_s: float = 0.0         # Song B start offset for downbeat-to-downbeat phase lock

    # Stem handoff timing (relative to transition start, in seconds)
    vocal_handoff: tuple[float, float] = (0.0, 0.0)  # (A_vocal_fade_out_s, B_vocal_fade_in_s)
    bass_crossover_bar: int = 1         # Exact bar where low-end switches from A to B
    drum_blend_bars: int = 2            # Drum overlap duration in bars

    # Filter targets per stem (stem_name -> target_cutoff_hz)
    filter_target_hz: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy_type": self.strategy_type,
            "duration_bars": self.duration_bars,
            "anchor_bar_a": self.anchor_bar_a,
            "anchor_bar_b": self.anchor_bar_b,
            "cue_offset_b_s": round(self.cue_offset_b_s, 6),
            "vocal_handoff": [round(x, 6) for x in self.vocal_handoff],
            "bass_crossover_bar": self.bass_crossover_bar,
            "drum_blend_bars": self.drum_blend_bars,
            "filter_target_hz": dict(self.filter_target_hz),
        }


__all__ = [
    "GainPoint",
    "EffectSpec",
    "StemAutomation",
    "TransitionPlan",
    "TransitionParams",
    "validate_plan",
]

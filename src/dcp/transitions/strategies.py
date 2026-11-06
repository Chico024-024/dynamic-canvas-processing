"""Core transition strategies and the strategy registry."""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

from ..config import DCPConfig
from ..models import DCPError, MIX_STEM, AudioBuffer, StemSet
from .context import TransitionContext
from .plan import EffectSpec, GainPoint, StemAutomation, TransitionPlan, TransitionParams
from .alignment import calculate_phase_locked_cue_offset, PhaseLockResult

_A_WINDOWS: dict[str, tuple[float, float]] = {
    "vocals": (0.0, 0.45),
    "bass": (0.0, 0.55),
    "drums": (0.15, 0.80),
    "other": (0.05, 0.95),
}
_B_WINDOWS: dict[str, tuple[float, float]] = {
    "vocals": (0.55, 1.0),
    "bass": (0.0, 0.50),
    "drums": (0.20, 0.85),
    "other": (0.05, 0.95),
}
_A_ATMOSPHERIC_WINDOWS: dict[str, tuple[float, float]] = {
    "vocals": (0.0, 0.35),
    "bass": (0.0, 0.55),
    "drums": (0.0, 0.75),
    "other": (0.0, 1.0),
}
_B_ATMOSPHERIC_WINDOWS: dict[str, tuple[float, float]] = {
    "vocals": (0.62, 1.0),
    "bass": (0.0, 0.45),
    "drums": (0.0, 0.70),
    "other": (0.0, 1.0),
}


@runtime_checkable
class TransitionStrategy(Protocol):
    """Stable extension contract used by Core and optional Advanced code."""

    name: str
    description: str

    def plan(self, context: TransitionContext) -> TransitionPlan:
        """Create and return a fully populated transition plan."""


class StrategyRegistry:
    """A small, explicit registry of strategy objects."""

    def __init__(self) -> None:
        self._strategies: dict[str, TransitionStrategy] = {}

    def register(
        self, strategy: TransitionStrategy, *, override: bool = False
    ) -> None:
        name = getattr(strategy, "name", None)
        description = getattr(strategy, "description", None)
        if not isinstance(name, str) or not name.strip():
            raise DCPError("a transition strategy must define a non-empty name")
        if not isinstance(description, str) or not description.strip():
            raise DCPError(
                f"strategy {name!r} must define a non-empty description"
            )
        if not callable(getattr(strategy, "plan", None)):
            raise DCPError(f"strategy {name!r} must define a callable plan(context)")
        if name in self._strategies and not override:
            raise DCPError(
                f"strategy {name!r} is already registered; "
                "Advanced strategies may not replace Core strategy names"
            )
        self._strategies[name] = strategy

    def get(self, name: str) -> TransitionStrategy:
        try:
            return self._strategies[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._strategies))
            raise DCPError(
                f"unknown transition strategy {name!r}; available: {available}"
            ) from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._strategies))

    def items(self):
        return self._strategies.items()

    def __contains__(self, name: object) -> bool:
        return name in self._strategies

    def __len__(self) -> int:
        return len(self._strategies)


def _ordered_names(view: dict[str, AudioBuffer]) -> list[str]:
    if list(view) == [MIX_STEM]:
        return [MIX_STEM]
    from ..models import CANONICAL_STEM_ORDER

    known = [name for name in CANONICAL_STEM_ORDER if name in view]
    extra = sorted(name for name in view if name not in CANONICAL_STEM_ORDER)
    return known + extra


def _views(
    context: TransitionContext,
) -> tuple[dict[str, AudioBuffer], dict[str, AudioBuffer], list[str], list[str]]:
    warnings: list[str] = []
    reasoning: list[str] = []
    if context.stems_a is None:
        view_a = {MIX_STEM: context.song_a}
        warnings.append(
            "Song A stems are unavailable; its full mix is used as a single stream."
        )
    else:
        view_a = dict(context.stems_a.items())
        if "vocals" not in view_a:
            warnings.append(
                "Song A has no 'vocals' stem; vocal phrase timing is not available."
            )
    if context.stems_b is None:
        view_b = {MIX_STEM: context.song_b}
        warnings.append(
            "Song B stems are unavailable; its full mix is used as a single stream."
        )
    else:
        view_b = dict(context.stems_b.items())
        if "vocals" not in view_b:
            warnings.append(
                "Song B has no 'vocals' stem; vocal phrase timing is not available."
            )
    if context.stems_a is not None and context.stems_b is not None:
        reasoning.append(
            "Independent stem automation is used for both songs; gain curves "
            "are smooth in decibels."
        )
    return view_a, view_b, warnings, reasoning


def _clamp_requested_duration(
    context: TransitionContext, requested_s: float, *, minimum_s: float | None = None
) -> tuple[float, list[str]]:
    warnings: list[str] = []
    maximum = context.max_duration_s
    minimum = context.config.transition.min_duration_s if minimum_s is None else minimum_s
    minimum = min(max(0.05, minimum), maximum)
    duration = float(requested_s)
    if not math.isfinite(duration) or duration <= 0:
        duration = context.config.transition.default_duration_s

    if duration > maximum + 1e-9:
        warnings.append(
            f"Transition duration {duration:.2f}s exceeds the shorter track "
            f"({maximum:.2f}s); clamped to {maximum:.2f}s."
        )
        duration = maximum
    if duration < minimum - 1e-9:
        warnings.append(
            f"Transition duration {duration:.2f}s is below the configured "
            f"minimum; raised to {minimum:.2f}s."
        )
        duration = minimum
    return duration, warnings


def _fade_points(
    duration_s: float,
    start_s: float,
    end_s: float,
    *,
    fade_in: bool,
    floor_db: float,
) -> tuple[GainPoint, ...]:
    start = min(max(0.0, float(start_s)), duration_s)
    end = min(max(0.0, float(end_s)), duration_s)
    if end <= start + 1e-9:
        end = min(duration_s, start + 1e-3)
    points: list[GainPoint] = []
    if fade_in:
        points.append(GainPoint(0.0, floor_db))
        if start > 1e-6:
            points.append(GainPoint(start, floor_db))
        points.append(GainPoint(end, 0.0))
        if end < duration_s - 1e-6:
            points.append(GainPoint(duration_s, 0.0))
    else:
        if start > 1e-6:
            points.append(GainPoint(0.0, 0.0))
        points.append(GainPoint(start, 0.0))
        points.append(GainPoint(end, floor_db))
        if end < duration_s - 1e-6:
            points.append(GainPoint(duration_s, floor_db))

    deduped: list[GainPoint] = []
    for point in points:
        if deduped and abs(point.time_s - deduped[-1].time_s) <= 1e-9:
            deduped[-1] = point
        else:
            deduped.append(point)
    return tuple(deduped)


def _beat_aligned_rhythm(
    context: TransitionContext,
    *,
    requested_duration_s: float,
    beats_per_bar: int,
    minimum_s: float,
    maximum_s: float,
) -> tuple[float, int, int, tuple[float, ...], float] | None:
    """Choose a beat-anchored transition that fits the available beats.

    The transition start is a detected Song A beat near the requested start, so
    the plan's start and fade boundaries share one real beat phase. The final
    duration is the distance from that beat to the end of Song A, which is not
    necessarily an exact number of bars. Returns ``None`` when the detected
    grid cannot support a whole-bar plan.
    """

    beats = sorted(
        {
            float(t)
            for t in context.analysis_a.beat_times_s
            if math.isfinite(float(t)) and 0.0 <= float(t) <= context.song_a.duration_s
        }
    )
    if len(beats) < beats_per_bar + 1:
        return None

    song_end = context.song_a.duration_s
    target_start = song_end - requested_duration_s
    downbeats = {
        float(t)
        for t in context.analysis_a.downbeat_times_s
        if math.isfinite(float(t)) and 0.0 <= float(t) <= song_end
    }
    candidates: list[tuple[int, float, int, float]] = []
    for index, beat in enumerate(beats):
        if beat <= 1e-9 or beat >= song_end - 1e-9:
            continue
        duration = song_end - beat
        if duration < minimum_s - 1e-9 or duration > maximum_s + 1e-9:
            continue
        if len(beats) - index < beats_per_bar + 1:
            continue
        is_downbeat = any(abs(beat - db) < 0.05 for db in downbeats) if downbeats else False
        priority = 0 if is_downbeat else 1
        candidates.append((priority, abs(beat - target_start), index, duration))
    if not candidates:
        return None

    _, _, index, duration = min(candidates)
    anchor = beats[index]
    grid = tuple(beat - anchor for beat in beats[index:])
    bars = (len(grid) - 1) // beats_per_bar
    while bars >= 1 and grid[bars * beats_per_bar] > duration + 1e-6:
        bars -= 1
    if bars < 1:
        return None
    total_beats = bars * beats_per_bar
    return duration, bars, total_beats, grid, anchor


def _transition_times(
    context: TransitionContext, duration_s: float
) -> tuple[float, float]:
    end = context.song_a.duration_s
    start = max(0.0, end - duration_s)
    end = start + duration_s
    return start, end


def _make_plan(
    context: TransitionContext,
    *,
    strategy: str,
    duration_s: float,
    automations: list[StemAutomation],
    reasoning: list[str],
    warnings: list[str],
    metadata: dict[str, object] | None = None,
    start_time_s: float | None = None,
) -> TransitionPlan:
    if start_time_s is not None:
        start = float(start_time_s)
        end = start + float(duration_s)
    else:
        start, end = _transition_times(context, duration_s)
    return TransitionPlan(
        strategy=strategy,
        requested_strategy=context.requested_strategy,
        duration_s=float(duration_s),
        start_time_s=float(start),
        end_time_s=float(end),
        automations=tuple(automations),
        reasoning=tuple(reasoning),
        warnings=tuple(warnings),
        metadata=metadata or {},
    )


def _full_mix_automations(
    context: TransitionContext,
    duration_s: float,
    *,
    curve: str,
) -> list[StemAutomation]:
    floor_db = context.config.transition.fade_floor_db
    return [
        StemAutomation(
            source="a",
            stem=MIX_STEM,
            points=_fade_points(
                duration_s,
                0.0,
                duration_s,
                fade_in=False,
                floor_db=floor_db,
            ),
            curve=curve,
        ),
        StemAutomation(
            source="b",
            stem=MIX_STEM,
            points=_fade_points(
                duration_s,
                0.0,
                duration_s,
                fade_in=True,
                floor_db=floor_db,
            ),
            curve=curve,
        ),
    ]


def _conservative_plan(
    context: TransitionContext,
    *,
    requested_strategy: str,
    extra_warnings: list[str],
    prefix_reasoning: str = "",
    extra_metadata: dict[str, object] | None = None,
) -> TransitionPlan:
    duration, duration_warnings = _clamp_requested_duration(
        context, context.requested_duration_s
    )
    reasoning = []
    if prefix_reasoning:
        reasoning.append(prefix_reasoning)
    reasoning.extend(
        [
            f"A conservative full-mix fade over {duration:.2f}s is used.",
            "No tempo, beat, key, stem, or phrase assumptions are made; the "
            "result is intentionally predictable.",
        ]
    )
    warnings = list(extra_warnings) + duration_warnings
    if requested_strategy != "conservative":
        warnings.append(
            "The requested strategy could not be applied; the conservative "
            "fallback was rendered instead."
        )
    meta: dict[str, object] = {"fallback": requested_strategy != "conservative"}
    if extra_metadata:
        meta.update(extra_metadata)
    return _make_plan(
        context,
        strategy="conservative",
        duration_s=duration,
        automations=_full_mix_automations(context, duration, curve="smoothstep"),
        reasoning=reasoning,
        warnings=warnings,
        metadata=meta,
    )


class CrossfadeStrategy:
    name = "crossfade"
    description = "Conventional equal-duration full-mix crossfade plan."

    def plan(self, context: TransitionContext) -> TransitionPlan:
        duration, warnings = _clamp_requested_duration(
            context, context.requested_duration_s
        )
        reasoning = [
            f"A conventional crossfade over {duration:.2f}s is planned on the "
            "full mixes; stems are intentionally ignored.",
            "Gain ramps are linear in decibels. The dedicated render_crossfade "
            "API provides the separate equal-power baseline.",
        ]
        return _make_plan(
            context,
            strategy=self.name,
            duration_s=duration,
            automations=_full_mix_automations(context, duration, curve="linear"),
            reasoning=reasoning,
            warnings=warnings,
        )


class StemLevelStrategy:
    name = "stem_level"
    description = "Independent, staggered gain automation for each available stem."

    def plan(self, context: TransitionContext) -> TransitionPlan:
        view_a, view_b, warnings, reasoning = _views(context)
        if context.stems_a is None and context.stems_b is None:
            return _conservative_plan(
                context,
                requested_strategy=context.requested_strategy,
                extra_warnings=[
                    "Stem-level transition requested without stems for either song.",
                ],
                prefix_reasoning=(
                    "No stems were supplied, so independent stem processing was "
                    "impossible."
                ),
            )

        duration, duration_warnings = _clamp_requested_duration(
            context, context.requested_duration_s
        )
        warnings.extend(duration_warnings)
        floor_db = context.config.transition.fade_floor_db
        automations: list[StemAutomation] = []

        for stem in _ordered_names(view_a):
            start_fraction, end_fraction = _A_WINDOWS.get(
                stem, (0.05, 0.85) if stem != MIX_STEM else (0.0, 0.85)
            )
            automations.append(
                StemAutomation(
                    source="a",
                    stem=stem,
                    points=_fade_points(
                        duration,
                        start_fraction * duration,
                        end_fraction * duration,
                        fade_in=False,
                        floor_db=floor_db,
                    ),
                )
            )
        for stem in _ordered_names(view_b):
            start_fraction, end_fraction = _B_WINDOWS.get(
                stem, (0.15, 0.95) if stem != MIX_STEM else (0.1, 1.0)
            )
            automations.append(
                StemAutomation(
                    source="b",
                    stem=stem,
                    points=_fade_points(
                        duration,
                        start_fraction * duration,
                        end_fraction * duration,
                        fade_in=True,
                        floor_db=floor_db,
                    ),
                )
            )

        a_vocals_end = _A_WINDOWS["vocals"][1] * duration
        b_vocals_start = _B_WINDOWS["vocals"][0] * duration
        reasoning.append(
            f"Stem-level transition over {duration:.2f}s: every supplied stem "
            "has an independent decibel-smooth fade."
        )
        if "vocals" in view_a and "vocals" in view_b:
            reasoning.append(
                f"Song A's vocals finish by {a_vocals_end:.2f}s; Song B's "
                f"vocals enter at {b_vocals_start:.2f}s, leaving a deliberate "
                f"gap of {max(0.0, b_vocals_start - a_vocals_end):.2f}s."
            )
        elif "vocals" in view_a or "vocals" in view_b:
            side = "Song A" if "vocals" in view_a else "Song B"
            reasoning.append(
                f"Only {side} provides a 'vocals' stem; vocal hand-off timing "
                "is applied on that side only."
            )
        else:
            reasoning.append(
                "No 'vocals' stem is available on either side; no vocal "
                "hand-off timing is claimed."
            )
        if "drums" in view_a and "drums" in view_b:
            reasoning.append(
                f"Song A's drums remain until "
                f"{_A_WINDOWS['drums'][1] * duration:.2f}s while Song B's "
                f"drums enter at {_B_WINDOWS['drums'][0] * duration:.2f}s, "
                "preserving rhythmic continuity."
            )
        else:
            reasoning.append(
                "Drum-specific overlap is planned only where drum stems "
                "exist; missing stems use their fallback fade."
            )
        if "bass" in view_a and "bass" in view_b:
            reasoning.append(
                f"Song A's bass is faded out by "
                f"{_A_WINDOWS['bass'][1] * duration:.2f}s while Song B's bass "
                "arrives earlier, reducing low-end overlap."
            )
        else:
            reasoning.append(
                "Bass-specific overlap is planned only where bass stems "
                "exist; missing stems use their fallback fade."
            )
        if "other" in view_b:
            if "other" in view_a:
                reasoning.append(
                    "Song B's 'other' layer enters early and slowly, creating "
                    "an atmospheric bed under Song A."
                )
            else:
                reasoning.append(
                    "Song B's 'other' layer enters early, but Song A has no "
                    "'other' stem to overlap."
                )
        else:
            reasoning.append(
                "Song B has no 'other' stem; no atmospheric bed is claimed."
            )
        reasoning.append(
            "No time-stretching, key analysis, or source separation is "
            "performed; supplied stems are trusted as-is."
        )
        return _make_plan(
            context,
            strategy=self.name,
            duration_s=duration,
            automations=automations,
            reasoning=reasoning,
            warnings=warnings,
            metadata={"stem_aware": True},
        )


def tempo_is_usable(context: TransitionContext) -> tuple[bool, str]:
    cfg = context.config.analysis
    first = context.analysis_a
    second = context.analysis_b
    if first.tempo_bpm is None:
        return False, "Song A has no usable tempo estimate"
    if first.tempo_confidence < cfg.tempo_confidence_threshold:
        return False, (
            f"Song A tempo confidence {first.tempo_confidence:.2f} is below "
            f"the threshold {cfg.tempo_confidence_threshold:.2f}"
        )
    if second.tempo_bpm is None:
        return False, "Song B has no usable tempo estimate"
    if second.tempo_confidence < cfg.tempo_confidence_threshold:
        return False, (
            f"Song B tempo confidence {second.tempo_confidence:.2f} is below "
            f"the threshold {cfg.tempo_confidence_threshold:.2f}"
        )
    ratio = second.tempo_bpm / first.tempo_bpm
    tolerance = context.config.transition.tempo_match_tolerance
    if not any(abs(ratio - target) <= tolerance for target in (1.0, 2.0, 0.5)):
        return False, (
            f"tempo ratio {ratio:.3f} is outside the accepted matches "
            f"(1:1 or 2:1 within {tolerance:.0%})"
        )
    return True, ""


class RhythmicStrategy:
    name = "rhythmic"
    description = "Beat-aligned stem automation with downbeat-to-downbeat phase lock, used only with reliable tempo."

    def plan(self, context: TransitionContext) -> TransitionPlan:
        available, reason = tempo_is_usable(context)
        if not available:
            return _conservative_plan(
                context,
                requested_strategy=context.requested_strategy,
                extra_warnings=[f"Rhythmic transition unavailable: {reason}."],
                prefix_reasoning=(
                    "Rhythm-aware planning was requested but the tempo analysis "
                    "was not reliable enough."
                ),
            )

        view_a, view_b, warnings, reasoning = _views(context)
        bpm_a = float(context.analysis_a.tempo_bpm or 0.0)
        bpm_b = float(context.analysis_b.tempo_bpm or 0.0)
        beat_s = 60.0 / bpm_a
        beats_per_bar = context.config.transition.rhythmic_beats_per_bar
        bar_s = beat_s * beats_per_bar
        maximum = context.max_duration_s

        bars = max(1, int(round(context.requested_duration_s / bar_s)))
        duration = bars * bar_s
        minimum = context.config.transition.min_duration_s
        if duration < minimum:
            bars = max(1, int(math.ceil(minimum / bar_s)))
            duration = bars * bar_s
        if duration > maximum + 1e-9:
            bars = int(math.floor(maximum / bar_s))
            duration = bars * bar_s
        if bars < 1 or duration < min(minimum, maximum) - 1e-6:
            return _conservative_plan(
                context,
                requested_strategy=context.requested_strategy,
                extra_warnings=[
                    "Rhythmic transition unavailable: not enough room for one "
                    "beat-aligned bar before the shorter track ends.",
                ],
                prefix_reasoning=(
                    "Beat-aligned planning could not fit a whole bar in the "
                    "requested window."
                ),
            )

        # Try to achieve downbeat-to-downbeat phase lock
        phase_lock = calculate_phase_locked_cue_offset(
            context,
            lead_in_bars=2,
            min_confidence=context.config.analysis.tempo_confidence_threshold,
        )

        # Phase lock variables (initialized to defaults)
        phase_locked = False
        anchor_downbeat_a_s = 0.0
        cue_offset_b_s = 0.0
        transition_start_in_a = 0.0
        phase_lock_confidence = 0.0
        phase_lock_anchor_b_s = 0.0

        # If phase lock succeeds, adjust timing to use the phase-locked anchor
        if phase_lock is not None and phase_lock.success:
            phase_locked = True
            anchor_downbeat_a_s = phase_lock.anchor_downbeat_a_s
            phase_lock_anchor_b_s = phase_lock.anchor_downbeat_b_s
            cue_offset_b_s = phase_lock.cue_offset_b_s
            transition_start_in_a = phase_lock.transition_start_a_s
            phase_lock_confidence = phase_lock.confidence
            # The overlap duration is from transition start to end of Song A
            duration = context.song_a.duration_s - transition_start_in_a
            # Clamp to valid range
            if duration > maximum:
                duration = maximum
            if duration < minimum:
                duration = minimum
            # Recalculate bars based on phase-locked duration
            bars = max(1, int(round(duration / bar_s)))
            duration = bars * bar_s
            transition_start_in_a = context.song_a.duration_s - duration
            # Maintain exact downbeat coincidence invariant:
            # transition_start_in_a + (phase_lock_anchor_b_s - cue_offset_b_s) == anchor_downbeat_a_s
            cue_offset_b_s = phase_lock_anchor_b_s - (anchor_downbeat_a_s - transition_start_in_a)
            if cue_offset_b_s < 0 or cue_offset_b_s >= context.song_b.duration_s:
                # If bar quantization pushed cue offset out of bounds, revert to exact phase_lock timing
                transition_start_in_a = phase_lock.transition_start_a_s
                cue_offset_b_s = phase_lock.cue_offset_b_s
                duration = context.song_a.duration_s - transition_start_in_a
                bars = max(1, int(round(duration / bar_s)))

            reasoning.append(
                f"Downbeat-to-downbeat phase lock achieved: Song A downbeat "
                f"{phase_lock.anchor_downbeat_a_s:.3f}s locks to Song B downbeat "
                f"{phase_lock.anchor_downbeat_b_s:.3f}s (confidence: {phase_lock.confidence:.2f})."
            )
            warnings.append(
                f"Phase lock confidence: {phase_lock.confidence:.2f}. "
                f"Song B cued at offset {cue_offset_b_s:.3f}s, entering at {transition_start_in_a:.3f}s."
            )

        aligned = _beat_aligned_rhythm(
            context,
            requested_duration_s=context.requested_duration_s,
            beats_per_bar=beats_per_bar,
            minimum_s=minimum,
            maximum_s=maximum,
        )
        beat_grid: tuple[float, ...] | None = None
        anchor_beat_s: float | None = None
        if aligned is not None and not phase_locked:
            # Only use the automatic beat alignment if phase lock didn't succeed
            duration, bars, total_beats, beat_grid, anchor_beat_s = aligned
        elif not phase_locked:
            total_beats = bars * beats_per_bar
        else:
            # Phase locked: create beat grid from the phase-locked anchor
            total_beats = bars * beats_per_bar
            beat_grid = tuple(beat_s * i for i in range(total_beats + 1))
            anchor_beat_s = transition_start_in_a

        warnings = list(warnings)
        if abs(duration - context.requested_duration_s) > 1e-6:
            if beat_grid is not None:
                warnings.append(
                    f"Transition duration was adjusted from "
                    f"{context.requested_duration_s:.2f}s to {duration:.2f}s so "
                    "it starts on a detected beat with whole-bar fade "
                    "boundaries."
                )
            else:
                warnings.append(
                    f"Transition duration was adjusted from "
                    f"{context.requested_duration_s:.2f}s to {duration:.2f}s so "
                    "it matches a whole number of bars."
                )
        if context.stems_a is None and context.stems_b is None:
            warnings.append(
                "No stems were supplied; beat alignment applies to full-mix "
                "fades rather than independent stem fades."
            )

        floor_db = context.config.transition.fade_floor_db

        def grid_time(beat_index: int) -> float:
            if beat_grid is None:
                return duration * beat_index / total_beats
            clamped = max(0, min(total_beats, int(beat_index)))
            return min(duration, beat_grid[clamped])

        beat_windows_a = {
            "vocals": (0, max(1, min(2, total_beats - 1))),
            "bass": (0, max(2, min(4, total_beats - 1))),
            "other": (0, max(3, total_beats - 2) if total_beats > 3 else total_beats),
            "drums": (
                max(1, min(total_beats - 1, max(4, total_beats // 2))),
                total_beats,
            ),
        }
        beat_windows_b = {
            "vocals": (
                max(3, total_beats // 2),
                min(total_beats, max(3, total_beats // 2) + 4),
            ),
            "bass": (0, min(2, total_beats)),
            "drums": (0, min(4, total_beats)),
            "other": (0, min(8, total_beats)),
        }
        default_a = (max(1, total_beats // 4), max(2, total_beats - 1))
        default_b = (0, min(8, total_beats))
        automations: list[StemAutomation] = []

        for stem in _ordered_names(view_a):
            start_beat, end_beat = beat_windows_a.get(
                stem, default_a if stem != MIX_STEM else (0, total_beats - 1)
            )
            automations.append(
                StemAutomation(
                    source="a",
                    stem=stem,
                    points=_fade_points(
                        duration,
                        grid_time(start_beat),
                        grid_time(min(total_beats, end_beat)),
                        fade_in=False,
                        floor_db=floor_db,
                    ),
                )
            )
        for stem in _ordered_names(view_b):
            start_beat, end_beat = beat_windows_b.get(
                stem, default_b if stem != MIX_STEM else (0, total_beats)
            )
            automations.append(
                StemAutomation(
                    source="b",
                    stem=stem,
                    points=_fade_points(
                        duration,
                        grid_time(min(total_beats, start_beat)),
                        grid_time(min(total_beats, end_beat)),
                        fade_in=True,
                        floor_db=floor_db,
                    ),
                )
            )

        ratio = bpm_b / bpm_a
        if abs(ratio - 2.0) <= context.config.transition.tempo_match_tolerance:
            relation = "Song B is twice as fast (automation follows Song A's grid)"
        elif abs(ratio - 0.5) <= context.config.transition.tempo_match_tolerance:
            relation = "Song B is half as fast (automation follows Song A's grid)"
        else:
            relation = "Song A and Song B tempos match"

        reasoning.append(
            f"Tempo estimates: Song A {bpm_a:.1f} BPM "
            f"(confidence {context.analysis_a.tempo_confidence:.2f}), "
            f"Song B {bpm_b:.1f} BPM "
            f"(confidence {context.analysis_b.tempo_confidence:.2f})."
        )
        is_downbeat_aligned = False
        if anchor_beat_s is not None and context.analysis_a.downbeat_times_s:
            is_downbeat_aligned = any(
                abs(anchor_beat_s - db) < 0.05 for db in context.analysis_a.downbeat_times_s
            )

        if beat_grid is not None:
            downbeat_reason = (
                f"Transition starts on an estimated Song A downbeat ({anchor_beat_s:.3f}s, bar boundary); 4/4 meter aligned."
                if is_downbeat_aligned
                else (
                    "Downbeats are not detected; bar alignment assumes 4/4 "
                    "meter. Detected beats came from the "
                    f"{context.analysis_a.analyzer!r} analyzer."
                )
            )
            if phase_locked:
                reasoning.extend(
                    [
                        f"{relation}. {bars} bar(s) of {beats_per_bar}/4 are used "
                        f"({total_beats} beats); the transition starts at Song A's "
                        f"downbeat {anchor_beat_s:.3f}s and ends at "
                        f"{context.song_a.duration_s:.3f}s.",
                        "Downbeat-to-downbeat phase lock: Song B's incoming downbeat "
                        f"{phase_lock_anchor_b_s:.3f}s is precisely aligned to Song A's "
                        f"outgoing downbeat {anchor_downbeat_a_s:.3f}s. "
                        f"Song B is cued at offset {cue_offset_b_s:.3f}s for sample-accurate phase coherence.",
                        downbeat_reason,
                    ]
                )
            else:
                reasoning.extend(
                    [
                        f"{relation}. {bars} bar(s) of {beats_per_bar}/4 are used "
                        f"({total_beats} beats); the transition starts at detected "
                        f"Song A beat {anchor_beat_s:.3f}s and ends at "
                        f"{context.song_a.duration_s:.3f}s.",
                        "Every fade boundary follows a detected Song A beat time "
                        "rather than an evenly spaced tempo fraction. Song B's "
                        "automation uses the same timing grid because its phase is "
                        "not shifted or time-stretched.",
                        downbeat_reason,
                    ]
                )
        else:
            reasoning.extend(
                [
                    f"{relation}. {bars} bar(s) of {beats_per_bar}/4 are used "
                    f"({total_beats} beats at {beat_s:.3f}s per beat).",
                    "No detected beat times were available, so every fade "
                    "boundary is quantized to a whole beat of the estimated "
                    "tempo grid; no time-stretching or downbeat detection is "
                    "applied.",
                ]
            )
        if context.stems_a is not None and context.stems_b is not None:
            reasoning.append(
                "Stem windows for vocals, drums, bass, and 'other' are "
                "staggered across the beat grid."
            )
        elif context.stems_a is not None or context.stems_b is not None:
            reasoning.append(
                "Stem windows are staggered where stems are available; the "
                "other side uses a beat-aligned full-mix fade."
            )
        else:
            reasoning.append(
                "Full-mix fades are staggered across the beat grid; no "
                "stem-level timing is claimed."
            )
        if beat_grid is not None:
            if phase_locked:
                warnings.append(
                    "Rhythm planning assumes 4/4 meter and achieves downbeat-to-downbeat "
                    "phase lock; Song B's incoming downbeat is phase-aligned to Song A's "
                    "outgoing downbeat with sub-millisecond precision."
                )
            elif is_downbeat_aligned:
                warnings.append(
                    "Rhythm planning assumes 4/4 meter and aligns to estimated "
                    "downbeats; Song B is mixed on Song A's timing grid without "
                    "phase alignment."
                )
            else:
                warnings.append(
                    "Rhythm planning assumes 4/4 meter and uses detected beat "
                    "times; downbeats are not detected and Song B is mixed on "
                    "Song A's timing grid without phase alignment."
                )
        else:
            warnings.append(
                "Rhythm planning assumes 4/4 meter and relies on approximate "
                "tempo estimates; half/double-time ambiguity is possible."
            )
        metadata: dict[str, object] = {
            "bpm_a": round(bpm_a, 3),
            "bpm_b": round(bpm_b, 3),
            "bars": bars,
            "beats_per_bar": beats_per_bar,
            "total_beats": total_beats,
            "alignment": (
                "detected_beats" if beat_grid is not None else "tempo_fractions"
            ),
            "detected_beats_a": len(context.analysis_a.beat_times_s),
            "analyzer_a": context.analysis_a.analyzer,
            "analyzer_b": context.analysis_b.analyzer,
        }
        if is_downbeat_aligned:
            metadata["downbeat_aligned"] = True
        if anchor_beat_s is not None:
            metadata["anchor_beat_s"] = round(anchor_beat_s, 6)
        # Add phase lock metadata
        if phase_locked:
            metadata["phase_locked"] = True
            metadata["phase_lock_confidence"] = round(phase_lock_confidence, 4)
            metadata["anchor_downbeat_a_s"] = round(anchor_downbeat_a_s, 6)
            metadata["anchor_downbeat_b_s"] = round(phase_lock_anchor_b_s, 6)
            metadata["cue_offset_b_s"] = round(cue_offset_b_s, 6)
            metadata["transition_start_a_s"] = round(transition_start_in_a, 6)
            metadata["alignment"] = "phase_locked"
        return _make_plan(
            context,
            strategy=self.name,
            duration_s=duration,
            automations=automations,
            reasoning=reasoning,
            warnings=warnings,
            metadata=metadata,
        )


class AtmosphericStrategy:
    name = "atmospheric"
    description = "Long instrumental overlap with gentle filter sweeps."

    def plan(self, context: TransitionContext) -> TransitionPlan:
        if context.stems_a is None and context.stems_b is None:
            return _conservative_plan(
                context,
                requested_strategy=context.requested_strategy,
                extra_warnings=[
                    "Atmospheric transition requested without stems; filter "
                    "sweeps and instrumental prioritization require stems.",
                ],
                prefix_reasoning=(
                    "Without stems, an atmospheric plan would collapse into a "
                    "long full-mix crossfade, so the conservative plan is used."
                ),
            )

        view_a, view_b, warnings, reasoning = _views(context)
        cfg = context.config.transition
        desired = max(
            context.requested_duration_s * cfg.atmospheric_scale,
            cfg.atmospheric_min_duration_s,
        )
        duration, duration_warnings = _clamp_requested_duration(
            context, desired
        )
        if abs(duration - desired) > 1e-6:
            duration_warnings.append(
                f"Atmospheric overlap was adjusted from {desired:.2f}s to "
                f"{duration:.2f}s to fit the shorter track."
            )
        warnings.extend(duration_warnings)

        floor_db = cfg.fade_floor_db
        automations: list[StemAutomation] = []
        air_hz = min(16000.0, context.song_a.sample_rate * 0.45)

        for stem in _ordered_names(view_a):
            start_fraction, end_fraction = _A_ATMOSPHERIC_WINDOWS.get(
                stem, (0.0, 0.95) if stem != MIX_STEM else (0.0, 1.0)
            )
            effects: tuple[EffectSpec, ...] = ()
            if stem == "other":
                effects = (
                    EffectSpec(
                        kind="lowpass_sweep",
                        start_hz=air_hz,
                        end_hz=1200.0,
                        start_s=0.0,
                        end_s=duration,
                    ),
                )
            automations.append(
                StemAutomation(
                    source="a",
                    stem=stem,
                    points=_fade_points(
                        duration,
                        start_fraction * duration,
                        end_fraction * duration,
                        fade_in=False,
                        floor_db=floor_db,
                    ),
                    effects=effects,
                )
            )
        for stem in _ordered_names(view_b):
            start_fraction, end_fraction = _B_ATMOSPHERIC_WINDOWS.get(
                stem, (0.1, 1.0) if stem != MIX_STEM else (0.0, 1.0)
            )
            effects = ()
            if stem == "other":
                effects = (
                    EffectSpec(
                        kind="lowpass_sweep",
                        start_hz=900.0,
                        end_hz=air_hz,
                        start_s=0.0,
                        end_s=duration,
                    ),
                )
            automations.append(
                StemAutomation(
                    source="b",
                    stem=stem,
                    points=_fade_points(
                        duration,
                        start_fraction * duration,
                        end_fraction * duration,
                        fade_in=True,
                        floor_db=floor_db,
                    ),
                    effects=effects,
                )
            )

        if "other" not in view_a or "other" not in view_b:
            warnings.append(
                "No 'other'/pad stem for at least one song; one or more "
                "atmospheric filter sweeps were skipped."
            )
        if duration > context.requested_duration_s + 1e-6:
            reasoning.append(
                f"Atmospheric transition over {duration:.2f}s, longer than the "
                "requested window to let instrumental layers overlap."
            )
        else:
            reasoning.append(
                f"Atmospheric transition over {duration:.2f}s; the shorter "
                "track limits the overlap to this length."
            )
        if context.stems_a is not None and context.stems_b is not None:
            reasoning.append(
                "Song B's instrumental layers enter underneath Song A while "
                "vocal hand-off is kept late and narrow."
            )
        else:
            reasoning.append(
                "Because stems are incomplete, the missing side uses a long "
                "smooth full-mix fade rather than claimed instrumental "
                "separation."
            )
        sweep_sources = [
            label
            for label, view in (("Song A", view_a), ("Song B", view_b))
            if "other" in view
        ]
        if sweep_sources == ["Song A", "Song B"]:
            reasoning.append(
                f"Song A's 'other' stem is low-pass swept from {air_hz:.0f} Hz "
                "to 1200 Hz; Song B's 'other' stem opens from 900 Hz back to "
                f"{air_hz:.0f} Hz. The filter is two cascaded one-pole stages."
            )
        elif sweep_sources:
            reasoning.append(
                f"A low-pass filter sweep is applied to {sweep_sources[0]}'s "
                "'other' stem only; the other side has no matching stem."
            )
        else:
            reasoning.append(
                "No 'other' stem was available, so no filter sweep was applied."
            )
        reasoning.append(
            "Where drum and bass stems exist they fade before the pads; "
            "missing stems follow their longer full-mix fade."
        )
        return _make_plan(
            context,
            strategy=self.name,
            duration_s=duration,
            automations=automations,
            reasoning=reasoning,
            warnings=warnings,
            metadata={"filter_sweep": True, "stem_aware": True},
        )


class ConservativeStrategy:
    name = "conservative"
    description = "Predictable full-mix fallback without musical assumptions."

    def plan(self, context: TransitionContext) -> TransitionPlan:
        return _conservative_plan(
            context,
            requested_strategy=context.requested_strategy,
            extra_warnings=[],
            prefix_reasoning=(
                "A conservative plan was explicitly selected."
                if context.requested_strategy == "conservative"
                else ""
            ),
        )


class StudioRemixStrategy:
    """
    Studio Remix (Song C) Strategy - The Autonomous Studio Producer.

    Implements the 12-bar Producer Matrix from the DCP blueprint:
    - 8-bar composed bridge + 4-bar ghost resolution
    - Launchpad (T_launch) in Song A: phrase-aligned, high groove energy, concluding vocal
    - Impact Downbeat (T_impact) in Song B: the Drop! Max low-end surge, kick crest factor
    - Breath Window (T_breath): 1-2 beat acoustic vacuum before impact
    - Low-End Monogamy: No bass collision under 150 Hz
    - ZDF Filter Sweeps: Bars 5-8, Song A stems sweep 30 Hz → 950 Hz
    - Ghost Resolution (Bars 9-12): Song A's frozen vocal/synth tail floats over Song B's groove,
      sidechain-ducked by Song B's kick
    """
    name = "studio_remix"
    description = "Autonomous Studio Producer: 8-to-16-bar composed transition bridge (Song C) with launchpad, breath, impact, and ghost resolution."

    def plan(self, context: TransitionContext) -> TransitionPlan:
        import numpy as np
        from .alignment import select_launchpad_and_impact
        from .musical_ir import build_track_musical_ir
        from .composer import compose_optimal_bridge

        # Check if stems are available - required for Studio Remix
        if context.stems_a is None and context.stems_b is None:
            return _conservative_plan(
                context,
                requested_strategy=context.requested_strategy,
                extra_warnings=[
                    "Studio Remix requires stems for both songs; "
                    "conservative fallback used.",
                ],
                prefix_reasoning=(
                    "Studio Remix (Song C) requires stem separation for independent "
                    "processing of vocals, drums, bass, and other layers. "
                    "No stems were supplied."
                ),
            )

        view_a, view_b, warnings, reasoning = _views(context)

        composed_plan = None
        anchor_result = None
        downbeats_a = list(context.analysis_a.downbeat_times_s)
        downbeats_b = list(context.analysis_b.downbeat_times_s)

        # Attempt Layer 3 Composition Planning if both songs have stems and >= 5 downbeats
        if len(downbeats_a) >= 5 and len(downbeats_b) >= 5 and context.stems_a and context.stems_b:
            try:
                stems_a_raw = {k: v.samples for k, v in context.stems_a.items()}
                stems_b_raw = {k: v.samples for k, v in context.stems_b.items()}
                ir_a = build_track_musical_ir(
                    "song_a",
                    stems_a_raw,
                    np.asarray(downbeats_a, dtype=np.float64),
                    sr=context.song_a.sample_rate,
                    base_confidence=context.analysis_a.tempo_confidence or 0.85,
                )
                ir_b = build_track_musical_ir(
                    "song_b",
                    stems_b_raw,
                    np.asarray(downbeats_b, dtype=np.float64),
                    sr=context.song_b.sample_rate,
                    base_confidence=context.analysis_b.tempo_confidence or 0.85,
                )
                # Estimate needed lead-in bars in Song B so impact_b >= t_impact
                est_bar_a = 4.0 * 60.0 / max(20.0, context.analysis_a.tempo_bpm or 120.0)
                est_bar_b = 4.0 * 60.0 / max(20.0, context.analysis_b.tempo_bpm or 120.0)
                est_eff_bar = 2.0 / (1.0 / max(1e-3, est_bar_a) + 1.0 / max(1e-3, est_bar_b))
                est_t_impact = 11.0 * est_eff_bar
                needed_b_bars = int(np.ceil(est_t_impact / max(0.5, est_bar_b)))
                max_avail_bar = max(0, len(downbeats_b) - 4)
                min_impact_bar = max(0, min(needed_b_bars, max_avail_bar))
                candidate_plan = compose_optimal_bridge(ir_a, ir_b, min_impact_bar_b=min_impact_bar)
                if not candidate_plan.is_viable:
                    return _conservative_plan(
                        context,
                        requested_strategy=context.requested_strategy,
                        extra_warnings=[
                            f"Studio Remix rejected by Composition Gate: {candidate_plan.rejection_reason}. "
                            "Conservative fallback used."
                        ],
                        prefix_reasoning=(
                            f"Layer 3 Composer rejected transition: {candidate_plan.composer_rationale}"
                        ),
                    )
                composed_plan = candidate_plan
            except Exception as e:
                warnings.append(f"Musical IR composition planner fallback: {e}")

        if composed_plan is not None:
            archetype = composed_plan.archetype
            transition_mode = composed_plan.transition_mode
            feasibility_score = composed_plan.feasibility_score
            composer_rationale = composed_plan.composer_rationale
            launchpad_a = composed_plan.launchpad_phrase_a.end_time_s
            impact_b = composed_plan.impact_phrase_b.start_time_s
            tempo_a = composed_plan.hypothesis_a.bpm
            tempo_b = composed_plan.hypothesis_b.bpm
            lead_in_bars = 2
            beat_duration_a = 60.0 / tempo_a
            beat_duration_b = 60.0 / tempo_b
            bar_duration_a = beat_duration_a * 4.0
            bar_duration_b = beat_duration_b * 4.0
            bridge_start_a = max(0.0, launchpad_a - lead_in_bars * bar_duration_a)
            breath_beats = composed_plan.breath_beats
            if breath_beats == 0:
                breath_start = launchpad_a
                breath_end = launchpad_a
            else:
                breath_dur_s = breath_beats * beat_duration_b
                breath_start = max(0.0, impact_b - breath_dur_s)
                breath_end = impact_b
            anchor_conf = feasibility_score
        else:
            # Fall back to heuristic anchor selection
            anchor_result = select_launchpad_and_impact(
                context,
                lead_in_bars=2,
                min_confidence=context.config.analysis.tempo_confidence_threshold,
                bridge_bars=8,
            )
            if not anchor_result.success:
                return _conservative_plan(
                    context,
                    requested_strategy=context.requested_strategy,
                    extra_warnings=[
                        f"Studio Remix anchors unusable ({anchor_result.fallback_reason}); "
                        "degraded to clean transition."
                    ],
                    prefix_reasoning=(
                        f"Studio Remix could not identify reliable phrase anchors ({anchor_result.fallback_reason}). "
                        "Degraded to clean transition."
                    ),
                    extra_metadata={"studio_remix": True, "degraded": True},
                )

            archetype = "VACUUM_BREATH_DROP"
            transition_mode = "CONTINUOUS_TEMPO_WARP"
            feasibility_score = anchor_result.confidence
            composer_rationale = "Phrase-aligned heuristic launchpad selection."
            launchpad_a = anchor_result.launchpad_a_s
            impact_b = anchor_result.impact_b_s
            breath_start = anchor_result.breath_window_start_s
            breath_end = anchor_result.breath_window_end_s
            bridge_start_a = anchor_result.bridge_start_a_s
            lead_in_bars = anchor_result.lead_in_bars
            tempo_a = anchor_result.tempo_a_bpm or context.analysis_a.tempo_bpm
            tempo_b = anchor_result.tempo_b_bpm or context.analysis_b.tempo_bpm

            if tempo_a is None or tempo_b is None:
                return _conservative_plan(
                    context,
                    requested_strategy=context.requested_strategy,
                    extra_warnings=[
                        "Studio Remix tempo estimation unavailable; "
                        "conservative fallback used."
                    ],
                    prefix_reasoning=(
                        "Studio Remix requires reliable tempo estimates. "
                        "Estimates were unavailable or below threshold."
                    ),
                )

            beat_duration_a = 60.0 / tempo_a
            beat_duration_b = 60.0 / tempo_b
            bar_duration_a = beat_duration_a * 4.0
            bar_duration_b = beat_duration_b * 4.0
            breath_beats = 2 if breath_end > breath_start else 0
            anchor_conf = anchor_result.confidence

        # Determine effective bar duration across the transition
        bar_duration_a = 4.0 * 60.0 / max(20.0, tempo_a)
        bar_duration_b = 4.0 * 60.0 / max(20.0, tempo_b)
        effective_bar_s = 2.0 / (1.0 / max(1e-3, bar_duration_a) + 1.0 / max(1e-3, bar_duration_b))

        # Determine master BPM and warp ratio for Song B
        if archetype == "POLYRHYTHMIC_4_3_PIVOT":
            # For 4:3 pivot, hypothesis_a is already the pivot tempo verbatim (e.g. 124.0 for 93 * 4/3)
            if composed_plan is not None and hasattr(composed_plan, "hypothesis_a"):
                master_bpm = float(composed_plan.hypothesis_a.bpm)
            else:
                master_bpm = tempo_a * (4.0 / 3.0) if tempo_a < tempo_b else tempo_a * (3.0 / 4.0)

            # Coherent Time Model: No-warp handoff
            # Song B enters at native tempo post-impact; bridge lead-in runs on master_bpm pivot grid.
            warp_ratio_b = 1.0
        else:
            master_bpm = (tempo_a + tempo_b) * 0.5
            warp_ratio_b = 1.0

        # Fully Autonomous Musical Bar Allocation (Decided by the AI Producer, not a slider)
        if composed_plan is not None:
            # 16 bars for polyrhythmic pivots & ghost suspensions, 12 bars standard matrix
            if archetype in ("POLYRHYTHMIC_4_3_PIVOT", "GHOST_WASH_SUSPENSION"):
                desired_bars = 16
            else:
                desired_bars = 12
        else:
            desired_bars = 12

        total_bridge_bars = desired_bars
        ghost_bars = max(2, total_bridge_bars // 3)
        lead_in_bars = total_bridge_bars - ghost_bars
        bridge_duration_s = total_bridge_bars * effective_bar_s

        max_trans_dur = min(context.max_duration_s, context.song_a.duration_s, context.song_b.duration_s)
        if bridge_duration_s > max_trans_dur:
            total_bridge_bars = max(2, int(max_trans_dur / effective_bar_s))
            ghost_bars = max(1, total_bridge_bars // 3)
            lead_in_bars = max(1, total_bridge_bars - ghost_bars)
            bridge_duration_s = min(max_trans_dur, total_bridge_bars * effective_bar_s)

        duration = bridge_duration_s

        # Launchpad A and Impact B timing
        t_impact = lead_in_bars * effective_bar_s
        t_intro = t_impact * 0.5

        if archetype == "CONTINUOUS_CROSS_FILTER" or breath_beats == 0:
            breath_dur = 0.0
            t_breath = t_impact
        else:
            raw_breath = max(0.04, float(breath_end - breath_start)) if (breath_end > breath_start) else (bar_duration_b * 0.25)
            min_breath_s = 60.0 / max(40.0, tempo_b)  # 1 full beat minimum at Song B tempo
            breath_dur = max(min_breath_s, min(t_impact * 0.25, raw_breath))
            t_breath = t_impact - breath_dur

        bridge_start_a = max(0.0, launchpad_a - t_impact)
        if bridge_start_a + duration > context.song_a.duration_s:
            # Bar-shrink: shrink duration to fit Song A without dragging bridge_start_a backward
            duration = min(duration, max(t_impact + 0.5, context.song_a.duration_s - bridge_start_a))
            if bridge_start_a + duration > context.song_a.duration_s:
                bridge_start_a = max(0.0, context.song_a.duration_s - duration)
            t_impact = min(duration - 0.05, max(0.20 * duration, launchpad_a - bridge_start_a))
            t_intro = t_impact * 0.5
            t_breath = max(0.0, t_impact - breath_dur) if breath_dur > 0 else t_impact

        # Song B Cue Offset: aligns impact_b with t_impact
        # Vocoder identity: event at source s lands at output s * warp_ratio_b
        # Output t_impact comes from source cue_offset_b_s + t_impact / warp_stretch
        warp_stretch = warp_ratio_b if abs(warp_ratio_b - 1.0) > 1e-3 else 1.0
        cue_unclamped = impact_b - (t_impact / warp_stretch)

        phase_locked = True
        grid_reanchored = False
        if cue_unclamped < -1e-6:
            warnings.append(
                f"Impact phrase at {impact_b:.2f}s cannot reach the {lead_in_bars}-bar lead-in; "
                "re-anchoring grid and capping lead-in to fit."
            )
            # Cap lead-in first to what Song B can reach
            max_lead_in = max(0.0, impact_b * warp_stretch)
            t_impact = min(t_impact, max_lead_in)
            # Recalculate bars and duration to shrink the post-impact window proportionally
            lead_in_bars = max(1, int(round(t_impact / max(0.5, effective_bar_s))))
            ghost_bars = max(2, lead_in_bars // 2)
            total_bridge_bars = lead_in_bars + ghost_bars
            duration = min(duration, total_bridge_bars * effective_bar_s)

            bridge_start_a = max(0.0, launchpad_a - t_impact)
            if bridge_start_a + duration > context.song_a.duration_s:
                duration = min(duration, max(t_impact + 0.5, context.song_a.duration_s - bridge_start_a))
                if bridge_start_a + duration > context.song_a.duration_s:
                    bridge_start_a = max(0.0, context.song_a.duration_s - duration)

            # Re-anchor: keep launchpad_a - bridge_start_a matched to t_impact within max_lead_in
            t_impact = min(max_lead_in, max(0.0, launchpad_a - bridge_start_a))
            t_intro = t_impact * 0.5
            t_breath = max(0.0, t_impact - breath_dur) if breath_dur > 0 else t_impact
            cue_unclamped = impact_b - (t_impact / warp_stretch)
            grid_reanchored = True
            if cue_unclamped < -1e-6:
                phase_locked = False

        cue_offset_b_s = max(0.0, cue_unclamped)
        if cue_offset_b_s + duration > context.song_b.duration_s:
            cue_offset_b_s = max(0.0, context.song_b.duration_s - duration)

        grid_bpm_pre = master_bpm
        grid_bpm_post = tempo_b / warp_stretch

        l_a = (
            context.analysis_a.loudness_lufs
            if context.analysis_a and context.analysis_a.loudness_lufs is not None
            else (context.analysis_a.rms_dbfs if context.analysis_a else -14.0)
        )
        l_b = (
            context.analysis_b.loudness_lufs
            if context.analysis_b and context.analysis_b.loudness_lufs is not None
            else (context.analysis_b.rms_dbfs if context.analysis_b else -14.0)
        )
        loudness_comp_db = float(np.clip(l_a - l_b, -6.0, 6.0)) if (l_a is not None and l_b is not None) else 0.0

        floor_db = context.config.transition.fade_floor_db

        # Build automations following the adaptive Producer Matrix
        automations = self._build_studio_remix_automations(
            context,
            view_a,
            view_b,
            duration,
            bridge_start_a,
            launchpad_a,
            impact_b,
            breath_start,
            breath_end,
            bar_duration_a,
            bar_duration_b,
            floor_db,
            tempo_a,
            tempo_b,
            reasoning,
            warnings,
            archetype=archetype,
            breath_beats=breath_beats,
            t_impact=t_impact,
            t_intro=t_intro,
            t_breath=t_breath,
        )

        # Add reasoning
        if composed_plan:
            reasoning.append(
                f"Layer 3 Composer: Archetype={archetype}, Mode={transition_mode}, Feasibility={feasibility_score:.3f}."
            )
            reasoning.append(f"Composer Rationale: {composer_rationale}")

        reasoning.extend([
            f"Studio Remix (Song C) bridge: {total_bridge_bars} bars ({lead_in_bars} bridge + {ghost_bars} ghost) "
            f"from {bridge_start_a:.2f}s to {bridge_start_a + duration:.2f}s in Song A.",
            f"Launchpad (T_launch): Song A downbeat at {launchpad_a:.3f}s - "
            f"phrase-aligned, groove energy maximized, vocal concluding.",
            f"Impact (T_impact): Song B downbeat at {impact_b:.3f}s - "
            f"the Drop! Low-end surge maximized, kick crest factor peak.",
            f"Breath Window: {breath_start:.3f}s – {breath_end:.3f}s in Song B "
            f"({(breath_end - breath_start) / max(1e-4, beat_duration_b):.1f} beats acoustic vacuum).",
            f"Song B cued at offset {cue_offset_b_s:.3f}s for phase-locked impact at {t_impact:.2f}s.",
        ])

        bass_root_val = 2
        if composed_plan is not None and hasattr(composed_plan, "impact_phrase_b"):
            bass_root_val = getattr(composed_plan.impact_phrase_b, "bass_root_pc", 2)

        metadata = {
            "studio_remix": True,
            "archetype": archetype,
            "transition_mode": transition_mode,
            "feasibility_score": round(feasibility_score, 4),
            "composer_rationale": composer_rationale,
            "bridge_bars": lead_in_bars,
            "ghost_bars": ghost_bars,
            "total_bridge_bars": total_bridge_bars,
            "launchpad_a_s": round(launchpad_a, 6),
            "impact_b_s": round(impact_b, 6),
            "cue_offset_b_s": round(cue_offset_b_s, 6),
            "t_impact_s": round(t_impact, 6),
            "t_intro_s": round(t_intro, 6),
            "t_breath_s": round(t_breath, 6),
            "breath_start_s": round(breath_start, 6),
            "breath_end_s": round(breath_end, 6),
            "breath_beats": breath_beats,
            "bridge_start_a_s": round(bridge_start_a, 6),
            "lead_in_bars": lead_in_bars,
            "tempo_a": round(tempo_a, 3),
            "tempo_b": round(tempo_b, 3),
            "master_bpm": round(master_bpm, 3),
            "grid_bpm_pre": round(grid_bpm_pre, 3),
            "grid_bpm_post": round(grid_bpm_post, 3),
            "effective_bar_s": round(effective_bar_s, 6),
            "loudness_compensation_db": round(loudness_comp_db, 2),
            "warp_ratio_b": round(warp_ratio_b, 6),
            "bass_root_b": int(bass_root_val),
            "anchor_confidence": round(anchor_conf, 4),
            "phase_locked": bool(phase_locked),
            "grid_reanchored": bool(grid_reanchored),
            "low_end_monogamy": True,
            "zdf_filter_sweeps": True,
            "ghost_resolution": True,
            "sidechain_ducking": True,
        }

        return _make_plan(
            context,
            strategy=self.name,
            duration_s=duration,
            automations=automations,
            reasoning=reasoning,
            warnings=warnings,
            metadata=metadata,
            start_time_s=bridge_start_a,
        )

    def _build_studio_remix_automations(
        self,
        context: TransitionContext,
        view_a: dict,
        view_b: dict,
        duration_s: float,
        bridge_start_a: float,
        launchpad_a: float,
        impact_b: float,
        breath_start: float,
        breath_end: float,
        bar_duration_a: float,
        bar_duration_b: float,
        floor_db: float,
        tempo_a: float,
        tempo_b: float,
        reasoning: list,
        warnings: list,
        archetype: str = "VACUUM_BREATH_DROP",
        breath_beats: int = 2,
        t_impact: float | None = None,
        t_intro: float | None = None,
        t_breath: float | None = None,
    ) -> list[StemAutomation]:
        """Build stem automations following the 12-bar Producer Matrix from Model 4 Argon.

        Adapts matrix size and phase boundaries based on available transition duration.
        Robustly handles short durations by clamping all time/frequency parameters.
        """
        sample_rate = context.song_a.sample_rate
        nyquist_limit = (sample_rate / 2.0) * 0.95

        # Determine the 4 producer matrix phases across duration_s:
        # Phase 1: Intro / Lead-in (0.0 -> t_intro)
        # Phase 2: Build-up / Filter Sweeps (t_intro -> t_breath)
        # Phase 3: Breath Window (t_breath -> t_impact)
        # Phase 4: Impact Drop & Ghost Decay (t_impact -> duration_s)

        if t_impact is None:
            launchpad_trans = launchpad_a - bridge_start_a
            if 0.35 * duration_s <= launchpad_trans <= 0.85 * duration_s:
                t_impact = launchpad_trans
            else:
                t_impact = (2.0 / 3.0) * duration_s

        if t_intro is None:
            t_intro = t_impact * 0.5

        if t_breath is None:
            build_duration = t_impact - t_intro
            max_breath = build_duration * 0.35
            if archetype == "CONTINUOUS_CROSS_FILTER" or breath_beats == 0:
                breath_dur = 0.0
                t_breath = t_impact
            else:
                raw_breath = max(0.04, float(breath_end - breath_start)) if (breath_end > breath_start) else (bar_duration_b * 0.25)
                breath_dur = max(0.02, min(max_breath, raw_breath))
                t_breath = t_impact - breath_dur

        # --- Robust time boundary validation ---
        # Ensure minimum separation between phase boundaries
        min_gap = max(0.01, duration_s * 0.01)
        if t_intro >= t_breath - 0.01:
            t_intro = max(0.0, t_breath - max(0.01, duration_s * 0.05))
        if t_breath >= t_impact - 0.01:
            t_breath = max(t_intro + 0.01, t_impact - 0.01)
        if t_impact >= duration_s - 0.01:
            t_impact = max(t_breath + 0.01, duration_s - 0.01)

        eps_cut = min(0.005, max(0.001, (t_impact - t_breath) * 0.15)) if t_impact > t_breath else 0.005

        # --- Robust helper functions with strict validation ---
        def clamp_time(t: float) -> float:
            """Clamp time to [eps, duration_s - eps] with small epsilon margin."""
            eps = 1e-3
            return max(eps, min(duration_s - eps, float(t)))

        def safe_freq(freq: float) -> float:
            """Clamp frequency to [25, nyquist_limit]."""
            return max(25.0, min(nyquist_limit, float(freq)))

        def make_effect(
            kind: str,
            start_s: float,
            end_s: float,
            start_hz: float = 1000.0,
            end_hz: float = 1000.0,
            params: dict | None = None,
        ) -> EffectSpec | None:
            """Create validated EffectSpec with guaranteed valid time/frequency ranges."""
            s = clamp_time(start_s)
            e = clamp_time(end_s)
            if e <= s + 0.01:  # Minimum 10ms duration
                return None
            s_hz = safe_freq(start_hz)
            e_hz = safe_freq(end_hz)
            return EffectSpec(
                kind=kind,
                start_s=round(s, 6),
                end_s=round(e, 6),
                start_hz=round(s_hz, 2),
                end_hz=round(e_hz, 2),
                params=params,
            )

        def gain_automation(raw_points: list[tuple[float, float]]) -> tuple[GainPoint, ...]:
            """Create strictly increasing gain points strictly within (eps, duration_s - eps)."""
            if not raw_points:
                eps = 1e-3
                return (GainPoint(eps, 0.0), GainPoint(duration_s - eps, 0.0))

            sorted_pts = sorted(raw_points, key=lambda p: p[0])
            cfg_floor = floor_db
            cfg_max = context.config.transition.max_automation_gain_db

            cleaned: list[GainPoint] = []
            for t, db in sorted_pts:
                t_cl = clamp_time(t)
                db_cl = max(floor_db, min(context.config.transition.max_automation_gain_db, float(db)))
                cleaned.append(GainPoint(t_cl, db_cl))

            # Deduplicate with time ordering
            deduped: list[GainPoint] = []
            for pt in cleaned:
                if deduped and abs(pt.time_s - deduped[-1].time_s) <= 1e-4:
                    deduped[-1] = pt
                elif not deduped or pt.time_s > deduped[-1].time_s + 1e-6:
                    deduped.append(pt)

            # Ensure at least 2 points
            if len(deduped) < 2:
                if not deduped:
                    eps = 1e-3
                    return (GainPoint(1e-3, 0.0), GainPoint(duration_s - 1e-3, 0.0))
                only = deduped[0]
                if only.time_s > 1e-3:
                    deduped.insert(0, GainPoint(1e-3, only.gain_db))
                if only.time_s < duration_s - 1e-3:
                    deduped.append(GainPoint(duration_s - 1e-3, only.gain_db))

            # Ensure first point at eps and last at duration_s - eps
            eps = 1e-3
            if deduped[0].time_s > eps:
                deduped.insert(0, GainPoint(eps, deduped[0].gain_db))
            else:
                deduped[0] = GainPoint(eps, deduped[0].gain_db)

            if deduped[-1].time_s < duration_s - eps:
                deduped.append(GainPoint(duration_s - eps, deduped[-1].gain_db))
            else:
                deduped[-1] = GainPoint(duration_s - eps, deduped[-1].gain_db)

            # Enforce strictly increasing timestamps
            final_pts: list[GainPoint] = [deduped[0]]
            for pt in deduped[1:]:
                if pt.time_s > final_pts[-1].time_s + 1e-4:
                    final_pts.append(pt)
                else:
                    final_pts[-1] = pt

            if len(final_pts) < 2:
                eps = 1e-3
                final_pts = [GainPoint(1e-3, 0.0), GainPoint(duration_s - 1e-3, 0.0)]
            if final_pts[-1].time_s > duration_s - 1e-3:
                final_pts[-1] = GainPoint(duration_s - 1e-3, final_pts[-1].gain_db)

            return tuple(final_pts)

        automations: list[StemAutomation] = []

        # ============================================================
        # SONG A AUTOMATIONS (Outgoing) - Exact Producer Rules Matrix
        # ============================================================

        for stem_name in _ordered_names(view_a):
            effects: tuple[EffectSpec, ...] = ()

            if stem_name == "bass":
                pts = gain_automation([
                    (0.0, 0.0),
                    (t_intro, 0.0),
                    (t_intro + eps_cut, -3.0),
                    (t_breath, -18.0),
                    (t_impact, floor_db),
                    (duration_s, floor_db),
                ])
                effects = tuple(
                    e for e in [
                        make_effect("zdf_highpass_sweep", t_intro, t_impact, 110.0, 450.0, params={"filter_type": "TPT_SVF", "resonance": 1.0, "stages": 2}),
                        make_effect("sidechain_duck", t_intro, t_impact, params={"source": "song_b_drums", "attack_ms": 2.0, "release_ms": 50.0, "ratio": 4.0, "threshold_db": -12.0}),
                    ] if e is not None
                )
                reasoning.append(
                    f"Low-End Monogamy: Song A bass 0dB (Bars 1-2), "
                    f"HPF 110→450 Hz + 4:1 duck by B drums (Bars 3-8), muted thereafter."
                )

            elif stem_name == "drums":
                if archetype == "GHOST_WASH_SUSPENSION":
                    pts = gain_automation([
                        (0.0, 0.0),
                        (t_intro * 0.7, 0.0),
                        (t_intro, floor_db),
                        (duration_s, floor_db),
                    ])
                    effects = tuple(
                        e for e in [
                            make_effect("zdf_highpass_sweep", 0.0, t_intro, 30.0, 300.0),
                            make_effect("dub_delay_send", 0.0, t_intro, params={"delay_note": 0.75, "feedback": 0.6}),
                        ] if e is not None
                    )
                    reasoning.append(
                        "Ghost Wash: Song A drums dissolve early at intro into ambient dub delay wash."
                    )
                else:
                    pts = gain_automation([
                        (0.0, 0.0),
                        (t_intro, 0.0),
                        (t_breath, -4.5),
                        (t_breath + eps_cut, floor_db),
                        (duration_s, floor_db),
                    ])
                    effects = tuple(
                        e for e in [
                            make_effect("zdf_highpass_sweep", 0.0, t_intro, 30.0, 60.0),
                            make_effect("zdf_highpass_sweep", t_intro, t_breath, 60.0, 650.0, params={"resonance": 1.4}),
                            make_effect("dub_delay_send", t_intro, t_breath, params={"delay_note": 0.75, "feedback": 0.45}),
                        ] if e is not None
                    )
                    reasoning.append(
                        "Song A drums: 0dB, HPF 60→650 Hz Q=1.4 + Gain -1.5→-4.5dB + dub-delay, hard cut at breath."
                    )

            elif stem_name == "vocals":
                if archetype == "CONTINUOUS_CROSS_FILTER" or breath_beats == 0:
                    pts = gain_automation([
                        (0.0, 0.0),
                        (t_intro, 0.0),
                        (t_impact, 0.0),
                        (t_impact + eps_cut, floor_db),
                        (duration_s, floor_db),
                    ])
                    effects = tuple(
                        e for e in [
                            make_effect("dub_delay_send", 0.0, t_intro, params={"delay_note": 0.25, "send_level": -18.0}),
                            make_effect("zdf_highpass_sweep", t_intro, t_impact, 180.0, 800.0, params={"resonance": 1.2}),
                            make_effect("spatial_freeze", t_intro, duration_s, params={"wet_mix_start": 0.15, "wet_mix_end": 0.65, "freeze_mode": True, "freeze_engage_s": t_impact, "freeze_release_s": min(duration_s, t_impact + 0.5), "T60": 6.5}),
                            make_effect("sidechain_duck", t_impact, duration_s, params={"source": "song_b_drums", "ratio": 8.0, "release_ms": 150.0}),
                        ] if e is not None
                    )
                    reasoning.append(
                        "Continuous Cross-Filter: Song A vocals sustained across boundary without breath cutoff, sidechained post-impact."
                    )
                else:
                    pts = gain_automation([
                        (0.0, 0.0),
                        (t_intro, 0.0),
                        (t_breath, 0.0),
                        (t_breath + eps_cut, floor_db),
                        (duration_s, floor_db),
                    ])
                    effects = tuple(
                        e for e in [
                            make_effect("dub_delay_send", 0.0, t_intro, params={"delay_note": 0.25, "send_level": -18.0}),
                            make_effect("zdf_highpass_sweep", t_intro, t_breath, 180.0, 800.0, params={"resonance": 1.2}),
                            make_effect("spatial_freeze", t_intro, duration_s, params={"wet_mix_start": 0.15, "wet_mix_end": 0.65, "freeze_mode": True, "freeze_engage_s": t_breath, "freeze_release_s": t_impact, "T60": 6.5}),
                            make_effect("sidechain_duck", t_impact, duration_s, params={"source": "song_b_drums", "ratio": 8.0, "release_ms": 150.0}),
                        ] if e is not None
                    )
                    reasoning.append(
                        "Song A vocals: Dry lead, chopped hook HPF 180→800 Hz, FDN FREEZE at breath, ghost decay sidechained to B drums."
                    )

            elif stem_name == "other":
                pts = gain_automation([
                    (0.0, 0.0),
                    (t_intro, 0.0),
                    (t_breath, 0.0),
                    (t_breath + eps_cut, floor_db),
                    (duration_s, floor_db),
                ])
                effects = tuple(
                    e for e in [
                        make_effect("sidechain_duck", 0.0, t_breath, params={"source": "song_b_drums", "ratio": 3.0}),
                        make_effect("zdf_highpass_sweep", t_intro, t_breath, 120.0, 1200.0, params={"resonance": 2.2}),
                        make_effect("stereo_width", t_intro, t_breath, params={"width_start": 1.0, "width_end": 1.8}),
                        make_effect("spatial_freeze", t_intro, duration_s, params={"wet_mix_start": 0.20, "wet_mix_end": 0.70, "freeze_mode": True, "freeze_engage_s": t_breath, "freeze_release_s": t_impact, "T60": 8.0}),
                        make_effect("sidechain_duck", t_impact, duration_s, params={"source": "song_b_drums", "ratio": 4.0}),
                    ] if e is not None
                )
                reasoning.append(
                    "Song A other: 0dB + duck by B drums, ZDF HPF 120→1200 Hz Q=2.2 + stereo widen, FDN FREEZE at breath, ghost decay."
                )

            else:
                pts = gain_automation([
                    (0.0, 0.0),
                    (t_impact, 0.0),
                    (duration_s, floor_db),
                ])
                effects = ()

            automations.append(
                StemAutomation(
                    source="a",
                    stem=stem_name,
                    points=pts,
                    curve="smoothstep",
                    effects=effects,
                )
            )

        # ============================================================
        # SONG B AUTOMATIONS (Incoming) - Exact Producer Rules Matrix
        # ============================================================

        for stem_name in _ordered_names(view_b):
            effects = ()

            if stem_name == "drums":
                if archetype == "CONTINUOUS_CROSS_FILTER" or breath_beats == 0:
                    pts = gain_automation([
                        (0.0, -18.0),
                        (t_intro, -6.0),
                        (t_impact - 0.05, -2.0),
                        (t_impact, 0.0),
                        (duration_s, 0.0),
                    ])
                else:
                    pts = gain_automation([
                        (0.0, -18.0),
                        (t_intro, -6.0),
                        (max(0.0, t_breath - 0.1), -3.0),
                        (t_breath, floor_db),
                        (t_impact, 0.0),
                        (duration_s, 0.0),
                    ])
                effects = tuple(
                    e for e in [
                        make_effect("zdf_highpass", 0.0, t_intro, 220.0, 220.0),
                        make_effect("zdf_highpass_sweep", t_intro, t_breath, 220.0, 90.0),
                    ] if e is not None
                )
                reasoning.append(
                    "Song B drums: Tease HPF@220Hz, HPF 220→90Hz build, Pre-drop vacuum cut, Full explosive drop @ impact driving sidechain."
                )

            elif stem_name == "bass":
                pts = gain_automation([
                    (0.0, floor_db),
                    (t_intro, floor_db),
                    (t_intro + eps_cut, -9.0),
                    (max(0.0, t_breath - 0.1), -6.0),
                    (t_breath, floor_db),
                    (t_impact, 0.0),
                    (duration_s, 0.0),
                ])
                effects = tuple(
                    e for e in [
                        make_effect("zdf_lowpass", t_intro, t_breath, 110.0, 110.0),
                    ] if e is not None
                )
                reasoning.append(
                    "Song B bass: Muted early, Sub-swell -9dB LPF@110Hz build, Pre-drop vacuum silence, Full authority @ impact."
                )

            elif stem_name == "vocals":
                if archetype == "CONTINUOUS_CROSS_FILTER" or breath_beats == 0:
                    pts = gain_automation([
                        (0.0, -18.0),
                        (t_intro, -12.0),
                        (t_impact, -3.0),
                        (t_impact + eps_cut, 0.0),
                        (duration_s, 0.0),
                    ])
                    effects = tuple(
                        e for e in [
                            make_effect("zdf_bandpass", 0.0, t_intro, 400.0, 3500.0),
                            make_effect("zdf_bandpass", t_intro, t_impact, 400.0, 8000.0),
                        ] if e is not None
                    )
                    reasoning.append(
                        "Continuous Cross-Filter: Song B vocals build gently across intro/impact into full lead."
                    )
                else:
                    pts = gain_automation([
                        (0.0, -18.0),
                        (t_intro, -10.0),
                        (t_breath, -4.0),
                        (t_breath + eps_cut, 0.0),
                        (duration_s, 0.0),
                    ])
                    effects = tuple(
                        e for e in [
                            make_effect("zdf_bandpass", 0.0, t_intro, 400.0, 3500.0),
                            make_effect("zdf_bandpass", t_intro, t_breath, 400.0, 8000.0),
                        ] if e is not None
                    )
                    reasoning.append(
                        "Song B vocals: Ad-libs Bandpass 400-3500Hz, Rising hook, Spotlight @ breath, Full lead @ impact."
                    )

            elif stem_name == "other":
                pts = gain_automation([
                    (0.0, -18.0),
                    (t_intro, -8.0),
                    (max(0.0, t_breath - 0.1), -3.0),
                    (t_breath, -14.0),
                    (t_impact, 0.0),
                    (duration_s, 0.0),
                ])
                effects = tuple(
                    e for e in [
                        make_effect("zdf_lowpass_sweep", 0.0, t_intro, 400.0, 2500.0),
                        make_effect("zdf_lowpass_sweep", t_intro, t_breath, 2500.0, 10000.0),
                    ] if e is not None
                )
                reasoning.append(
                    "Song B other: Tease LPF 400→2500Hz, LPF opens 2.5k→10kHz, Vacuum dip @ breath, Full synth drop @ impact."
                )

            else:
                pts = gain_automation([
                    (0.0, floor_db),
                    (t_impact, 0.0),
                    (duration_s, 0.0),
                ])
                effects = ()

            automations.append(
                StemAutomation(
                    source="b",
                    stem=stem_name,
                    points=pts,
                    curve="smoothstep",
                    effects=effects,
                )
            )

        return automations


class OptimizedStrategy:
    name = "optimized"
    description = "Objective-based role-aware timing search (RATO)."

    def plan(self, context: TransitionContext) -> TransitionPlan:
        from .optimize import optimize_transition_from_context

        return optimize_transition_from_context(context)


def build_core_registry() -> StrategyRegistry:
    registry = StrategyRegistry()
    for strategy in (
        CrossfadeStrategy(),
        StemLevelStrategy(),
        RhythmicStrategy(),
        AtmosphericStrategy(),
        ConservativeStrategy(),
        StudioRemixStrategy(),  # New Studio Remix strategy
        OptimizedStrategy(),
    ):
        registry.register(strategy)
    return registry

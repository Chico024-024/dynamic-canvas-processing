"""Core transition strategies and the strategy registry."""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

from ..config import DCPConfig
from ..models import DCPError, MIX_STEM, AudioBuffer, StemSet
from .context import TransitionContext
from .plan import EffectSpec, GainPoint, StemAutomation, TransitionPlan

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
    candidates: list[tuple[float, int, float]] = []
    for index, beat in enumerate(beats):
        if beat <= 1e-9 or beat >= song_end - 1e-9:
            continue
        duration = song_end - beat
        if duration < minimum_s - 1e-9 or duration > maximum_s + 1e-9:
            continue
        if len(beats) - index < beats_per_bar + 1:
            continue
        candidates.append((abs(beat - target_start), index, duration))
    if not candidates:
        return None

    _, index, duration = min(candidates)
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
) -> TransitionPlan:
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
    return _make_plan(
        context,
        strategy="conservative",
        duration_s=duration,
        automations=_full_mix_automations(context, duration, curve="smoothstep"),
        reasoning=reasoning,
        warnings=warnings,
        metadata={"fallback": requested_strategy != "conservative"},
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
    description = "Beat-aligned stem automation, used only with reliable tempo."

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

        aligned = _beat_aligned_rhythm(
            context,
            requested_duration_s=context.requested_duration_s,
            beats_per_bar=beats_per_bar,
            minimum_s=minimum,
            maximum_s=maximum,
        )
        beat_grid: tuple[float, ...] | None = None
        anchor_beat_s: float | None = None
        if aligned is not None:
            duration, bars, total_beats, beat_grid, anchor_beat_s = aligned
        else:
            total_beats = bars * beats_per_bar

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
        if beat_grid is not None:
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
                    "Downbeats are not detected; bar alignment assumes 4/4 "
                    "meter. Detected beats came from the "
                    f"{context.analysis_a.analyzer!r} analyzer.",
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
        if anchor_beat_s is not None:
            metadata["anchor_beat_s"] = round(anchor_beat_s, 6)
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
        OptimizedStrategy(),
    ):
        registry.register(strategy)
    return registry

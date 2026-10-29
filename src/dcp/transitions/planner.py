"""Transition planner orchestration."""

from __future__ import annotations

import math

from ..analysis import TrackAnalysis, analyze_track_with_backend
from ..config import DEFAULT_CONFIG, DCPConfig
from ..models import AudioBuffer, DCPError, StemSet
from .context import TransitionContext
from .plan import TransitionPlan, validate_plan
from .strategies import (
    StrategyRegistry,
    build_core_registry,
    tempo_is_usable,
)


class TransitionPlanner:
    """Selects and runs a strategy, then validates the resulting plan."""

    def __init__(
        self,
        *,
        config: DCPConfig | None = None,
        registry: StrategyRegistry | None = None,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self.registry = registry or build_core_registry()

    def plan(
        self,
        song_a: AudioBuffer,
        song_b: AudioBuffer,
        *,
        stems_a: StemSet | None = None,
        stems_b: StemSet | None = None,
        strategy: str = "auto",
        duration_s: float | None = None,
        analyzer: str = "auto",
        analysis_a: TrackAnalysis | None = None,
        analysis_b: TrackAnalysis | None = None,
    ) -> TransitionPlan:
        if duration_s is None:
            requested = self.config.transition.default_duration_s
        else:
            requested = float(duration_s)
        if not math.isfinite(requested) or requested <= 0:
            raise DCPError(
                f"transition duration must be a positive finite number, got "
                f"{duration_s}"
            )
        if min(song_a.duration_s, song_b.duration_s) <= 0.05:
            raise DCPError("both tracks must be longer than 0.05 seconds")

        first_analysis = analysis_a or analyze_track_with_backend(
            song_a, self.config.analysis, backend=analyzer
        )
        second_analysis = analysis_b or analyze_track_with_backend(
            song_b, self.config.analysis, backend=analyzer
        )

        context = TransitionContext(
            song_a=song_a,
            song_b=song_b,
            stems_a=stems_a,
            stems_b=stems_b,
            analysis_a=first_analysis,
            analysis_b=second_analysis,
            requested_duration_s=requested,
            requested_strategy=strategy,
            config=self.config,
        )

        selected = self._select(strategy, context)
        implementation = self.registry.get(selected)
        try:
            plan = implementation.plan(context)
        except DCPError:
            raise
        except Exception as exc:
            raise DCPError(
                f"transition strategy {selected!r} failed: {exc}"
            ) from exc

        validate_plan(plan, self.config, sample_rate=song_a.sample_rate)
        return plan

    def _select(self, requested: str, context: TransitionContext) -> str:
        normalized = requested.strip().lower()
        if normalized != "auto":
            if normalized not in self.registry:
                available = ", ".join(self.registry.names())
                raise DCPError(
                    f"unknown transition strategy {requested!r}; available: "
                    f"auto, {available}"
                )
            return normalized

        if "rhythmic" in self.registry and tempo_is_usable(context)[0]:
            return "rhythmic"
        if context.stems_a is not None or context.stems_b is not None:
            if "stem_level" in self.registry:
                return "stem_level"
        return "conservative"


def plan_transition(
    song_a: AudioBuffer,
    song_b: AudioBuffer,
    *,
    stems_a: StemSet | None = None,
    stems_b: StemSet | None = None,
    strategy: str = "auto",
    duration_s: float | None = None,
    analyzer: str = "auto",
    config: DCPConfig | None = None,
    analysis_a: TrackAnalysis | None = None,
    analysis_b: TrackAnalysis | None = None,
) -> TransitionPlan:
    """Convenience wrapper around :class:`TransitionPlanner`."""

    planner = TransitionPlanner(config=config)
    return planner.plan(
        song_a,
        song_b,
        stems_a=stems_a,
        stems_b=stems_b,
        strategy=strategy,
        duration_s=duration_s,
        analyzer=analyzer,
        analysis_a=analysis_a,
        analysis_b=analysis_b,
    )

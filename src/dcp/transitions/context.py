"""Inputs shared by transition-planning strategies."""

from __future__ import annotations

from dataclasses import dataclass

from ..analysis import TrackAnalysis
from ..config import DCPConfig
from ..models import AudioBuffer, StemSet


@dataclass(frozen=True)
class TransitionContext:
    """Everything a strategy may inspect when creating a plan."""

    song_a: AudioBuffer
    song_b: AudioBuffer
    stems_a: StemSet | None
    stems_b: StemSet | None
    analysis_a: TrackAnalysis
    analysis_b: TrackAnalysis
    requested_duration_s: float
    requested_strategy: str
    config: DCPConfig

    @property
    def max_duration_s(self) -> float:
        return min(self.song_a.duration_s, self.song_b.duration_s)

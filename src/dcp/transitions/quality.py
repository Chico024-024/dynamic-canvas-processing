"""Pluggable candidate-quality scoring.

The optimizer generates transition candidates and asks a :class:`QualityScorer`
to rank them. Core ships exactly one scorer, :class:`WeightedLossScorer` (in
:mod:`dcp.transitions.optimize`), which wraps the existing hand-tuned weighted
loss sum. Alternative scorers can be supplied by callers; this module defines
only the stable contract and the feature record.

A scorer is a *ranking function*, not a perceptual quality model. Nothing here
learns from data, and no scorer can establish musical quality on its own.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Protocol, runtime_checkable

from ..models import DCPError


class QualityScoringError(DCPError):
    """Raised when a scorer fails or returns an invalid score."""


@dataclass(frozen=True)
class CandidateFeatures:
    """Everything a scorer may inspect about one rendered candidate.

    Features are deliberately limited to values DCP can measure offline:
    normalized loss terms, transition timing, Song A tempo, automation
    complexity, and vocal overlap. They are not a claim about musical quality.
    """

    candidate: str
    losses: Mapping[str, float]
    duration_s: float
    tempo_bpm: float | None
    automation_count: int
    gain_point_count: int
    effect_count: int
    vocal_overlap_s: float
    stem_aware: bool

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, str) or not self.candidate.strip():
            raise ValueError("candidate must be a non-empty string")
        if not isinstance(self.losses, Mapping) or not self.losses:
            raise ValueError("losses must be a non-empty mapping")
        frozen: dict[str, float] = {}
        for name, value in self.losses.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("loss names must be non-empty strings")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"loss {name!r} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"loss {name!r} must be finite")
            frozen[name] = float(value)
        object.__setattr__(self, "losses", MappingProxyType(frozen))

        if not math.isfinite(float(self.duration_s)) or self.duration_s <= 0:
            raise ValueError("duration_s must be a positive finite number")
        if self.tempo_bpm is not None:
            if not math.isfinite(float(self.tempo_bpm)) or self.tempo_bpm <= 0:
                raise ValueError("tempo_bpm must be positive and finite when given")
        for name, value in (
            ("automation_count", self.automation_count),
            ("gain_point_count", self.gain_point_count),
            ("effect_count", self.effect_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not math.isfinite(float(self.vocal_overlap_s)) or self.vocal_overlap_s < 0:
            raise ValueError("vocal_overlap_s must be finite and non-negative")
        if not isinstance(self.stem_aware, bool):
            raise ValueError("stem_aware must be a boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate": self.candidate,
            "stem_aware": self.stem_aware,
            "duration_s": round(float(self.duration_s), 6),
            "tempo_bpm": (
                None if self.tempo_bpm is None else round(float(self.tempo_bpm), 3)
            ),
            "vocal_overlap_s": round(float(self.vocal_overlap_s), 6),
            "automation_count": self.automation_count,
            "gain_point_count": self.gain_point_count,
            "effect_count": self.effect_count,
            "losses": {
                name: round(float(value), 6)
                for name, value in self.losses.items()
            },
        }


@runtime_checkable
class QualityScorer(Protocol):
    """Stable, pluggable candidate-scoring contract.

    ``score`` must return a finite number; **lower is better**, matching the
    minimization convention of the loss objective. A scorer may be
    deterministic hand-tuned math (Core's default) or a later learned model.
    """

    name: str

    def score(self, features: CandidateFeatures) -> float:
        """Return a finite quality score for one candidate."""


__all__ = [
    "CandidateFeatures",
    "QualityScorer",
    "QualityScoringError",
]

"""Core data models shared by analysis, planning, and rendering."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

# Reserved stem name meaning "the full mix of this source".
MIX_STEM = "__mix__"

CANONICAL_STEM_ORDER: tuple[str, ...] = ("drums", "bass", "other", "vocals")


class DCPError(Exception):
    """Base class for expected, user-facing DCP errors."""


class AudioFormatError(DCPError):
    """Raised when audio cannot be read, validated, or written."""


class AnalysisError(DCPError):
    """Raised when analysis cannot be completed."""


class PlanValidationError(DCPError):
    """Raised when a transition plan is internally inconsistent."""


class RenderError(DCPError):
    """Raised when a transition cannot be rendered."""


@dataclass(frozen=True)
class AudioBuffer:
    """De-interleaved audio samples.

    ``samples`` has shape ``(channels, frames)`` and is expected to contain
    finite floating-point values in the nominal range [-1.0, 1.0]. DCP
    processes internally in float64; file output is controlled by the I/O
    layer.
    """

    samples: np.ndarray
    sample_rate: int

    def __post_init__(self) -> None:
        if not isinstance(self.samples, np.ndarray):
            raise TypeError("samples must be a numpy.ndarray")
        if self.samples.ndim != 2:
            raise ValueError(
                "samples must be 2-D with shape (channels, frames); "
                f"got shape {self.samples.shape}"
            )
        if self.samples.shape[0] < 1:
            raise ValueError("audio must contain at least one channel")
        if self.samples.shape[1] < 1:
            raise ValueError("audio must contain at least one frame")
        if int(self.sample_rate) <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")

    @property
    def channels(self) -> int:
        return int(self.samples.shape[0])

    @property
    def frames(self) -> int:
        return int(self.samples.shape[1])

    @property
    def duration_s(self) -> float:
        return self.frames / float(self.sample_rate)

    @property
    def peak(self) -> float:
        return float(np.max(np.abs(self.samples)))

    @property
    def rms(self) -> float:
        return float(np.sqrt(np.mean(np.square(self.samples), dtype=np.float64)))

    def mono(self) -> np.ndarray:
        return np.mean(self.samples, axis=0)


@dataclass(frozen=True)
class StemSet:
    """A named collection of mono/stereo stems with identical format."""

    stems: Mapping[str, AudioBuffer]
    source_label: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.stems, Mapping) or not self.stems:
            raise ValueError("a StemSet must contain at least one stem")
        frozen = dict(self.stems)
        for name, audio in frozen.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("stem names must be non-empty strings")
            if name == MIX_STEM:
                raise ValueError(f"stem name {MIX_STEM!r} is reserved")
            if not isinstance(audio, AudioBuffer):
                raise TypeError(f"stem {name!r} is not an AudioBuffer")
        first = next(iter(frozen.values()))
        for name, audio in frozen.items():
            if audio.sample_rate != first.sample_rate:
                raise ValueError(
                    f"stem {name!r} has sample rate {audio.sample_rate}, "
                    f"expected {first.sample_rate}"
                )
            if audio.channels != first.channels:
                raise ValueError(
                    f"stem {name!r} has {audio.channels} channels, "
                    f"expected {first.channels}"
                )
        object.__setattr__(self, "stems", MappingProxyType(frozen))

    @property
    def names(self) -> tuple[str, ...]:
        known = [name for name in CANONICAL_STEM_ORDER if name in self.stems]
        extra = sorted(name for name in self.stems if name not in CANONICAL_STEM_ORDER)
        return tuple(known + extra)

    @property
    def sample_rate(self) -> int:
        return next(iter(self.stems.values())).sample_rate

    @property
    def channels(self) -> int:
        return next(iter(self.stems.values())).channels

    def __getitem__(self, name: str) -> AudioBuffer:
        return self.stems[name]

    def __contains__(self, name: object) -> bool:
        return name in self.stems

    def items(self):
        return self.stems.items()

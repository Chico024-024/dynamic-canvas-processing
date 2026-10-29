"""Replaceable stem-separation boundary.

DCP Core does not bundle a source-separation model in this milestone. Users can
supply stems directly; future separators (local or hosted) must implement the
``StemSeparator`` protocol and return one of the canonical stem categories when
possible. Model licensing and resource requirements must be reviewed before a
backend is integrated.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..models import AudioBuffer, DCPError, StemSet


class StemSeparationUnavailable(DCPError):
    """Raised when no usable separation backend is configured."""


@runtime_checkable
class StemSeparator(Protocol):
    """Minimal contract for optional stem-separation backends."""

    name: str

    def is_available(self) -> bool:
        """Return True when the backend and its model weights can be used."""

    def separate(
        self, audio: AudioBuffer, *, timeout_s: float | None = None
    ) -> StemSet:
        """Separate audio into named stems.

        Implementations must keep audio local unless the user has explicitly
        authorized an external service, must respect ``timeout_s`` where
        possible, and should return the canonical names ``vocals``, ``drums``,
        ``bass``, and ``other`` when their model supports them.
        """


class UnconfiguredSeparator:
    """Explicit placeholder used when no separation backend is installed."""

    name = "none"

    def is_available(self) -> bool:
        return False

    def separate(
        self, audio: AudioBuffer, *, timeout_s: float | None = None
    ) -> StemSet:
        raise StemSeparationUnavailable(
            "No stem-separation backend is installed. Provide stem WAV files "
            "directly (for example with --stems-a/--stems-b) or integrate a "
            "licensed separator that implements dcp.stems.StemSeparator."
        )

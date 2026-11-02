"""Stem management, separation contracts, and optional backends."""

from .demucs_adapter import DemucsAdapter, demucs_available, get_demucs_separator
from .separation import (
    StemSeparationUnavailable,
    StemSeparator,
    UnconfiguredSeparator,
)

__all__ = [
    "DemucsAdapter",
    "StemSeparationUnavailable",
    "StemSeparator",
    "UnconfiguredSeparator",
    "demucs_available",
    "get_demucs_separator",
]

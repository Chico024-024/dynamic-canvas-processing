"""Stem management, separation contracts, and optional backends."""

from .demucs_adapter import DemucsAdapter, demucs_available, get_demucs_separator
from .dsp_splitter import decompose_audio_to_stemset
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
    "decompose_audio_to_stemset",
    "demucs_available",
    "get_demucs_separator",
]

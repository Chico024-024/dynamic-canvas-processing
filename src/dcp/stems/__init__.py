"""Stem management, separation contracts, and optional backends."""

from .demucs_adapter import DemucsAdapter, demucs_available, get_demucs_separator
from .dsp_splitter import decompose_audio_to_stemset
from .hierarchical import get_available_backends, separate_hierarchical
from .roformer_adapter import (
    DEFAULT_ROFORMER_MODEL,
    RoformerAdapter,
    get_roformer_separator,
    roformer_available,
)
from .separation import (
    StemSeparationUnavailable,
    StemSeparator,
    UnconfiguredSeparator,
)

__all__ = [
    "DEFAULT_ROFORMER_MODEL",
    "DemucsAdapter",
    "RoformerAdapter",
    "StemSeparationUnavailable",
    "StemSeparator",
    "UnconfiguredSeparator",
    "decompose_audio_to_stemset",
    "demucs_available",
    "get_available_backends",
    "get_demucs_separator",
    "get_roformer_separator",
    "roformer_available",
    "separate_hierarchical",
]

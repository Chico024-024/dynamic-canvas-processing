"""Audio analysis primitives and optional librosa-backed beat tracking."""

from .basic import TrackAnalysis, amplitude_to_dbfs, analyze_track
from .librosa_backend import (
    analyze_track_librosa,
    analyze_track_with_backend,
    librosa_available,
)

__all__ = [
    "TrackAnalysis",
    "amplitude_to_dbfs",
    "analyze_track",
    "analyze_track_librosa",
    "analyze_track_with_backend",
    "librosa_available",
]

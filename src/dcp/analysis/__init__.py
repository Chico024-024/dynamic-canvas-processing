"""Audio analysis primitives and optional librosa/neural-backed beat tracking."""

from .basic import TrackAnalysis, amplitude_to_dbfs, analyze_track
from .key_detection import (
    camelot_distance,
    detect_key,
    estimate_downbeats,
    key_compatibility_score,
)
from .librosa_backend import (
    analyze_track_librosa,
    librosa_available,
)
from .neural_beat import (
    analyze_track_neural,
    analyze_track_with_backend,
    neural_beat_available,
)
from .neural_key import (
    analyze_track_with_neural_key,
    detect_key_neural,
    neural_key_available,
)
from .structure import (
    analyze_track_with_structure,
    neural_structure_available,
    segment_structure_neural,
)

__all__ = [
    "TrackAnalysis",
    "amplitude_to_dbfs",
    "analyze_track",
    "analyze_track_librosa",
    "analyze_track_neural",
    "analyze_track_neural_key",
    "analyze_track_with_backend",
    "analyze_track_with_neural_key",
    "analyze_track_with_structure",
    "camelot_distance",
    "detect_key",
    "detect_key_neural",
    "estimate_downbeats",
    "key_compatibility_score",
    "librosa_available",
    "neural_beat_available",
    "neural_key_available",
    "neural_structure_available",
    "segment_structure_neural",
]

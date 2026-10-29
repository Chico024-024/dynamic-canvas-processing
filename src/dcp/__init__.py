"""DCP Core: Dynamic Canvas Processing."""

from .audio_io import load_audio, save_audio
from .config import (
    DCPConfig,
    DEFAULT_CONFIG,
    AnalysisConfig,
    OptimizationConfig,
    RenderConfig,
    TransitionConfig,
)
from .models import (
    MIX_STEM,
    AudioBuffer,
    AudioFormatError,
    DCPError,
    PlanValidationError,
    RenderError,
    StemSet,
)
from .synthetic import SyntheticTrack, generate_synthetic_track
from .transitions import optimize_transition

__version__ = "0.1.0"

__all__ = [
    "MIX_STEM",
    "AnalysisConfig",
    "AudioBuffer",
    "AudioFormatError",
    "DCPConfig",
    "DCPError",
    "DEFAULT_CONFIG",
    "OptimizationConfig",
    "PlanValidationError",
    "RenderConfig",
    "RenderError",
    "StemSet",
    "SyntheticTrack",
    "TransitionConfig",
    "__version__",
    "generate_synthetic_track",
    "load_audio",
    "optimize_transition",
    "save_audio",
]

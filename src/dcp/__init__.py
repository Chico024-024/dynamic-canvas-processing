"""DCP Core: Dynamic Canvas Processing."""

__version__ = "0.1.0"

from .audio_io import load_audio, save_audio
from .config import (
    CrossfadeConfig,
    DCPConfig,
    DCPFeatureConfig,
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
from .visualizer import generate_html_report, save_html_report

__all__ = [
    "MIX_STEM",
    "AnalysisConfig",
    "AudioBuffer",
    "AudioFormatError",
    "CrossfadeConfig",
    "DCPConfig",
    "DCPFeatureConfig",
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
    "generate_html_report",
    "generate_synthetic_track",
    "load_audio",
    "optimize_transition",
    "save_audio",
    "save_html_report",
]

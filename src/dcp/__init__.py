"""DCP Core: Dynamic Canvas Processing."""

__version__ = "0.1.0"

from .automix import AutomixPlan, AutomixResult, plan_automix, render_automix
from .audio_io import load_audio, save_audio
from .config import (
    AutomixConfig,
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
    "AutomixConfig",
    "AutomixPlan",
    "AutomixResult",
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
    "plan_automix",
    "render_automix",
    "save_audio",
    "save_html_report",
]

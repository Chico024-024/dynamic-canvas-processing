"""Configuration objects for the DCP Core engine.

All configuration is expressed as immutable dataclasses with conservative,
documented defaults. Values that affect audio processing are intentionally
visible rather than hidden in module globals.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AnalysisConfig:
    """Parameters for basic track analysis and tempo estimation."""

    stft_window: int = 1024
    stft_hop: int = 256
    silence_dbfs: float = -60.0
    silence_window_s: float = 0.05
    tempo_min_bpm: float = 70.0
    tempo_max_bpm: float = 180.0
    tempo_confidence_threshold: float = 0.30
    tempo_smoothing_frames: int = 5


@dataclass(frozen=True)
class TransitionConfig:
    """Parameters shared by transition planning strategies."""

    default_duration_s: float = 8.0
    min_duration_s: float = 2.0
    max_automation_gain_db: float = 6.0
    fade_floor_db: float = -120.0
    tempo_match_tolerance: float = 0.02
    rhythmic_beats_per_bar: int = 4
    atmospheric_scale: float = 1.4
    atmospheric_min_duration_s: float = 10.0


@dataclass(frozen=True)
class RenderConfig:
    """Parameters for offline rendering."""

    prevent_clipping: bool = True
    peak_ceiling: float = 0.999
    wav_subtype: str = "PCM_24"
    effect_chunk_samples: int = 256


@dataclass(frozen=True)
class OptimizationConfig:
    """Parameters for the objective-based transition optimizer (RATO).

    ``max_candidates`` bounds how many candidate timing plans are generated and
    scored. The ``w_*`` fields are the (hand-tuned, not yet perceptually
    validated) weights of the seven normalized loss terms in the objective.
    """

    max_candidates: int = 8
    w_loudness: float = 1.0
    w_continuity: float = 1.0
    w_musical: float = 1.0
    w_vocal: float = 1.0
    w_rhythm: float = 1.0
    w_distortion: float = 1.0
    w_complexity: float = 1.0


@dataclass(frozen=True)
class CrossfadeConfig:
    """Parameters for the standalone Crossfade feature (volume overlap)."""

    enabled: bool = True
    default_overlap_s: float = 8.0
    curve: str = "equal_power"  # "equal_power", "s_curve", or "linear"
    trim_silence: bool = False
    silence_threshold_dbfs: float = -42.0


@dataclass(frozen=True)
class AutomixConfig:
    """Parameters for the Apple Music-style Automix engine."""

    enabled: bool = True
    default_duration_s: float = 8.0
    beat_align: bool = True
    downbeat_phase_lock: bool = True
    tempo_sync: bool = True
    max_tempo_stretch_pct: float = 0.08
    smart_transition_points: bool = True
    bass_swap: bool = True
    bass_crossover_hz: float = 160.0
    loudness_match: bool = True
    max_loudness_trim_db: float = 4.5
    curve: str = "s_curve"


@dataclass(frozen=True)
class DCPFeatureConfig:
    """Modular feature toggles and gain staging for DCP (Dynamic Canvas Processing)."""

    enabled: bool = True
    enable_zdf_sweeps: bool = True
    enable_fdn_freeze: bool = True
    enable_sidechain_duck: bool = True
    enable_generative_assets: bool = True
    enable_vocal_chops: bool = True
    enable_drum_regroove: bool = True
    enable_master_glue: bool = True
    enable_tempo_lock: bool = True
    effect_edge_crossfade_ms: float = 6.0
    riser_level_db: float = -8.0
    crash_level_db: float = -6.0
    sub_808_level_db: float = -3.0
    vocal_chop_level_db: float = -6.0
    regroove_gain: float = 1.0
    ghost_tail_gain: float = 0.7


@dataclass(frozen=True)
class DCPConfig:
    """Top-level configuration container."""

    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    transition: TransitionConfig = field(default_factory=TransitionConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    crossfade: CrossfadeConfig = field(default_factory=CrossfadeConfig)
    automix: AutomixConfig = field(default_factory=AutomixConfig)
    dcp_features: DCPFeatureConfig = field(default_factory=DCPFeatureConfig)
    validation_max_gap_s: float = 3.0
    validation_click_threshold: float = 1.5
    validation_silence_dbfs: float = -70.0


DEFAULT_CONFIG = DCPConfig()

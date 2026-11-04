# src/dcp/transitions/musical_ir.py
"""
Layer 2: Machine-Readable Musical Representation (TrackMusicalIR).
Decouples raw MIR signal extraction from Layer 3 Compositional Planning.
Explicitly tracks provenance and uncertainty across derived metrical hypotheses.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any, Dict, List, Literal, Tuple

import numpy as np

from dcp.transitions.bijective_warp import robust_quantile_normalize
from dcp.analysis.key_detection import _extract_chroma

HypothesisRelation = Literal[
    "detected_primary",
    "derived_half_time",
    "derived_double_time",
    "derived_4_3",
    "derived_3_4",
]


@dataclass(frozen=True)
class MetricalHypothesis:
    hypothesis_id: str
    relation_to_base: HypothesisRelation
    parent_hypothesis_id: str | None
    bpm: float
    base_detection_confidence: float      # Confidence of the underlying beat tracker [0.0, 1.0]
    metrical_plausibility_weight: float   # Prior weight for this metrical level [0.0, 1.0]
    beat_times_s: Tuple[float, ...]
    downbeat_times_s: Tuple[float, ...]
    accent_profile_16: Tuple[float, ...]  # 16-bin normalized onset density across a 4-beat bar


@dataclass(frozen=True)
class PhraseIR:
    phrase_index: int
    bar_start: int
    bar_end: int
    start_time_s: float
    end_time_s: float
    structural_role: Literal["intro", "verse_or_build", "climactic_peak", "breakdown", "outro"]
    boundary_salience: float              # Novelty contrast across the bar boundary [0.0, 1.0]
    chroma_12d: Tuple[float, ...]         # L2-normalized 12-pitch-class profile
    bass_root_pc: int                     # Dominant sub-bass pitch class [0..11]
    vocal_activity: float                 # Normalized vocal energy in phrase [0.0, 1.0]
    vocal_resolves_at_end: bool           # True if vocal energy drops in final 2 beats of phrase
    low_end_weight: float                 # Combined normalized drums + bass weight [0.0, 1.0]
    drum_crest_factor: float              # Normalized transient punch on first downbeat [0.0, 1.0]


@dataclass(frozen=True)
class TrackMusicalIR:
    track_id: str
    sample_rate: int
    duration_s: float
    base_bpm: float
    metrical_hypotheses: Tuple[MetricalHypothesis, ...]
    phrases: Tuple[PhraseIR, ...]
    metadata_limitations: Tuple[str, ...] = (
        "derived_metrical_hypotheses_share_base_beat_grid_provenance",
        "structural_roles_are_energy_contour_proxies_not_semantic_lyrics",
    )

    def to_json(self) -> str:
        def _default(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj) if isinstance(obj, np.floating) else int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        return json.dumps(asdict(self), indent=2, default=_default)


def build_track_musical_ir(
    track_id: str,
    stems: Dict[str, np.ndarray],
    downbeats_s: np.ndarray,
    sr: int = 48000,
    base_confidence: float = 0.85,
    chroma_per_bar: np.ndarray | None = None,
) -> TrackMusicalIR:
    """
    Constructs a validated Layer 2 TrackMusicalIR from stems and downbeat timestamps.
    """
    downbeats = np.asarray(downbeats_s, dtype=np.float64)
    if len(downbeats) < 5:
        raise ValueError(f"Track {track_id} requires at least 5 downbeats to form 4-bar phrases.")

    bar_durations = np.diff(downbeats)
    median_bar_s = float(np.median(bar_durations))
    base_bpm = float(60.0 * 4.0 / max(median_bar_s, 1e-3))

    # Interpolate 4 beats per bar from downbeats
    beats_list: List[float] = []
    for m in range(len(downbeats) - 1):
        b0, b1 = downbeats[m], downbeats[m + 1]
        for step in range(4):
            beats_list.append(float(b0 + (step / 4.0) * (b1 - b0)))
    beats_list.append(float(downbeats[-1]))
    base_beats = np.array(beats_list, dtype=np.float64)

    # Extract 16-step accent profile from drums stem
    drums = stems.get("drums", np.zeros(int(downbeats[-1] * sr), dtype=np.float32))
    drums_mono = np.mean(drums, axis=0) if drums.ndim == 2 else drums
    accent_bins = np.zeros(16, dtype=np.float64)
    n_bars = len(downbeats) - 1

    for m in range(n_bars):
        s0 = int(round(downbeats[m] * sr))
        s1 = min(int(round(downbeats[m + 1] * sr)), len(drums_mono))
        if s1 > s0 + 16:
            edges = np.linspace(s0, s1, 17, dtype=int)
            for b in range(16):
                seg = drums_mono[edges[b] : edges[b + 1]]
                accent_bins[b] += float(np.sqrt(np.mean(seg * seg) + 1e-12))
    accent_norm = accent_bins / (np.linalg.norm(accent_bins) + 1e-12)
    accent_tuple = tuple(float(x) for x in accent_norm)

    # Generate multi-hypothesis metrical interpretations with explicit provenance
    hypotheses: List[MetricalHypothesis] = [
        MetricalHypothesis(
            hypothesis_id="primary",
            relation_to_base="detected_primary",
            parent_hypothesis_id=None,
            bpm=base_bpm,
            base_detection_confidence=float(base_confidence),
            metrical_plausibility_weight=1.0,
            beat_times_s=tuple(float(t) for t in base_beats),
            downbeat_times_s=tuple(float(t) for t in downbeats),
            accent_profile_16=accent_tuple,
        )
    ]

    # Half-time hypothesis (valid when base_bpm >= 128 BPM, e.g., 186 BPM -> 93 BPM)
    if base_bpm >= 128.0:
        hypotheses.append(
            MetricalHypothesis(
                hypothesis_id="half_time",
                relation_to_base="derived_half_time",
                parent_hypothesis_id="primary",
                bpm=0.5 * base_bpm,
                base_detection_confidence=float(base_confidence),
                metrical_plausibility_weight=0.85 if base_bpm >= 150.0 else 0.65,
                beat_times_s=tuple(float(t) for t in base_beats[::2]),
                downbeat_times_s=tuple(float(t) for t in downbeats[::2]),
                accent_profile_16=accent_tuple,
            )
        )

    # Double-time hypothesis (valid when base_bpm <= 105 BPM, e.g., 93 BPM -> 186 BPM)
    if base_bpm <= 105.0:
        double_beats = np.interp(
            np.arange(0, len(base_beats) - 0.5, 0.5),
            np.arange(len(base_beats)),
            base_beats,
        )
        hypotheses.append(
            MetricalHypothesis(
                hypothesis_id="double_time",
                relation_to_base="derived_double_time",
                parent_hypothesis_id="primary",
                bpm=2.0 * base_bpm,
                base_detection_confidence=float(base_confidence),
                metrical_plausibility_weight=0.80,
                beat_times_s=tuple(float(t) for t in double_beats),
                downbeat_times_s=tuple(float(t) for t in downbeats),
                accent_profile_16=accent_tuple,
            )
        )

    # 4/3 Dotted Metric Pivot hypothesis (e.g., 93 BPM * 4/3 = 124 BPM)
    hypotheses.append(
        MetricalHypothesis(
            hypothesis_id="metric_4_3",
            relation_to_base="derived_4_3",
            parent_hypothesis_id="primary",
            bpm=base_bpm * (4.0 / 3.0),
            base_detection_confidence=float(base_confidence),
            metrical_plausibility_weight=0.55,
            beat_times_s=tuple(float(t) for t in base_beats),
            downbeat_times_s=tuple(float(t) for t in downbeats),
            accent_profile_16=accent_tuple,
        )
    )

    # Compute per-bar RMS across stems using robust quantile normalization
    bar_rms: Dict[str, np.ndarray] = {}
    for name in ("drums", "bass", "vocals", "other"):
        arr = stems.get(name, np.zeros_like(drums_mono))
        mono = np.mean(arr, axis=0) if arr.ndim == 2 else arr
        vals = np.zeros(n_bars, dtype=np.float64)
        for m in range(n_bars):
            s0 = int(round(downbeats[m] * sr))
            s1 = min(int(round(downbeats[m + 1] * sr)), len(mono))
            if s1 > s0:
                seg = mono[s0:s1]
                vals[m] = float(np.sqrt(np.mean(seg * seg) + 1e-12))
        bar_rms[name] = robust_quantile_normalize(vals)

    # Compute first-beat drum crest factor per bar
    crest = np.zeros(n_bars, dtype=np.float64)
    for m in range(n_bars):
        bar_dur = downbeats[m + 1] - downbeats[m]
        s0 = int(round(downbeats[m] * sr))
        s1 = min(int(round((downbeats[m] + 0.25 * bar_dur) * sr)), len(drums_mono))
        if s1 > s0:
            seg = drums_mono[s0:s1]
            crest[m] = float(np.max(np.abs(seg)) / (np.sqrt(np.mean(seg * seg)) + 1e-9))
    crest_norm = robust_quantile_normalize(crest)

    # Group into 4-bar PhraseIR objects
    phrases: List[PhraseIR] = []
    voc_raw = stems.get("vocals", np.zeros_like(drums_mono))
    voc_mono = np.mean(voc_raw, axis=0) if voc_raw.ndim == 2 else voc_raw

    for p_idx, m_start in enumerate(range(0, n_bars - 3, 4)):
        m_end = min(m_start + 4, n_bars)
        low_w = float(np.mean(0.6 * bar_rms["bass"][m_start:m_end] + 0.4 * bar_rms["drums"][m_start:m_end]))
        voc_w = float(np.mean(bar_rms["vocals"][m_start:m_end]))
        oth_w = float(np.mean(bar_rms["other"][m_start:m_end]))

        # Check whether vocal resolves in the last 2 beats of the 4-bar phrase
        t_p_end = downbeats[m_end]
        t_last_2b = t_p_end - 0.5 * (downbeats[m_end] - downbeats[m_end - 1])
        s_tail0 = max(0, int(round(t_last_2b * sr)))
        s_tail1 = min(len(voc_mono), int(round(t_p_end * sr)))
        tail_rms = float(np.sqrt(np.mean(voc_mono[s_tail0:s_tail1] ** 2) + 1e-12)) if s_tail1 > s_tail0 else 0.0
        s_full0 = max(0, int(round(downbeats[m_start] * sr)))
        full_rms = float(np.sqrt(np.mean(voc_mono[s_full0:s_tail1] ** 2) + 1e-12)) if s_tail1 > s_full0 else 1e-9
        voc_resolves = bool(tail_rms < 0.55 * full_rms or voc_w < 0.25)

        # Energy-contour structural role proxy
        rel_pos = m_start / max(1, n_bars)
        if rel_pos < 0.12:
            role: Any = "intro"
        elif rel_pos > 0.85 and low_w < 0.45:
            role = "outro"
        elif low_w >= 0.65 and oth_w >= 0.50:
            role = "climactic_peak"
        elif low_w < 0.35:
            role = "breakdown"
        else:
            role = "verse_or_build"

        # Boundary salience (energy jump across m_start)
        if m_start >= 4:
            prev_low = float(np.mean(0.6 * bar_rms["bass"][m_start - 4 : m_start] + 0.4 * bar_rms["drums"][m_start - 4 : m_start]))
            salience = float(np.clip(abs(low_w - prev_low), 0.0, 1.0))
        else:
            salience = 0.5

        if chroma_per_bar is not None and len(chroma_per_bar) >= m_end:
            c_vec = np.mean(chroma_per_bar[m_start:m_end], axis=0)
            c_vec = c_vec / (np.linalg.norm(c_vec) + 1e-12)
            bass_pc = int(np.argmax(c_vec))
        else:
            # Auto-compute chroma from harmonic stems (other + vocals + bass)
            harm_sources = []
            for stem_k in ("other", "vocals", "bass"):
                if stem_k in stems:
                    s_arr = stems[stem_k]
                    harm_sources.append(np.mean(s_arr, axis=0) if s_arr.ndim == 2 else s_arr)
            if harm_sources:
                combined_harm = np.sum(harm_sources, axis=0)
                seg_harm = combined_harm[s_full0:s_tail1]
                if len(seg_harm) >= 512:
                    c_vec = _extract_chroma(seg_harm, sr)
                else:
                    c_vec = np.ones(12, dtype=np.float64) / np.sqrt(12.0)
            else:
                c_vec = np.ones(12, dtype=np.float64) / np.sqrt(12.0)
            c_vec = c_vec / (np.linalg.norm(c_vec) + 1e-12)

            # Compute bass root pitch class from bass stem
            bass_raw = stems.get("bass")
            if bass_raw is not None:
                bass_mono = np.mean(bass_raw, axis=0) if bass_raw.ndim == 2 else bass_raw
                seg_bass = bass_mono[s_full0:s_tail1]
                if len(seg_bass) >= 512 and np.max(np.abs(seg_bass)) > 1e-4:
                    bass_chroma = _extract_chroma(seg_bass, sr)
                    bass_pc = int(np.argmax(bass_chroma))
                else:
                    bass_pc = int(np.argmax(c_vec))
            else:
                bass_pc = int(np.argmax(c_vec))

        phrases.append(
            PhraseIR(
                phrase_index=p_idx,
                bar_start=m_start,
                bar_end=m_end,
                start_time_s=float(downbeats[m_start]),
                end_time_s=float(downbeats[m_end]),
                structural_role=role,
                boundary_salience=salience,
                chroma_12d=tuple(float(x) for x in c_vec),
                bass_root_pc=bass_pc,
                vocal_activity=voc_w,
                vocal_resolves_at_end=voc_resolves,
                low_end_weight=low_w,
                drum_crest_factor=float(crest_norm[m_start]),
            )
        )

    return TrackMusicalIR(
        track_id=track_id,
        sample_rate=sr,
        duration_s=float(downbeats[-1]),
        base_bpm=base_bpm,
        metrical_hypotheses=tuple(hypotheses),
        phrases=tuple(phrases),
    )


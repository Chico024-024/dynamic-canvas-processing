"""Apple Music-style Automix Engine.

Automix is a distinct, self-contained feature separate from both:
- **Crossfade**: Simple full-mix volume overlap without musical alignment.
- **DCP (Dynamic Canvas Processing)**: Stem-level compositional engine / "Song C"
  Studio Producer with ZDF sweeps, FDN freeze, vocal chops, drum re-grooving, and
  generative risers/808s.

Automix focuses on making consecutive songs flow into one continuous, natural mix:
1. **Smart Transition Point Detection**: Identifies the optimal outro cue in Song A
   (after vocal peaks taper or before trailing dead air) and intro cue in Song B
   (skipping leading silence to catch the first rhythmic/musical entry).
2. **Downbeat & Beat Phase Alignment**: Snaps the transition window to whole bars
   and phase-locks Song B's incoming downbeat to Song A's outgoing downbeat using
   the Downbeat Coincidence Law.
3. **Bounded Tempo Sync (±8%)**: Smoothly time-stretches only Song B's transition
   overlap window when the two tracks' tempos (or 2x/0.5x octave multiples) are
   within ±8%, then seamlessly releases back to Song B's native tempo once Song A
   exits—avoiding beat flamming during the blend while keeping the body of Song B
   100% unwarped.
4. **Complementary Bass Swap Crossover**: Splits the overlap into Low (<160 Hz)
   and Mid/High (>=160 Hz) bands using zero-phase Butterworth crossover filters
   (where Low + MidHigh == Mix bit-exactly) so the low-end swaps cleanly at the
   midpoint bar without +6 dB sub-bass booming or phase cancellation.
5. **Perceptual Loudness Matching**: Measures active track loudness (LUFS / gated RMS)
   and applies a bounded gain trim to Song B during the transition so there is no
   sudden volume jump between tracks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

import numpy as np
from scipy import signal

from .analysis import TrackAnalysis, analyze_track_with_backend
from .audio_io import match_format
from .config import AutomixConfig, DCPConfig, DEFAULT_CONFIG
from .models import AudioBuffer, RenderError
from .rendering.common import RenderResult, add_unique_note, finalize_output
from .rendering.dsp_fallback import time_stretch_phase_vocoder_numpy
from .transitions.curves import crossfade_envelopes, smoothstep


@dataclass(frozen=True)
class AutomixPlan:
    """Explainable execution plan for an Apple Music-style Automix transition."""

    duration_s: float
    start_time_a_s: float
    end_time_a_s: float
    cue_offset_b_s: float
    bars: int
    beat_aligned: bool
    phase_locked: bool
    tempo_synced: bool
    stretch_ratio_b: float
    bass_swap_enabled: bool
    bass_crossover_hz: float
    loudness_trim_b_db: float
    curve: str
    tempo_a_bpm: float | None = None
    tempo_b_bpm: float | None = None
    anchor_downbeat_a_s: float | None = None
    anchor_downbeat_b_s: float | None = None
    reasoning: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": "automix",
            "duration_s": round(self.duration_s, 6),
            "start_time_a_s": round(self.start_time_a_s, 6),
            "end_time_a_s": round(self.end_time_a_s, 6),
            "cue_offset_b_s": round(self.cue_offset_b_s, 6),
            "bars": self.bars,
            "beat_aligned": self.beat_aligned,
            "phase_locked": self.phase_locked,
            "tempo_synced": self.tempo_synced,
            "stretch_ratio_b": round(self.stretch_ratio_b, 6),
            "bass_swap_enabled": self.bass_swap_enabled,
            "bass_crossover_hz": round(self.bass_crossover_hz, 2),
            "loudness_trim_b_db": round(self.loudness_trim_b_db, 3),
            "curve": self.curve,
            "tempo_a_bpm": None if self.tempo_a_bpm is None else round(self.tempo_a_bpm, 3),
            "tempo_b_bpm": None if self.tempo_b_bpm is None else round(self.tempo_b_bpm, 3),
            "anchor_downbeat_a_s": (
                None if self.anchor_downbeat_a_s is None else round(self.anchor_downbeat_a_s, 6)
            ),
            "anchor_downbeat_b_s": (
                None if self.anchor_downbeat_b_s is None else round(self.anchor_downbeat_b_s, 6)
            ),
            "reasoning": list(self.reasoning),
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AutomixResult:
    """Rendered audio and explainable plan from the Automix engine."""

    audio: AudioBuffer
    plan: AutomixPlan
    notes: tuple[str, ...] = ()
    master_gain_db: float = 0.0


def _detect_smart_boundaries(
    audio: AudioBuffer,
    threshold_dbfs: float = -42.0,
) -> tuple[float, float, float]:
    """Find (audible_start_s, optimal_outro_start_s, audible_end_s) for a track.

    Scans 50ms RMS windows to skip leading/trailing dead air and locates where
    the final 30% of the song begins to taper in vocal/midrange energy.
    """
    mono = audio.mono()
    sr = audio.sample_rate
    hop = max(1, int(0.05 * sr))
    n_hops = len(mono) // hop
    if n_hops < 4:
        return 0.0, max(0.0, audio.duration_s * 0.75), audio.duration_s

    thresh_lin = 10.0 ** (threshold_dbfs / 20.0)
    rms_env = np.array(
        [
            float(np.sqrt(np.mean(mono[i * hop : (i + 1) * hop] ** 2) + 1e-12))
            for i in range(n_hops)
        ],
        dtype=np.float64,
    )

    # 1. Audible start
    audible_start_s = 0.0
    for i in range(n_hops):
        if rms_env[i] >= thresh_lin:
            audible_start_s = max(0.0, (i - 1) * 0.05)
            break

    # 2. Audible end (before trailing silence)
    audible_end_s = audio.duration_s
    for i in range(n_hops - 1, -1, -1):
        if rms_env[i] >= thresh_lin:
            audible_end_s = min(audio.duration_s, (i + 2) * 0.05)
            break

    if audible_end_s <= audible_start_s + 0.5:
        audible_start_s = 0.0
        audible_end_s = audio.duration_s

    # 3. Outro taper detection in the final 35% of audible region
    search_start_idx = max(0, int(0.65 * (audible_end_s / 0.05)))
    search_end_idx = min(n_hops, int(audible_end_s / 0.05))
    peak_rms = float(np.percentile(rms_env[:search_end_idx], 85)) if search_end_idx > 2 else 0.1
    taper_thresh = max(thresh_lin * 2.0, peak_rms * 0.55)

    outro_start_s = max(audible_start_s, audible_end_s * 0.75)
    for i in range(search_start_idx, search_end_idx):
        # If sustained energy from i to end stays below taper threshold, mark outro
        remaining = rms_env[i:search_end_idx]
        if len(remaining) >= 4 and float(np.mean(remaining)) <= taper_thresh:
            outro_start_s = i * 0.05
            break

    return audible_start_s, outro_start_s, audible_end_s


def _compute_gated_loudness_db(audio: AudioBuffer, analysis: TrackAnalysis | None = None) -> float:
    """Return integrated loudness in LUFS if available, else gated RMS in dBFS."""
    if analysis is not None and analysis.loudness_lufs is not None and math.isfinite(analysis.loudness_lufs):
        return float(analysis.loudness_lufs)

    mono = audio.mono()
    sr = audio.sample_rate
    hop = max(1, int(0.1 * sr))
    n_hops = len(mono) // hop
    if n_hops < 2:
        rms = max(audio.rms, 1e-6)
        return float(20.0 * np.log10(rms))

    trimmed = mono[: n_hops * hop].reshape(n_hops, hop)
    block_rms = np.sqrt(np.mean(trimmed * trimmed, axis=1) + 1e-12)
    block_db = 20.0 * np.log10(block_rms)
    # Gate out blocks below -50 dBFS or 20 dB below peak block
    max_block_db = float(np.max(block_db))
    gate_db = max(-50.0, max_block_db - 20.0)
    active = block_rms[block_db >= gate_db]
    if active.size == 0:
        return float(20.0 * np.log10(max(audio.rms, 1e-6)))
    mean_rms = float(np.sqrt(np.mean(active * active)))
    return float(20.0 * np.log10(max(mean_rms, 1e-6)))


def _resolve_tempo_stretch_ratio(
    bpm_a: float | None,
    bpm_b: float | None,
    *,
    conf_a: float,
    conf_b: float,
    min_conf: float,
    max_stretch_pct: float,
) -> tuple[bool, float, str]:
    """Compute bounded stretch ratio so Song B matches Song A's metrical pulse.

    Checks 1:1, 2:1 (double-time), and 1:2 (half-time) relationships.
    Note on convention in time_stretch_phase_vocoder_numpy:
      rate > 1.0 speeds up audio (shortens duration),
      rate < 1.0 slows down audio (lengthens duration).
    So to bring Song B (at bpm_b) to Song A's tempo (bpm_a), rate = bpm_a / (bpm_b * mult).
    """
    if bpm_a is None or bpm_b is None or bpm_a <= 0 or bpm_b <= 0:
        return False, 1.0, "tempo unavailable"
    if conf_a < min_conf or conf_b < min_conf:
        return False, 1.0, f"tempo confidence below threshold ({min_conf:.2f})"

    best_rate = 1.0
    best_delta = float("inf")
    best_label = "1:1"
    for mult, label in ((1.0, "1:1"), (0.5, "half-time 1:2"), (2.0, "double-time 2:1")):
        effective_b = bpm_b * mult
        rate = bpm_a / effective_b
        delta = abs(rate - 1.0)
        if delta < best_delta:
            best_delta = delta
            best_rate = rate
            best_label = label

    if best_delta <= max_stretch_pct + 1e-6:
        return True, float(best_rate), best_label
    return False, 1.0, f"tempo delta {best_delta * 100:.1f}% exceeds ±{max_stretch_pct * 100:.1f}% bound"


def plan_automix(
    song_a: AudioBuffer,
    song_b: AudioBuffer,
    *,
    duration_s: float | None = None,
    config: DCPConfig | None = None,
    automix_config: AutomixConfig | None = None,
    analysis_a: TrackAnalysis | None = None,
    analysis_b: TrackAnalysis | None = None,
) -> AutomixPlan:
    """Create an explainable Apple Music-style Automix plan for two tracks."""
    cfg = config or DEFAULT_CONFIG
    amx = automix_config or cfg.automix

    req_dur = amx.default_duration_s if duration_s is None else float(duration_s)
    if not math.isfinite(req_dur) or req_dur <= 0:
        raise RenderError(f"Automix duration_s must be positive and finite, got {duration_s}")

    first_analysis = analysis_a or analyze_track_with_backend(song_a, cfg.analysis, backend="auto")
    second_analysis = analysis_b or analyze_track_with_backend(song_b, cfg.analysis, backend="auto")

    reasoning: list[str] = []
    warnings: list[str] = []

    # 1. Smart transition points (audible start/end & outro taper)
    if amx.smart_transition_points:
        _, outro_a_s, end_a_s = _detect_smart_boundaries(song_a)
        start_b_s, _, _ = _detect_smart_boundaries(song_b)
        if end_a_s < song_a.duration_s - 0.15:
            reasoning.append(
                f"Smart Outro Trim: Song A trailing silence trimmed at {end_a_s:.2f}s "
                f"(track length {song_a.duration_s:.2f}s)."
            )
        if start_b_s > 0.10:
            reasoning.append(
                f"Smart Intro Cue: Song B leading silence skipped to {start_b_s:.2f}s."
            )
    else:
        outro_a_s = song_a.duration_s * 0.75
        end_a_s = song_a.duration_s
        start_b_s = 0.0

    # Clamp overlap duration against available audible audio
    avail_a = max(0.2, end_a_s)
    avail_b = max(0.2, song_b.duration_s - start_b_s)
    max_overlap = min(avail_a, avail_b)
    min_overlap = min(cfg.transition.min_duration_s, max_overlap)
    overlap_s = min(max(req_dur, min_overlap), max_overlap)

    # 2. Tempo analysis & bounded tempo sync
    bpm_a = first_analysis.tempo_bpm
    bpm_b = second_analysis.tempo_bpm
    conf_a = first_analysis.tempo_confidence
    conf_b = second_analysis.tempo_confidence
    min_conf = cfg.analysis.tempo_confidence_threshold

    tempo_synced = False
    stretch_ratio_b = 1.0
    if amx.tempo_sync:
        tempo_synced, stretch_ratio_b, sync_note = _resolve_tempo_stretch_ratio(
            bpm_a,
            bpm_b,
            conf_a=conf_a,
            conf_b=conf_b,
            min_conf=min_conf,
            max_stretch_pct=amx.max_tempo_stretch_pct,
        )
        if tempo_synced and bpm_a is not None and bpm_b is not None:
            pct = (stretch_ratio_b - 1.0) * 100.0
            reasoning.append(
                f"Tempo Sync ({sync_note}): Song A {bpm_a:.1f} BPM <-> Song B {bpm_b:.1f} BPM "
                f"(overlap stretch {pct:+.2f}%, releasing to native tempo after handoff)."
            )
        else:
            warnings.append(f"Tempo sync skipped ({sync_note}); using filter-assisted glide.")

    # 3. Beat & Downbeat Phase Alignment
    beat_aligned = False
    phase_locked = False
    bars = max(1, int(round(overlap_s / 2.0)))
    anchor_a_s: float | None = None
    anchor_b_s: float | None = None
    cue_offset_b_s = start_b_s
    start_time_a_s = max(0.0, end_a_s - overlap_s)

    if amx.beat_align and bpm_a is not None and bpm_a > 0 and conf_a >= min_conf:
        beat_s = 60.0 / bpm_a
        bar_s = beat_s * 4.0
        bars = max(1, int(round(overlap_s / bar_s)))
        quantized_s = bars * bar_s
        while bars > 1 and quantized_s > max_overlap + 1e-6:
            bars -= 1
            quantized_s = bars * bar_s
        if quantized_s <= max_overlap + 1e-6 and quantized_s >= min(min_overlap, max_overlap) - 1e-6:
            overlap_s = quantized_s
            beat_aligned = True

        downbeats_a = [d for d in first_analysis.downbeat_times_s if 0.0 <= d <= end_a_s]
        downbeats_b = [d for d in second_analysis.downbeat_times_s if start_b_s <= d < song_b.duration_s]

        # Synthesize 4/4 bar grid if downbeats are not populated by backend analyzer
        if not downbeats_a and bpm_a and bpm_a > 0:
            downbeats_a = [i * bar_s for i in range(int(end_a_s / bar_s) + 1)]
        if not downbeats_b and bpm_b and bpm_b > 0:
            bar_s_b = (60.0 / bpm_b) * 4.0
            downbeats_b = [start_b_s + i * bar_s_b for i in range(int((song_b.duration_s - start_b_s) / bar_s_b) + 1)]

        if amx.downbeat_phase_lock and downbeats_a and downbeats_b:
            # Target transition start near end_a_s - overlap_s (or outro_a_s if close)
            ideal_start_a = max(0.0, end_a_s - overlap_s)
            lead_in_bars = max(1, bars // 2)
            lead_in_s = lead_in_bars * bar_s
            ideal_anchor_a = ideal_start_a + lead_in_s

            valid_a = [
                d for d in downbeats_a
                if lead_in_s <= d <= end_a_s and (d - lead_in_s + overlap_s) <= song_a.duration_s + 1e-6
            ]
            # For Song B, account for stretch_ratio_b during lead-in
            lead_in_src_b = lead_in_s / max(stretch_ratio_b, 0.5)
            valid_b = [
                d for d in downbeats_b
                if (d - lead_in_src_b) >= 0.0
                and (d - lead_in_src_b + overlap_s / max(stretch_ratio_b, 0.5)) <= song_b.duration_s
            ]

            if valid_a and valid_b:
                anchor_a_s = min(valid_a, key=lambda d: abs(d - ideal_anchor_a))
                anchor_b_s = valid_b[0]
                start_time_a_s = max(0.0, anchor_a_s - lead_in_s)
                cue_offset_b_s = max(0.0, anchor_b_s - lead_in_src_b)
                end_a_s = min(song_a.duration_s, start_time_a_s + overlap_s)
                overlap_s = end_a_s - start_time_a_s
                phase_locked = True
                reasoning.append(
                    f"Downbeat Phase-Lock: Song A downbeat at {anchor_a_s:.2f}s locked to "
                    f"Song B downbeat at {anchor_b_s:.2f}s across {bars} bar(s)."
                )
        elif beat_aligned:
            start_time_a_s = max(0.0, end_a_s - overlap_s)
            reasoning.append(
                f"Beat-Grid Quantization: Transition quantized to {bars} bar(s) ({overlap_s:.2f}s) at {bpm_a:.1f} BPM."
            )

    if not beat_aligned:
        start_time_a_s = max(0.0, end_a_s - overlap_s)
        end_a_s = start_time_a_s + overlap_s
        reasoning.append(
            f"Smooth Time-Window Overlap: {overlap_s:.2f}s continuous transition starting at {start_time_a_s:.2f}s."
        )

    # Ensure cue_offset_b_s leaves enough Song B frames for the overlap
    required_b_src_s = overlap_s / max(stretch_ratio_b, 0.5)
    if cue_offset_b_s + required_b_src_s > song_b.duration_s:
        cue_offset_b_s = max(0.0, song_b.duration_s - required_b_src_s)

    # 4. Loudness matching
    loudness_trim_b_db = 0.0
    if amx.loudness_match:
        loud_a = _compute_gated_loudness_db(song_a, first_analysis)
        loud_b = _compute_gated_loudness_db(song_b, second_analysis)
        raw_delta = loud_a - loud_b
        if abs(raw_delta) > 0.25:
            loudness_trim_b_db = float(
                np.clip(raw_delta, -amx.max_loudness_trim_db, amx.max_loudness_trim_db)
            )
            reasoning.append(
                f"Perceptual Loudness Match: Song A ({loud_a:.1f} dB) vs Song B ({loud_b:.1f} dB) -> "
                f"applied {loudness_trim_b_db:+.2f} dB trim to Song B."
            )

    if amx.bass_swap:
        reasoning.append(
            f"Frequency-Conscious Bass Swap: Complementary crossover at {amx.bass_crossover_hz:.0f} Hz "
            "prevents low-end booming during overlap."
        )

    return AutomixPlan(
        duration_s=overlap_s,
        start_time_a_s=start_time_a_s,
        end_time_a_s=start_time_a_s + overlap_s,
        cue_offset_b_s=cue_offset_b_s,
        bars=bars,
        beat_aligned=beat_aligned,
        phase_locked=phase_locked,
        tempo_synced=tempo_synced,
        stretch_ratio_b=stretch_ratio_b,
        bass_swap_enabled=amx.bass_swap,
        bass_crossover_hz=amx.bass_crossover_hz,
        loudness_trim_b_db=loudness_trim_b_db,
        curve=amx.curve,
        tempo_a_bpm=bpm_a,
        tempo_b_bpm=bpm_b,
        anchor_downbeat_a_s=anchor_a_s,
        anchor_downbeat_b_s=anchor_b_s,
        reasoning=tuple(reasoning),
        warnings=tuple(warnings),
        metadata={
            "feature": "automix",
            "bars": bars,
            "beat_aligned": beat_aligned,
            "phase_locked": phase_locked,
            "tempo_synced": tempo_synced,
            "stretch_ratio_b": round(stretch_ratio_b, 6),
            "bass_swap": amx.bass_swap,
            "loudness_trim_b_db": round(loudness_trim_b_db, 3),
            "cue_offset_b_s": round(cue_offset_b_s, 6),
        },
    )


def _split_low_high_complementary(
    samples: np.ndarray,
    sr: int,
    crossover_hz: float = 160.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Split audio into complementary (low, mid_high) bands where low + mid_high == samples."""
    if samples.shape[1] < 32:
        return np.zeros_like(samples), samples.copy()
    f_c = float(np.clip(crossover_hz, 40.0, 0.40 * sr))
    sos_lp = signal.butter(2, f_c / (0.5 * sr), btype="lowpass", output="sos")
    low = signal.sosfiltfilt(sos_lp, samples.astype(np.float64), axis=1)
    mid_high = samples.astype(np.float64) - low
    return low, mid_high


def render_automix(
    song_a: AudioBuffer,
    song_b: AudioBuffer,
    *,
    plan: AutomixPlan | None = None,
    duration_s: float | None = None,
    config: DCPConfig | None = None,
    automix_config: AutomixConfig | None = None,
    analysis_a: TrackAnalysis | None = None,
    analysis_b: TrackAnalysis | None = None,
) -> AutomixResult:
    """Render an Apple Music-style Automix transition between two tracks."""
    cfg = config or DEFAULT_CONFIG
    amx = automix_config or cfg.automix

    adapted_b, format_notes = match_format(song_b, song_a)
    sr = song_a.sample_rate
    channels = song_a.channels

    active_plan = plan or plan_automix(
        song_a,
        adapted_b,
        duration_s=duration_s,
        config=cfg,
        automix_config=amx,
        analysis_a=analysis_a,
        analysis_b=analysis_b,
    )

    start_frame_a = int(round(active_plan.start_time_a_s * sr))
    overlap_frames = int(round(active_plan.duration_s * sr))
    end_frame_a = min(song_a.frames, start_frame_a + overlap_frames)
    overlap_frames = max(1, end_frame_a - start_frame_a)

    cue_frame_b = int(round(active_plan.cue_offset_b_s * sr))
    cue_frame_b = max(0, min(adapted_b.frames - 1, cue_frame_b))
    b_from_cue = adapted_b.samples[:, cue_frame_b:]

    # Apply perceptual loudness trim to Song B
    trim_gain = float(10.0 ** (active_plan.loudness_trim_b_db / 20.0))
    if abs(trim_gain - 1.0) > 1e-4:
        b_from_cue = b_from_cue * trim_gain

    # Bounded tempo sync on Song B's overlap segment only, with smooth splice into native tail
    stretch_ratio = active_plan.stretch_ratio_b
    if active_plan.tempo_synced and abs(stretch_ratio - 1.0) > 0.002:
        # Number of source frames in Song B needed to produce `overlap_frames` after stretch
        # Since time_stretch_phase_vocoder_numpy with rate > 1 speeds up (output_len = input_len * rate),
        # Wait: let's check time_stretch_phase_vocoder_numpy:
        # time_steps = np.arange(0, n_frames - 1, 1.0 / rate) -> len(time_steps) = n_frames * rate!
        # Wait: when rate = bpm_a / bpm_b > 1 (Song A is faster than Song B), we want Song B to play FASTER
        # (shorter output duration for the same beats). In time_stretch_phase_vocoder_numpy:
        # step = 1.0 / rate. If rate > 1, step < 1, so there are MORE output frames (slower)!
        # Therefore, speed_factor = bpm_a / bpm_b corresponds to stretch_ratio = bpm_b / bpm_a in time_stretch_phase_vocoder_numpy!
        vocoder_ratio = float(np.clip(1.0 / stretch_ratio, 0.92, 1.08))
        src_overlap_frames = min(
            b_from_cue.shape[1],
            max(128, int(round(overlap_frames / max(vocoder_ratio, 0.5)))),
        )
        b_overlap_raw = b_from_cue[:, :src_overlap_frames]
        stretched_b = time_stretch_phase_vocoder_numpy(b_overlap_raw, vocoder_ratio).astype(np.float64)

        if stretched_b.shape[1] >= overlap_frames:
            b_overlap = stretched_b[:, :overlap_frames]
        else:
            pad_n = overlap_frames - stretched_b.shape[1]
            b_overlap = np.pad(stretched_b, ((0, 0), (0, pad_n)), mode="edge")

        b_tail_native = b_from_cue[:, src_overlap_frames:]
        # Smooth 30ms raised-cosine splice between stretched overlap end and native tail start
        splice_len = min(int(0.030 * sr), b_overlap.shape[1] // 4, b_tail_native.shape[1] // 4)
        if splice_len > 8 and src_overlap_frames + splice_len <= b_from_cue.shape[1]:
            w_out = 0.5 * (1.0 + np.cos(np.pi * np.arange(splice_len, dtype=np.float64) / splice_len))
            w_in = 1.0 - w_out
            native_pre = b_from_cue[:, src_overlap_frames - splice_len : src_overlap_frames]
            b_overlap[:, -splice_len:] = b_overlap[:, -splice_len:] * w_out + native_pre * w_in
    else:
        avail_b = min(overlap_frames, b_from_cue.shape[1])
        if avail_b < overlap_frames:
            b_overlap = np.pad(b_from_cue[:, :avail_b], ((0, 0), (0, overlap_frames - avail_b)))
        else:
            b_overlap = b_from_cue[:, :overlap_frames].copy()
        b_tail_native = b_from_cue[:, avail_b:]

    a_overlap = song_a.samples[:, start_frame_a : start_frame_a + overlap_frames].copy()

    # Compute high/mid fade envelopes
    fade_out_hi, fade_in_hi = crossfade_envelopes(overlap_frames, curve=active_plan.curve)

    if active_plan.bass_swap_enabled and overlap_frames >= 64:
        low_a, hi_a = _split_low_high_complementary(
            a_overlap, sr, crossover_hz=active_plan.bass_crossover_hz
        )
        low_b, hi_b = _split_low_high_complementary(
            b_overlap, sr, crossover_hz=active_plan.bass_crossover_hz
        )

        # Bass swap envelope: Song A bass holds until 35%, fades out by 55%;
        # Song B bass enters at 40% and reaches full level by 65%.
        p = np.linspace(0.0, 1.0, overlap_frames, endpoint=True, dtype=np.float64)
        bass_out_phase = smoothstep(np.clip((p - 0.30) / 0.25, 0.0, 1.0))
        bass_in_phase = smoothstep(np.clip((p - 0.42) / 0.25, 0.0, 1.0))
        fade_out_low = np.cos(bass_out_phase * np.pi / 2.0)
        fade_in_low = np.sin(bass_in_phase * np.pi / 2.0)

        blended_overlap = (
            low_a * fade_out_low[np.newaxis, :]
            + hi_a * fade_out_hi[np.newaxis, :]
            + low_b * fade_in_low[np.newaxis, :]
            + hi_b * fade_in_hi[np.newaxis, :]
        )
    else:
        blended_overlap = (
            a_overlap * fade_out_hi[np.newaxis, :]
            + b_overlap * fade_in_hi[np.newaxis, :]
        )

    total_frames = start_frame_a + overlap_frames + b_tail_native.shape[1]
    output = np.zeros((channels, total_frames), dtype=np.float64)
    output[:, :start_frame_a] = song_a.samples[:, :start_frame_a]
    output[:, start_frame_a : start_frame_a + overlap_frames] = blended_overlap
    if b_tail_native.shape[1] > 0:
        output[:, start_frame_a + overlap_frames :] = b_tail_native

    notes: list[str] = [
        f"Automix transition over {overlap_frames / sr:.3f}s "
        f"({'phase-locked' if active_plan.phase_locked else ('beat-aligned' if active_plan.beat_aligned else 'smooth')}, "
        f"curve={active_plan.curve}, bass_swap={active_plan.bass_swap_enabled})."
    ]
    for reason in active_plan.reasoning:
        add_unique_note(notes, reason)
    for note in format_notes:
        add_unique_note(notes, f"Song B: {note}.")

    finalized = finalize_output(
        output,
        sr,
        config=cfg.render,
        notes=notes,
        label="automix output",
    )
    return AutomixResult(
        audio=finalized.audio,
        plan=active_plan,
        notes=finalized.notes,
        master_gain_db=finalized.master_gain_db,
    )


__all__ = [
    "AutomixPlan",
    "AutomixResult",
    "plan_automix",
    "render_automix",
]

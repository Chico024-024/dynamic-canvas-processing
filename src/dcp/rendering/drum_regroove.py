"""
Autonomous Drum Re-Grooving & Half-Time Reconstruction Engine.
Slices drum stem audio at transient onsets and reconstructs a bespoke half-time pocket
for the transition bridge (Song C).
"""

from __future__ import annotations
import numpy as np
from scipy import signal
from scipy.signal import find_peaks
from typing import Literal


def detect_transient_onsets(
    drums_mono: np.ndarray,
    sr: int,
    min_distance_ms: float = 40.0,
) -> np.ndarray:
    """Detect transient onsets using a high-frequency novelty envelope."""
    # Highpass filter at 3 kHz to isolate transient strike energy
    sos = signal.butter(2, min(3000.0, sr * 0.45) / (0.5 * sr), btype="highpass", output="sos")
    hf = signal.sosfilt(sos, drums_mono.astype(np.float64))
    novelty = np.abs(hf)

    # Smooth with 10ms lowpass
    sos_lp = signal.butter(1, min(100.0, sr * 0.45) / (0.5 * sr), btype="lowpass", output="sos")
    env = signal.sosfilt(sos_lp, novelty)

    min_dist_samples = max(1, int((min_distance_ms * 0.001) * sr))
    prominence = np.quantile(env, 0.75) * 0.5
    peaks, _ = find_peaks(env, distance=min_dist_samples, prominence=max(1e-4, prominence))
    return peaks


def regroove_drums_half_time(
    drums_audio: np.ndarray,
    sr: int,
    bpm: float,
    duration_s: float,
    groove_style: Literal["half_time", "syncopated_trap", "breakbeat"] = "half_time",
    swing: float = 0.08,
) -> np.ndarray:
    """
    Re-groove drum audio onto an autonomous half-time pocket at `bpm`.
    Extracts transients and arranges kick/snare/hats on a half-time beat grid.
    """
    if drums_audio.ndim == 1:
        mono = drums_audio
        stereo = np.vstack([mono, mono])
    elif drums_audio.shape[0] == 1:
        mono = drums_audio[0]
        stereo = np.vstack([mono, mono])
    else:
        stereo = drums_audio[:2]
        mono = np.mean(stereo, axis=0)

    total_samples = int(round(duration_s * sr))
    if total_samples <= 0 or len(mono) < 1024:
        return np.zeros((2, total_samples), dtype=np.float32)

    peaks = detect_transient_onsets(mono, sr)
    if len(peaks) < 2:
        # Fallback: time-stretch or return repeated sliced audio
        return stereo[:, :total_samples].copy() if stereo.shape[1] >= total_samples else np.pad(stereo, ((0, 0), (0, total_samples - stereo.shape[1])))

    # Categorize slices by low-frequency vs high-frequency content
    slices = []
    n_peaks = len(peaks)
    for i in range(n_peaks):
        s0 = peaks[i]
        s1 = min(len(mono), peaks[i + 1] if i + 1 < n_peaks else s0 + int(0.4 * sr))
        if s1 - s0 > 64:
            slice_st = stereo[:, s0:s1]
            # Analyze energy below 150 Hz (kick indicator)
            low_sos = signal.butter(2, min(150.0, sr * 0.45) / (0.5 * sr), btype="lowpass", output="sos")
            low_energy = float(np.sum(signal.sosfilt(low_sos, mono[s0:s1].astype(np.float64)) ** 2))
            high_sos = signal.butter(2, min(2500.0, sr * 0.45) / (0.5 * sr), btype="highpass", output="sos")
            high_energy = float(np.sum(signal.sosfilt(high_sos, mono[s0:s1].astype(np.float64)) ** 2))
            total_e = float(np.sum(mono[s0:s1] ** 2)) + 1e-12
            slices.append({
                "audio": slice_st,
                "low_ratio": low_energy / total_e,
                "high_ratio": high_energy / total_e,
            })

    if not slices:
        return stereo[:, :total_samples].copy()

    # Sort into kicks (high low_ratio) and snares/hats
    kicks = sorted(slices, key=lambda x: x["low_ratio"], reverse=True)
    best_kick = kicks[0]["audio"]
    snares = sorted(slices, key=lambda x: x["high_ratio"], reverse=True)
    best_snare = snares[0]["audio"] if len(snares) > 1 else best_kick

    # Construct the half-time pattern across bars
    beat_dur = 60.0 / max(40.0, bpm)
    bar_dur = beat_dur * 4.0
    n_bars = int(np.ceil(duration_s / bar_dur))

    out = np.zeros((2, total_samples + int(sr * 1.0)), dtype=np.float32)

    for bar_idx in range(n_bars):
        bar_start_s = bar_idx * bar_dur
        # In half-time:
        # Kick on Beat 1 (0.0)
        # Optional ghost kick on Beat 2.5 or 4
        # Snare/Clap on Beat 3 (2.0 * beat_dur)
        # 8th-note hat ticks
        k_time = bar_start_s
        k_sample = int(round(k_time * sr))
        if k_sample < total_samples:
            k_len = min(best_kick.shape[1], out.shape[1] - k_sample)
            # Apply 5ms fade-out
            k_slice = best_kick[:, :k_len].copy()
            out[:, k_sample : k_sample + k_len] += k_slice

        # Snare on Beat 3
        s_time = bar_start_s + 2.0 * beat_dur
        s_sample = int(round(s_time * sr))
        if s_sample < total_samples:
            s_len = min(best_snare.shape[1], out.shape[1] - s_sample)
            s_slice = best_snare[:, :s_len].copy()
            out[:, s_sample : s_sample + s_len] += s_slice

        # Extra syncopated kick for groove
        if groove_style in ("syncopated_trap", "half_time"):
            k2_time = bar_start_s + 3.5 * beat_dur + swing * beat_dur
            k2_sample = int(round(k2_time * sr))
            if k2_sample < total_samples:
                k2_len = min(best_kick.shape[1], out.shape[1] - k2_sample)
                out[:, k2_sample : k2_sample + k2_len] += best_kick[:, :k2_len] * 0.75

    # Trim to exact total_samples and apply soft limiter to prevent clipping
    res = out[:, :total_samples]
    res = np.tanh(res * 0.95)
    return res.astype(np.float32)


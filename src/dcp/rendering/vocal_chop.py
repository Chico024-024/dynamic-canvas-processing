"""
Autonomous Vocal Chop & Rhythmic Sampler Engine.
Extracts phrase-resolving vocal slices and arranges a rhythmic 16th-note stutter pattern
under the ghost freeze bus for modern DAW-quality transition textures.
"""

from __future__ import annotations
import numpy as np
from scipy import signal


def generate_vocal_chops(
    vocal_audio: np.ndarray,
    sr: int,
    bpm: float,
    duration_s: float,
    level_db: float = -8.0,
    stutter_prob: float = 0.65,
    seed: int = 42,
) -> np.ndarray:
    """
    Slices the input vocal audio into 16th-note snippets and constructs
    a syncopated, rhythmic chop pattern.
    """
    if vocal_audio.ndim == 1:
        stereo = np.vstack([vocal_audio, vocal_audio])
    elif vocal_audio.shape[0] == 1:
        stereo = np.vstack([vocal_audio[0], vocal_audio[0]])
    else:
        stereo = vocal_audio[:2]

    total_samples = int(round(duration_s * sr))
    if total_samples <= 0 or stereo.shape[1] < 1024:
        return np.zeros((2, total_samples), dtype=np.float32)

    sixteenth_dur = (60.0 / max(40.0, bpm)) * 0.25
    slice_len = int(round(sixteenth_dur * sr))
    if slice_len <= 16:
        return np.zeros((2, total_samples), dtype=np.float32)

    # Gather audible 16th-note slices from the input audio
    num_avail = stereo.shape[1] // slice_len
    candidate_slices = []
    hann = (0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(slice_len) / slice_len))).astype(np.float32)

    for i in range(num_avail):
        s0 = i * slice_len
        s1 = s0 + slice_len
        seg = stereo[:, s0:s1]
        rms = np.sqrt(np.mean(seg ** 2))
        if rms > 0.015:  # Filter out silence
            windowed = seg * hann[np.newaxis, :]
            candidate_slices.append(windowed)

    if not candidate_slices:
        return np.zeros((2, total_samples), dtype=np.float32)

    # Deterministic pseudo-random generation of rhythmic chops
    rng = np.random.RandomState(seed)
    n_slots = int(np.ceil(duration_s / sixteenth_dur))
    out = np.zeros((2, total_samples + slice_len), dtype=np.float32)

    # 16th-note syncopation pattern mask: indices 0, 2, 3, 6, 8, 10, 11, 14
    groove_indices = {0, 2, 3, 6, 8, 10, 11, 14}

    for slot in range(n_slots):
        step_in_bar = slot % 16
        if step_in_bar in groove_indices and rng.uniform() < stutter_prob:
            idx = rng.randint(0, len(candidate_slices))
            chop = candidate_slices[idx]
            pos = int(round(slot * sixteenth_dur * sr))
            if pos < total_samples:
                avail = min(chop.shape[1], out.shape[1] - pos)
                out[:, pos : pos + avail] += chop[:, :avail]

    res = out[:, :total_samples]

    # Apply quarter-note sidechain pumping
    beat_dur = sixteenth_dur * 4.0
    t = np.arange(total_samples, dtype=np.float64) / sr
    beat_phase = (t % beat_dur) / beat_dur
    duck = 0.25 + 0.75 * np.clip(beat_phase / 0.5, 0.0, 1.0)
    res *= duck.astype(np.float32)[np.newaxis, :]

    linear_gain = float(np.power(10.0, level_db / 20.0))
    return (res * linear_gain).astype(np.float32)


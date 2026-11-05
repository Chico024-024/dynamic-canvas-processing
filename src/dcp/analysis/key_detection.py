"""Harmonic key detection, Camelot compatibility, and downbeat estimation.

This module provides a lightweight, CPU-only musical key estimator based on
chroma feature extraction and the Krumhansl-Schmuckler key-finding algorithm.
It requires only numpy and scipy (already Core dependencies).

The estimator works by:
1. Computing a chromagram (12 pitch-class energy bins) from the STFT.
2. Correlating the chroma profile against major and minor key templates.
3. Returning the best-matching key, its Camelot wheel code, and a confidence.

Key detection is approximate and heuristic. It does not claim musicological
accuracy, especially on heavily processed or atonal material.

Downbeat estimation uses spectral energy accents at known beat positions to
identify likely bar boundaries in 4/4 meter. It is a best-effort heuristic,
not a trained model.

DCP Core exports these as informational analysis fields. The planned Advanced
SDK may replace them with more sophisticated neural-network-based detection
and phrase-aware analysis.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import stft as scipy_stft

# ---------------------------------------------------------------------------
# Krumhansl-Kessler key profiles (Krumhansl 1990)
# Index 0 = C, 1 = C#/Db, ..., 11 = B
# ---------------------------------------------------------------------------
_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float64,
)
_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float64,
)

_NOTE_NAMES: tuple[str, ...] = (
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
)

# ---------------------------------------------------------------------------
# Camelot wheel mapping: (pitch_class_index, mode) -> code
# ---------------------------------------------------------------------------
_CAMELOT_CODES: dict[tuple[int, str], str] = {
    (0, "major"): "8B",  (0, "minor"): "5A",
    (1, "major"): "3B",  (1, "minor"): "12A",
    (2, "major"): "10B", (2, "minor"): "7A",
    (3, "major"): "5B",  (3, "minor"): "2A",
    (4, "major"): "12B", (4, "minor"): "9A",
    (5, "major"): "7B",  (5, "minor"): "4A",
    (6, "major"): "2B",  (6, "minor"): "11A",
    (7, "major"): "9B",  (7, "minor"): "6A",
    (8, "major"): "4B",  (8, "minor"): "1A",
    (9, "major"): "11B", (9, "minor"): "8A",
    (10, "major"): "6B", (10, "minor"): "3A",
    (11, "major"): "1B", (11, "minor"): "10A",
}


# ---------------------------------------------------------------------------
# Chroma extraction
# ---------------------------------------------------------------------------

def _extract_chroma(
    mono: np.ndarray,
    sample_rate: int,
    *,
    n_fft: int = 4096,
    hop_length: int = 1024,
) -> np.ndarray:
    """Extract a 12-bin chroma feature vector from mono audio.

    Returns a ``(12,)`` array of mean pitch-class energies, normalized to sum
    to 1. Uses power-spectrum magnitude mapped to pitch classes between C2
    (≈65 Hz) and C7 (≈2100 Hz).
    """
    window = min(n_fft, 1 << int(np.floor(np.log2(max(64, mono.shape[0])))))
    window = max(64, window)
    hop = min(hop_length, window // 2)
    hop = max(1, hop)

    nfft = window * 4  # 4x zero-padding for fine sub-bass frequency resolution
    _, _, spectrum = scipy_stft(
        mono,
        fs=sample_rate,
        window="hann",
        nperseg=window,
        noverlap=window - hop,
        nfft=nfft,
        boundary=None,
        padded=False,
    )

    magnitude = np.abs(spectrum) ** 2  # power spectrum
    freqs = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)

    chroma = np.zeros(12, dtype=np.float64)
    for bin_idx in range(1, len(freqs)):
        freq = freqs[bin_idx]
        if freq < 65.0 or freq > 2100.0:
            continue
        midi_f = 69.0 + 12.0 * np.log2(freq / 440.0)
        pc = int(np.floor(midi_f)) % 12
        w = 1.0 - (midi_f - np.floor(midi_f))  # spread energy across adjacent pitch classes
        energy = float(np.mean(magnitude[bin_idx, :]))
        chroma[pc] += energy * w
        chroma[(pc + 1) % 12] += energy * (1.0 - w)

    total = float(np.sum(chroma))
    if total > 1e-12:
        chroma /= total

    return chroma


# ---------------------------------------------------------------------------
# Key-profile correlation
# ---------------------------------------------------------------------------

def _correlate_profiles(chroma: np.ndarray) -> tuple[int, str, float]:
    """Find the key with highest Pearson correlation to the chroma profile.

    Returns ``(root_pitch_class, mode, confidence)`` where confidence is a
    heuristic based on the margin between the best and second-best match.
    """
    best_corr = -2.0
    best_root = 0
    best_mode = "major"
    all_corrs: list[float] = []

    for root in range(12):
        rotated = np.roll(chroma, -root)

        major_corr = float(np.corrcoef(rotated, _MAJOR_PROFILE)[0, 1])
        minor_corr = float(np.corrcoef(rotated, _MINOR_PROFILE)[0, 1])

        if not np.isfinite(major_corr):
            major_corr = 0.0
        if not np.isfinite(minor_corr):
            minor_corr = 0.0

        all_corrs.extend([major_corr, minor_corr])

        if major_corr > best_corr:
            best_corr = major_corr
            best_root = root
            best_mode = "major"
        if minor_corr > best_corr:
            best_corr = minor_corr
            best_root = root
            best_mode = "minor"

    all_corrs.sort(reverse=True)
    if len(all_corrs) >= 2 and all_corrs[0] > 0:
        margin = all_corrs[0] - all_corrs[1]
        confidence = float(np.clip(margin * 5.0 + 0.3 * max(0.0, best_corr), 0.0, 1.0))
    else:
        confidence = 0.0

    return best_root, best_mode, confidence


# ---------------------------------------------------------------------------
# Public key detection API
# ---------------------------------------------------------------------------

def detect_key(
    mono: np.ndarray,
    sample_rate: int,
) -> tuple[str, str, str, float]:
    """Detect the musical key of a mono audio signal.

    Parameters
    ----------
    mono : np.ndarray
        1-D mono audio samples.
    sample_rate : int
        Sample rate in Hz.

    Returns
    -------
    key : str
        Key name, e.g. ``"Am"``, ``"C"``, ``"F#m"``.
    camelot : str
        Camelot wheel code, e.g. ``"8A"``, ``"1B"``.
    mode : str
        ``"major"`` or ``"minor"``.
    confidence : float
        Heuristic confidence in ``[0, 1]``.
    """
    if mono.ndim != 1 or len(mono) < sample_rate:
        return "unknown", "?", "unknown", 0.0

    chroma = _extract_chroma(mono, sample_rate)

    if float(np.sum(chroma)) < 1e-12:
        return "unknown", "?", "unknown", 0.0

    root, mode, confidence = _correlate_profiles(chroma)

    note = _NOTE_NAMES[root]
    suffix = "m" if mode == "minor" else ""
    key = f"{note}{suffix}"
    camelot = _CAMELOT_CODES.get((root, mode), "?")

    return key, camelot, mode, confidence


# ---------------------------------------------------------------------------
# Camelot compatibility
# ---------------------------------------------------------------------------

def camelot_distance(code_a: str, code_b: str) -> int:
    """Compute the Camelot wheel distance between two key codes.

    Returns 0 for identical, 1 for adjacent (harmonically compatible), higher
    for more distant keys, or -1 if either code is invalid.

    Compatible transitions (distance ≤ 1):
    - Same code (0).
    - Same letter, ±1 number with wrapping 1↔12 (1).
    - Same number, different letter — relative major/minor (1).
    """
    if not code_a or not code_b or code_a == "?" or code_b == "?":
        return -1
    try:
        num_a = int(code_a[:-1])
        letter_a = code_a[-1].upper()
        num_b = int(code_b[:-1])
        letter_b = code_b[-1].upper()
    except (ValueError, IndexError):
        return -1

    if letter_a not in ("A", "B") or letter_b not in ("A", "B"):
        return -1
    if not 1 <= num_a <= 12 or not 1 <= num_b <= 12:
        return -1

    if num_a == num_b and letter_a == letter_b:
        return 0
    if num_a == num_b and letter_a != letter_b:
        return 1

    diff = abs(num_a - num_b)
    circular = min(diff, 12 - diff)

    if letter_a == letter_b:
        return circular
    return circular + 1


def key_compatibility_score(code_a: str, code_b: str) -> float:
    """Score harmonic compatibility between two Camelot codes.

    Returns a value in ``[0.0, 1.0]`` where 1.0 is perfectly compatible and
    0.0 is maximally distant. Returns 0.5 (neutral) when either code is
    unknown.
    """
    dist = camelot_distance(code_a, code_b)
    if dist < 0:
        return 0.5
    if dist == 0:
        return 1.0
    if dist == 1:
        return 0.9
    if dist == 2:
        return 0.6
    return max(0.0, 1.0 - dist * 0.15)


# ---------------------------------------------------------------------------
# Downbeat estimation from beat times
# ---------------------------------------------------------------------------

def estimate_downbeats(
    mono: np.ndarray,
    sample_rate: int,
    beat_times_s: tuple[float, ...] | list[float],
    *,
    beats_per_bar: int = 4,
) -> tuple[float, ...]:
    """Estimate downbeat (bar-start) positions from beat times using energy.

    Computes RMS energy at each beat position and finds the phase offset
    (0 to ``beats_per_bar - 1``) that maximizes energy on downbeats. This is
    a heuristic assuming 4/4 meter; it does not detect time-signature changes.

    Returns a tuple of downbeat times in seconds. May be empty if fewer than
    ``beats_per_bar`` beats are provided.
    """
    beats = sorted(
        float(t) for t in beat_times_s if 0.0 <= float(t) <= len(mono) / sample_rate
    )
    if len(beats) < beats_per_bar:
        return ()

    # Compute RMS energy in a short window around each beat
    half_window = max(1, int(0.05 * sample_rate))  # 50ms window
    energies = np.zeros(len(beats), dtype=np.float64)
    n_samples = len(mono)

    for i, beat_time in enumerate(beats):
        center = int(round(beat_time * sample_rate))
        start = max(0, center - half_window)
        end = min(n_samples, center + half_window)
        if end > start:
            energies[i] = float(np.sqrt(np.mean(np.square(mono[start:end]))))

    # Try each phase offset and score
    best_phase = 0
    best_score = -1.0

    for phase in range(beats_per_bar):
        # Select beats at this phase offset
        downbeat_indices = list(range(phase, len(beats), beats_per_bar))
        if not downbeat_indices:
            continue
        score = float(np.mean(energies[downbeat_indices]))
        if score > best_score:
            best_score = score
            best_phase = phase

    # Collect downbeat times at the best phase
    downbeat_times = tuple(
        beats[i] for i in range(best_phase, len(beats), beats_per_bar)
    )
    return downbeat_times


__all__ = [
    "camelot_distance",
    "detect_key",
    "estimate_downbeats",
    "key_compatibility_score",
]

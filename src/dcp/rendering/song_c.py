"""
Song C Synthesized Elements Generator.
Generates bespoke transitional production elements:
1. Noise Riser: White/pink noise with exponential bandpass/highpass sweep and rhythmic sidechain pump.
2. Reverse Crash: Time-reversed cymbal swell that psychoacoustically sucks energy into the drop.
3. Sub-Bass 808 Slam: Tuned sine sub drop (35-65 Hz) at the impact downbeat.
"""

from __future__ import annotations
import numpy as np
from scipy import signal
from typing import Dict, Any


def generate_noise_riser(
    duration_s: float,
    sr: int,
    start_hz: float = 250.0,
    end_hz: float = 12000.0,
    level_db: float = -14.0,
    sidechain_bpm: float = 126.0,
) -> np.ndarray:
    """Generate stereo noise riser with exponential filter sweep and sidechain ducking."""
    n_samples = max(64, int(round(duration_s * sr)))
    t = np.arange(n_samples, dtype=np.float64) / sr
    t_norm = t / max(duration_s, 1e-4)

    # Gaussian white noise
    rng = np.random.RandomState(42)  # Deterministic seed for reproducible testing
    noise = rng.randn(2, n_samples).astype(np.float32)

    # Exponential cutoff frequency trajectory
    f_start = max(40.0, min(start_hz, sr * 0.45))
    f_end = max(40.0, min(end_hz, sr * 0.45))
    cutoffs = f_start * np.power(f_end / f_start, t_norm)

    # Process in 512-sample blocks with a 2nd-order butterworth bandpass/highpass
    block_size = 512
    out = np.zeros_like(noise)
    n_blocks = int(np.ceil(n_samples / block_size))

    # Continuous filter states across blocks: shape (n_sections=2, n_channels=2, 2)
    zi = np.zeros((2, 2, 2), dtype=np.float64)

    for b in range(n_blocks):
        s0 = b * block_size
        s1 = min(s0 + block_size, n_samples)
        fc = cutoffs[s0]
        # Bandpass between fc and min(sr*0.48, fc*2.5)
        f_low = max(20.0, fc * 0.5)
        f_high = min(sr * 0.48, max(f_low + 50.0, fc * 2.0))
        sos = signal.butter(2, [f_low / (0.5 * sr), f_high / (0.5 * sr)], btype="bandpass", output="sos")
        chunk = noise[:, s0:s1].astype(np.float64)
        filtered, zi = signal.sosfilt(sos, chunk, axis=1, zi=zi)
        out[:, s0:s1] = filtered.astype(np.float32)

    # Exponential volume ramp (slow start -> accelerating surge into drop)
    amp_ramp = np.power(t_norm, 1.8).astype(np.float32)
    out *= amp_ramp[np.newaxis, :]

    # Quarter-note sidechain ducking envelope synced to BPM
    if sidechain_bpm > 20.0:
        beat_dur = 60.0 / sidechain_bpm
        beat_phase = (t % beat_dur) / beat_dur
        # Pump shape: dip to 0.35 at beat onset, recovery in first 65% of beat
        pump = 0.35 + 0.65 * np.clip(beat_phase / 0.65, 0.0, 1.0)
        out *= pump.astype(np.float32)[np.newaxis, :]

    linear_gain = float(np.power(10.0, level_db / 20.0))
    return out * linear_gain


def generate_reverse_crash(
    duration_s: float = 1.8,
    sr: int = 48000,
    level_db: float = -9.0,
) -> np.ndarray:
    """Generate a reverse crash / cymbal swell that peaks at the very last sample."""
    n_samples = max(64, int(round(duration_s * sr)))
    t = np.arange(n_samples, dtype=np.float64) / sr

    # Forward crash: bandpassed white noise (3-9 kHz) with exponential decay
    rng = np.random.RandomState(1337)
    noise = rng.randn(2, n_samples).astype(np.float32)
    sos = signal.butter(4, [2500.0 / (0.5 * sr), min(sr * 0.45, 10000.0) / (0.5 * sr)], btype="bandpass", output="sos")
    filtered = signal.sosfilt(sos, noise.astype(np.float64), axis=1).astype(np.float32)

    # Exponential decay envelope for forward crash (T60 ~ 1.5s)
    decay = np.exp(-t * 3.5).astype(np.float32)
    forward = filtered * decay[np.newaxis, :]

    # Time reverse to create the reverse swell
    reverse = np.ascontiguousarray(forward[:, ::-1])

    # Smooth fade-in at the beginning so it rises seamlessly out of the background
    fade_len = min(n_samples // 4, int(0.25 * sr))
    if fade_len > 0:
        fade_in = (0.5 * (1.0 - np.cos(np.pi * np.arange(fade_len) / fade_len))).astype(np.float32)
        reverse[:, :fade_len] *= fade_in[np.newaxis, :]

    linear_gain = float(np.power(10.0, level_db / 20.0))
    return reverse * linear_gain


def generate_sub_bass_808(
    duration_s: float = 1.8,
    sr: int = 48000,
    root_pitch_class: int = 2,  # 0=C, 2=D, 7=G, etc.
    octave_hz: float = 43.65,    # Default F0 around low F/G/D
    level_db: float = -6.0,
) -> np.ndarray:
    """Generate an 808-style sine sub-bass slam with pitch drop and soft tube saturation."""
    n_samples = max(64, int(round(duration_s * sr)))
    t = np.arange(n_samples, dtype=np.float64) / sr

    # Compute base frequency from root pitch class (in 35 - 65 Hz sub range)
    # A0 = 27.5 Hz, C1 = 32.7 Hz. Pitch class 0 (C1) = 32.7, 7 (G1) = 49.0, 2 (D1) = 36.7
    f0 = 32.703 * (2.0 ** (root_pitch_class / 12.0))
    while f0 > 65.0:
        f0 *= 0.5
    while f0 < 35.0:
        f0 *= 2.0

    # Fast initial pitch dive (+1 octave dropping within 40ms)
    pitch_dive = 1.0 + 1.2 * np.exp(-t * 35.0)
    inst_freq = f0 * pitch_dive
    phase = 2.0 * np.pi * np.cumsum(inst_freq) / sr

    sine = np.sin(phase).astype(np.float32)

    # Amplitude ADSR envelope: 3ms attack, exponential decay
    attack_samples = max(1, int(0.003 * sr))
    env = np.ones(n_samples, dtype=np.float32)
    env[:attack_samples] = np.linspace(0.0, 1.0, attack_samples, dtype=np.float32)
    env[attack_samples:] = np.exp(-3.2 * (t[attack_samples:] - t[attack_samples])).astype(np.float32)

    sub = sine * env

    # Mild asymmetric tube saturation (tanh drive) adds 2nd & 3rd harmonics for smaller speaker audibility
    driven = np.tanh(sub * 1.8) * 0.7
    stereo = np.vstack([driven, driven]).astype(np.float32)

    linear_gain = float(np.power(10.0, level_db / 20.0))
    return stereo * linear_gain


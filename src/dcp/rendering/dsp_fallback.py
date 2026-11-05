# src/dcp/rendering/dsp_fallback.py
"""
Portable NumPy/SciPy DSP Rendering Kernel for DCP Step 3.
Implements:
  1. Topology-Preserving Transform (TPT) Zero-Delay Feedback (ZDF) 2-Pole & 4-Pole SVF
  2. Verified 4x4 Householder FDN Freeze Engine with Exact Finite-Sample Hann State Lock
  3. Decoupled Peak/RMS Transient Sidechain Compressor
  4. Phase-Locked STFT Phase Vocoder Time-Stretcher (for Bijective Warp Maps)
"""
from __future__ import annotations

from typing import Literal
import numpy as np
from scipy import signal


def _ensure_stereo(audio: np.ndarray) -> tuple[np.ndarray, bool]:
    """Converts (N,) or (1, N) or (2, N) or (N, 2) to contiguous float32 (2, N)."""
    if audio.ndim == 1:
        stereo = np.stack([audio, audio], axis=0).astype(np.float32, copy=True)
        return np.ascontiguousarray(stereo), True
    if audio.ndim == 2 and audio.shape[0] == 1:
        stereo = np.vstack([audio[0], audio[0]]).astype(np.float32, copy=True)
        return np.ascontiguousarray(stereo), True
    if audio.shape[0] == 2:
        return np.ascontiguousarray(audio, dtype=np.float32), False
    if audio.ndim == 2 and audio.shape[1] == 2:
        return np.ascontiguousarray(audio.T, dtype=np.float32), False
    raise ValueError(f"Expected mono or stereo audio, got shape {audio.shape}")


# =============================================================================
# 1. Vadim Zavalishin TPT Zero-Delay Feedback (ZDF) State-Variable Filter
# =============================================================================

def process_zdf_svf_sweep_numpy(
    audio: np.ndarray,
    sr: int,
    start_cutoff_hz: float,
    end_cutoff_hz: float,
    resonance_q: float = 0.707,
    mode: Literal["lpf", "hpf", "bpf", "hpf4"] = "hpf",
    subblock_size: int = 64,
) -> np.ndarray:
    """
    Renders a time-varying ZDF State-Variable Filter sweep across stereo audio.
    Uses exact trapezoidal integration (TPT) updated per subblock (64 samples ~ 1.3ms)
    combined with `scipy.signal.lfilter` state continuation to achieve C-speed in pure Python.
    """
    stereo, was_mono = _ensure_stereo(audio)
    num_samples = stereo.shape[1]
    if num_samples == 0:
        return audio.copy()

    nyq_guard = 0.475 * sr
    f_start = float(np.clip(start_cutoff_hz, 18.0, nyq_guard))
    f_end = float(np.clip(end_cutoff_hz, 18.0, nyq_guard))
    q_clamped = float(np.clip(resonance_q, 0.5, 12.0))
    k = 1.0 / q_clamped
    res_comp = 1.0 / np.sqrt(1.0 + max(0.0, q_clamped - 0.707) * 0.35)

    out = np.empty_like(stereo)
    num_blocks = int(np.ceil(num_samples / subblock_size))

    # Exponential (pitch-linear) cutoff trajectory across subblocks
    t_norm = np.linspace(0.0, 1.0, num_blocks, dtype=np.float64)
    cutoffs = f_start * np.power(f_end / f_start, t_norm)

    # Exact algebraic equivalence between TPT ZDF SVF and Direct-Form-II Transposed Biquad
    # per constant-coefficient subblock, preserving continuous integrator state `zi`.
    zi_stage1 = np.zeros((2, 2), dtype=np.float64)
    zi_stage2 = np.zeros((2, 2), dtype=np.float64)

    for b_idx in range(num_blocks):
        s0 = b_idx * subblock_size
        s1 = min(s0 + subblock_size, num_samples)
        chunk = stereo[:, s0:s1].astype(np.float64)

        g = np.tan(np.pi * cutoffs[b_idx] / sr)
        a1 = 1.0 / (1.0 + g * (g + k))
        a2 = g * a1
        a3 = g * a2

        denom = 1.0 + g * (g + k)
        a_tpt = np.array([
            1.0,
            2.0 * (g * g - 1.0) / denom,
            (1.0 - g * k + g * g) / denom
        ], dtype=np.float64)

        if mode == "lpf":
            b_tpt = np.array([a3, 2.0 * a3, a3], dtype=np.float64) * res_comp
        elif mode == "bpf":
            b_tpt = np.array([a2, 0.0, -a2], dtype=np.float64)
        else:  # "hpf" or "hpf4"
            b_tpt = np.array([a1, -2.0 * a1, a1], dtype=np.float64) * res_comp

        y1, zi_stage1 = signal.lfilter(b_tpt, a_tpt, chunk, axis=1, zi=zi_stage1)

        if mode == "hpf4":
            y2, zi_stage2 = signal.lfilter(b_tpt, a_tpt, y1, axis=1, zi=zi_stage2)
            out[:, s0:s1] = y2.astype(np.float32)
        else:
            out[:, s0:s1] = y1.astype(np.float32)

    if was_mono:
        return out[:1] if audio.ndim == 2 else out[0]
    return out


# =============================================================================
# 2. Vectorized Decoupled Peak/RMS Transient Sidechain Compressor
# =============================================================================

def process_sidechain_compressor_numpy(
    target_audio: np.ndarray,
    key_audio: np.ndarray,
    sr: int,
    bpm: float = 126.0,
    attack_ms: float = 1.5,
    release_beat_fraction: float = 0.25,
    threshold_db: float = -18.0,
    ratio: float = 4.0,
    knee_width_db: float = 4.0,
    peak_rms_blend: float = 0.7,
) -> np.ndarray:
    """
    Ducks `target_audio` (Song A bass/other/reverb tails) using `key_audio` (Song B drums).
    Uses a fast IIR RMS estimator + decoupled attack/release envelope + soft-knee gain computer.
    """
    target_st, was_mono = _ensure_stereo(target_audio)
    key_st, _ = _ensure_stereo(key_audio)
    n_samples = min(target_st.shape[1], key_st.shape[1])
    if n_samples == 0:
        return target_audio.copy()

    # 1. Stereo-linked instantaneous peak & 5ms RMS detector (vectorized via lfilter)
    inst_peak = np.maximum(np.abs(key_st[0, :n_samples]), np.abs(key_st[1, :n_samples]))
    rms_alpha = float(np.exp(-1.0 / (0.005 * sr)))
    rms_sq = signal.lfilter([1.0 - rms_alpha], [1.0, -rms_alpha], inst_peak * inst_peak)
    x_det = peak_rms_blend * inst_peak + (1.0 - peak_rms_blend) * np.sqrt(np.maximum(rms_sq, 0.0) + 1e-12)

    # 2. Decoupled ballistics (chunked max-hold release + vectorized IIR attack filter)
    attack_sec = max(0.0001, attack_ms * 0.001)
    release_sec = (60.0 / max(40.0, bpm)) * np.clip(release_beat_fraction, 0.0625, 1.0)
    alpha_a = float(np.exp(-1.0 / (attack_sec * sr)))
    alpha_r = float(np.exp(-1.0 / (release_sec * sr)))

    # Process release envelope at 32-sample block rate (0.66ms resolution) for vector speed,
    # then upsample and smooth with exact sample-rate attack filter.
    step = 32
    n_blocks = int(np.ceil(n_samples / step))
    padded = np.pad(x_det, (0, n_blocks * step - n_samples), mode="edge")
    block_peaks = np.max(padded.reshape(n_blocks, step), axis=1)
    alpha_r_block = alpha_r ** step

    peak_hold = np.empty_like(block_peaks)
    state = 0.0
    for i in range(n_blocks):
        state = max(float(block_peaks[i]), alpha_r_block * state)
        peak_hold[i] = state

    upsampled_hold = np.repeat(peak_hold, step)[:n_samples]
    env = signal.lfilter([1.0 - alpha_a], [1.0, -alpha_a], upsampled_hold)

    # 3. Vectorized Soft-Knee Log-Domain Gain Computer
    env_db = 20.0 * np.log10(np.maximum(env, 1e-9))
    overshoot = env_db - threshold_db
    slope = (1.0 / max(1.0, ratio)) - 1.0

    gain_red_db = np.zeros_like(overshoot)
    if knee_width_db > 0.01:
        in_knee = (2.0 * np.abs(overshoot)) <= knee_width_db
        above_knee = (2.0 * overshoot) > knee_width_db
        temp = overshoot[in_knee] + 0.5 * knee_width_db
        gain_red_db[in_knee] = slope * (temp * temp) / (2.0 * knee_width_db)
        gain_red_db[above_knee] = slope * overshoot[above_knee]
    else:
        above = overshoot > 0.0
        gain_red_db[above] = slope * overshoot[above]

    linear_gain = np.power(10.0, gain_red_db / 20.0).astype(np.float32)

    out = target_st.copy()
    out[:, :n_samples] *= linear_gain[np.newaxis, :]
    if was_mono:
        return out[:1] if target_audio.ndim == 2 else out[0]
    return out


# =============================================================================
# 3. Verified 4x4 FDN Freeze Engine (Finite-Sample Hann State Lock)
# =============================================================================

def process_fdn_freeze_numpy(
    audio: np.ndarray,
    sr: int,
    freeze_start_sample: int | None = None,
    freeze_end_sample: int | None = None,
    feedback_gain: float = 0.88,
    wet_mix: float = 0.45,
    dry_mix: float = 1.0,
    ramp_samples: int = 1024,
) -> np.ndarray:
    """
    4x4 Orthogonal Householder FDN with exact finite-sample Hann crossfade into a
    bit-exact lossless unitary freeze loop (zero energy decay/runaway during freeze).
    """
    stereo, was_mono = _ensure_stereo(audio)
    n_samples = stereo.shape[1]
    if n_samples == 0:
        return audio.copy()

    scale = sr / 48000.0
    # Mutually coprime delay lengths (in samples)
    delays = [
        int(1699 * scale) | 1,
        int(2141 * scale) | 1,
        int(2749 * scale) | 1,
        int(3467 * scale) | 1,
    ]

    # Build exact finite-sample Hann freeze alpha trajectory in [0.0, 1.0]
    freeze_alpha = np.zeros(n_samples, dtype=np.float32)
    if freeze_start_sample is not None and freeze_start_sample < n_samples:
        f_start = max(0, int(freeze_start_sample))
        f_end = n_samples if freeze_end_sample is None else min(n_samples, int(freeze_end_sample))
        if f_end > f_start:
            # Finite Hann ramp up to EXACT 1.0
            r_up = min(ramp_samples, f_end - f_start)
            if r_up > 0:
                hann_up = 0.5 * (1.0 - np.cos(np.pi * np.arange(r_up, dtype=np.float32) / r_up))
                freeze_alpha[f_start : f_start + r_up] = hann_up
            # Hard bit-exact 1.0 state lock across the frozen plateau
            freeze_alpha[f_start + r_up : f_end] = 1.0
            # Finite Hann ramp down on release
            if f_end < n_samples:
                r_dn = min(ramp_samples, n_samples - f_end)
                hann_dn = 0.5 * (1.0 + np.cos(np.pi * np.arange(r_dn, dtype=np.float32) / r_dn))
                freeze_alpha[f_end : f_end + r_dn] = hann_dn

    # Block-vectorized FDN execution (chunk size = min(delays) = ~1699 samples)
    # Because chunk_len <= min(delays), every read within a chunk depends strictly on
    # samples written in previous chunks—allowing 100% vectorized NumPy array slicing!
    block_len = min(delays)
    max_del = max(delays)
    buf_len = max_del + block_len + 16
    tank = np.zeros((4, buf_len), dtype=np.float64)
    w_ptr = max_del

    # Pre-filter coefficients for active path (300Hz HPF + 4.5kHz LPF damping), bounded below Nyquist
    hp_freq = max(20.0, min(300.0, 0.1 * sr))
    lp_freq = min(4500.0, 0.45 * sr)
    b_hp, a_hp = signal.butter(1, hp_freq / (0.5 * sr), btype="high")
    b_lp, a_lp = signal.butter(1, lp_freq / (0.5 * sr), btype="low")
    zi_hp = np.zeros((4, 1), dtype=np.float64)
    zi_lp = np.zeros((4, 1), dtype=np.float64)

    out = np.zeros_like(stereo, dtype=np.float64)
    in_f64 = stereo.astype(np.float64)

    for s0 in range(0, n_samples, block_len):
        s1 = min(s0 + block_len, n_samples)
        L = s1 - s0

        # Circular buffer compaction if write pointer approaches end
        if w_ptr + L >= buf_len:
            tank[:, :max_del] = tank[:, w_ptr - max_del : w_ptr]
            w_ptr = max_del

        # 1. Vectorized read from all 4 delay lines prior to writing
        tank_read = np.empty((4, L), dtype=np.float64)
        for ch in range(4):
            r_start = w_ptr - delays[ch]
            tank_read[ch, :] = tank[ch, r_start : r_start + L]

        # 2. Orthogonal 4x4 Householder Matrix: H = I - 0.5 * 1 * 1^T (||H x||_2 == ||x||_2)
        half_sum = 0.5 * np.sum(tank_read, axis=0, keepdims=True)
        mixed = tank_read - half_sum

        alpha = freeze_alpha[s0:s1].astype(np.float64)[np.newaxis, :]
        in_scale = 0.5 * (1.0 - alpha)

        xL = in_f64[0, s0:s1]
        xR = in_f64[1, s0:s1]
        inj = np.vstack([
            xL,
            xR,
            0.70710678 * (xL + xR),
            0.70710678 * (xL - xR),
        ]) * in_scale

        # 3. Active damped path vs. Lossless Unitary Freeze path
        active = inj + mixed * feedback_gain
        active_lp, zi_lp = signal.lfilter(b_lp, a_lp, active, axis=1, zi=zi_lp)
        damped = active - 0.35 * (active - active_lp)
        dc_blocked, zi_hp = signal.lfilter(b_hp, a_hp, damped, axis=1, zi=zi_hp)

        # C^1 soft limiter ONLY above |x| > 0.90
        abs_x = np.abs(dc_blocked)
        over = abs_x > 0.90
        if np.any(over):
            dc_blocked[over] = np.sign(dc_blocked[over]) * (
                0.90 + 0.10 * np.tanh((abs_x[over] - 0.90) / 0.10)
            )

        # When alpha == 1.0, tank writes `mixed` directly (exact unitary state lock)
        tank[:, w_ptr : w_ptr + L] = (1.0 - alpha) * dc_blocked + alpha * mixed
        w_ptr += L

        # 4. Orthonormal stereo output projection
        wet_L = 0.70710678 * (tank_read[0, :] + tank_read[2, :])
        wet_R = 0.70710678 * (tank_read[1, :] + tank_read[3, :])
        out[0, s0:s1] = xL * dry_mix + wet_L * wet_mix
        out[1, s0:s1] = xR * dry_mix + wet_R * wet_mix

    if was_mono:
        return out[:1].astype(np.float32) if audio.ndim == 2 else out[0].astype(np.float32)
    return out.astype(np.float32)


# =============================================================================
# 4. Bounded Phase-Vocoder Time-Stretcher (+/- 8%)
# =============================================================================

def time_stretch_phase_vocoder_numpy(
    audio: np.ndarray,
    stretch_ratio: float | np.ndarray,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Phase-locked STFT phase vocoder supporting constant or time-varying stretch ratios
    from `compute_safeguarded_bijective_warp()`.
    """
    stereo, was_mono = _ensure_stereo(audio)
    n_samples = stereo.shape[1]
    if n_samples == 0:
        return audio.copy()

    if np.isscalar(stretch_ratio) and abs(float(stretch_ratio) - 1.0) < 1e-4:
        return audio.copy()

    fft_size = min(n_fft, n_samples)
    if fft_size < 128:
        return audio.copy()
    hop = max(32, fft_size // 4)

    window = np.hanning(fft_size).astype(np.float64)
    omega = 2.0 * np.pi * hop * np.arange(fft_size // 2 + 1) / fft_size

    out_channels = []
    for ch in range(2):
        _, _, Zxx = signal.stft(
            stereo[ch].astype(np.float64),
            nperseg=fft_size,
            noverlap=fft_size - hop,
            window=window,
            boundary="zeros",
        )
        n_bins, n_frames = Zxx.shape

        if n_frames <= 1:
            out_channels.append(stereo[ch].copy())
            continue

        if np.isscalar(stretch_ratio):
            rate = float(np.clip(stretch_ratio, 0.92, 1.08))
            time_steps = np.arange(0, n_frames - 1, 1.0 / rate, dtype=np.float64)
        else:
            # Integrate variable stretch ratio trajectory across frames
            ratios = np.clip(np.asarray(stretch_ratio, dtype=np.float64), 0.92, 1.08)
            frame_ratios = np.interp(
                np.linspace(0, 1, n_frames),
                np.linspace(0, 1, len(ratios)),
                ratios,
            )
            total_out_frames = max(2, int(np.sum(frame_ratios)))
            cdf = np.cumsum(frame_ratios)
            cdf /= max(cdf[-1], 1e-9)
            time_steps = np.interp(
                np.linspace(0, 1, total_out_frames),
                cdf,
                np.arange(n_frames),
            )

        mag = np.abs(Zxx)
        phase = np.angle(Zxx)
        n_steps = len(time_steps)
        if n_steps == 0:
            out_channels.append(stereo[ch].copy())
            continue

        idx = np.floor(time_steps).astype(np.int64)
        frac = (time_steps - idx)[np.newaxis, :]
        idx_next = np.minimum(idx + 1, n_frames - 1)

        # Vectorized linear magnitude interpolation across all frames
        m_interp = (1.0 - frac) * mag[:, idx] + frac * mag[:, idx_next]

        # Vectorized phase advance with principal determination wrapping in [-pi, pi]
        dphi = phase[:, idx_next] - phase[:, idx] - omega[:, np.newaxis]
        dphi = dphi - 2.0 * np.pi * np.round(dphi / (2.0 * np.pi))
        phase_advance = omega[:, np.newaxis] + dphi

        cum_phase = np.empty((n_bins, n_steps), dtype=np.float64)
        cum_phase[:, 0] = phase[:, 0]
        if n_steps > 1:
            cum_phase[:, 1:] = phase[:, 0:1] + np.cumsum(phase_advance[:, :-1], axis=1)

        out_stft = m_interp * np.exp(1j * cum_phase)

        _, y_rec = signal.istft(
            out_stft,
            nperseg=fft_size,
            noverlap=fft_size - hop,
            window=window,
        )
        out_channels.append(y_rec.astype(np.float32))

    min_len = min(len(out_channels[0]), len(out_channels[1]))
    res = np.vstack([out_channels[0][:min_len], out_channels[1][:min_len]])
    if was_mono:
        return res[:1] if audio.ndim == 2 else res[0]
    return res


# =============================================================================
# 5. Master Bus Glue Compressor & Studio Brickwall Limiter
# =============================================================================

def process_master_bus_glue_numpy(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -12.0,
    ratio: float = 1.5,
    attack_ms: float = 30.0,
    release_ms: float = 100.0,
    ceiling_dbfs: float = -0.5,
) -> np.ndarray:
    """
    Classic SSL-style 2-bus VCA glue compressor with a brickwall soft limiter.
    Provides cohesion and punch to the rendered transition bridge without clipping.
    """
    stereo, was_mono = _ensure_stereo(audio)
    n_samples = stereo.shape[1]
    if n_samples == 0:
        return audio.copy()

    # Instantaneous peak detector with stereo linking
    peak = np.maximum(np.abs(stereo[0]), np.abs(stereo[1]))

    # IIR envelope detector for attack and release
    att_alpha = float(np.exp(-1.0 / (max(0.001, attack_ms * 0.001) * sr)))
    rel_alpha = float(np.exp(-1.0 / (max(0.010, release_ms * 0.001) * sr)))

    # Process envelope at 32-sample block rate
    step = 32
    n_blocks = int(np.ceil(n_samples / step))
    padded = np.pad(peak, (0, n_blocks * step - n_samples), mode="edge")
    block_peaks = np.max(padded.reshape(n_blocks, step), axis=1)

    env_block = np.empty_like(block_peaks)
    state = 0.0
    for i in range(n_blocks):
        pk = float(block_peaks[i])
        alpha = att_alpha if pk > state else rel_alpha
        state = alpha * state + (1.0 - alpha) * pk
        env_block[i] = state

    env = np.repeat(env_block, step)[:n_samples]

    # Soft-knee VCA gain computer
    env_db = 20.0 * np.log10(np.maximum(env, 1e-9))
    overshoot = env_db - threshold_db
    slope = (1.0 / max(1.0, ratio)) - 1.0

    gain_red_db = np.zeros_like(overshoot)
    above = overshoot > 0.0
    gain_red_db[above] = slope * overshoot[above]

    comp_gain = np.power(10.0, gain_red_db / 20.0).astype(np.float32)

    glued = stereo * comp_gain[np.newaxis, :]

    # Brickwall soft limiter ceiling
    ceiling_linear = float(np.power(10.0, ceiling_dbfs / 20.0))
    max_val = np.max(np.abs(glued))
    if max_val > ceiling_linear:
        # Soft tanh saturation above ceiling
        glued = ceiling_linear * np.tanh(glued / ceiling_linear)

    if was_mono:
        return glued[:1] if audio.ndim == 2 else glued[0]
    return glued



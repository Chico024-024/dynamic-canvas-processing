# src/dcp/transitions/bijective_warp.py
"""
Safeguarded Bijective Phase-to-Time Warping & Robust Quantile Normalization.
Replaces naive min-max scaling and constant-offset drift across variable-BPM bridges.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BijectiveWarpMap:
    """Analytical phase-to-time mapping across a variable-tempo transition bridge."""

    total_output_samples: int
    total_bridge_sec: float
    out_sample_grid: np.ndarray     # Shape (M,)
    beat_phase_phi: np.ndarray      # Shape (M,) in [0, N_b]
    src_sample_a: np.ndarray        # Shape (M,): Exact fractional sample in Song A
    src_sample_b: np.ndarray        # Shape (M,): Exact fractional sample in Song B
    stretch_ratio_a: np.ndarray     # Instantaneous d(t_out)/d(t_src_a)
    stretch_ratio_b: np.ndarray     # Instantaneous d(t_out)/d(t_src_b)


def robust_quantile_normalize(
    x: np.ndarray,
    p_low: float = 10.0,
    p_high: float = 90.0,
) -> np.ndarray:
    """Outlier-resistant normalization using inter-quantile range [P10, P90]."""
    if x.size == 0:
        return x
    lo, hi = np.percentile(x, [p_low, p_high])
    span = hi - lo
    if span < 1e-9:
        ptp = np.ptp(x)
        return (x - np.min(x)) / (ptp if ptp > 1e-9 else 1.0)
    return np.clip((x - lo) / span, 0.0, 1.0)


def compute_safeguarded_bijective_warp(
    beat_samples_a: np.ndarray,
    beat_samples_b: np.ndarray,
    sr: int = 48000,
    control_hop: int = 256,
    tol_sec: float = 1e-12,
    max_iter: int = 32,
) -> BijectiveWarpMap:
    """
    Maps bridge beat-phase phi in [0, N_b] to output clock time t_out via exact analytical
    integration of beat duration S(phi), inverted via Bracketed Newton-Bisection.
    Guarantees 0.000 ms coincidence at both Launchpad (phi=0) and Impact (phi=N_b).
    """
    N_b = len(beat_samples_a) - 1
    if len(beat_samples_b) - 1 != N_b or N_b <= 0:
        raise ValueError(f"Beat grids must have identical positive beat count N_b (got {N_b}).")

    S_A = float(beat_samples_a[1] - beat_samples_a[0]) / sr
    S_B = float(beat_samples_b[-1] - beat_samples_b[-2]) / sr

    half_sum = 0.5 * (S_A + S_B)
    half_diff = 0.5 * (S_B - S_A)
    pi_over_Nb = np.pi / N_b

    total_bridge_sec = N_b * half_sum
    total_output_samples = int(np.round(total_bridge_sec * sr))

    out_samples = np.arange(0, total_output_samples, control_hop, dtype=np.float64)
    if len(out_samples) == 0 or out_samples[-1] != total_output_samples:
        out_samples = np.append(out_samples, float(total_output_samples))
    t_target = out_samples / sr

    # Bracketed Newton-Raphson (Newton-Bisection hybrid) with residual stopping criterion
    phi_lo = np.zeros_like(t_target)
    phi_hi = np.full_like(t_target, float(N_b))
    phi = (t_target / max(total_bridge_sec, 1e-9)) * N_b

    for _ in range(max_iter):
        t_val = half_sum * phi - half_diff * (N_b / np.pi) * np.sin(pi_over_Nb * phi)
        residual = t_val - t_target
        if np.max(np.abs(residual)) <= tol_sec:
            break

        # Update brackets
        phi_lo = np.where(residual < 0.0, phi, phi_lo)
        phi_hi = np.where(residual > 0.0, phi, phi_hi)

        deriv = np.maximum(half_sum - half_diff * np.cos(pi_over_Nb * phi), 1e-9)
        newton_candidate = phi - residual / deriv

        # Safeguard: if Newton steps outside bracket [phi_lo, phi_hi], fall back to bisection
        out_of_bounds = (newton_candidate <= phi_lo) | (newton_candidate >= phi_hi)
        phi = np.where(out_of_bounds, 0.5 * (phi_lo + phi_hi), newton_candidate)

    phi[0] = 0.0
    phi[-1] = float(N_b)

    k = np.minimum(np.floor(phi).astype(np.int64), N_b - 1)
    delta = phi - k

    src_a = (1.0 - delta) * beat_samples_a[k] + delta * beat_samples_a[k + 1]
    src_b = (1.0 - delta) * beat_samples_b[k] + delta * beat_samples_b[k + 1]

    S_phi = half_sum - half_diff * np.cos(pi_over_Nb * phi)
    local_S_a = np.maximum((beat_samples_a[k + 1] - beat_samples_a[k]) / sr, 1e-6)
    local_S_b = np.maximum((beat_samples_b[k + 1] - beat_samples_b[k]) / sr, 1e-6)

    return BijectiveWarpMap(
        total_output_samples=total_output_samples,
        total_bridge_sec=total_bridge_sec,
        out_sample_grid=out_samples,
        beat_phase_phi=phi,
        src_sample_a=src_a,
        src_sample_b=src_b,
        stretch_ratio_a=S_phi / local_S_a,
        stretch_ratio_b=S_phi / local_S_b,
    )


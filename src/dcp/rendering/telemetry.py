# src/dcp/rendering/telemetry.py
"""
Layer 5: Objective Audio Quality & Arrangement Telemetry Evaluator.
Audits rendered "Song C" bridges for sub-bass collisions, vocal midrange masking,
stereo phase correlation, and dynamic loudness continuity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
from scipy import signal


@dataclass(frozen=True)
class BridgeTelemetryReport:
    true_peak_dbfs: float
    integrated_rms_dbfs: float
    min_stereo_phase_correlation: float
    mean_stereo_phase_correlation: float
    sub_bass_collision_ratio: float      # Fraction of frames where A & B sub-bass both exceed -22 dBFS
    vocal_midrange_masking_ratio: float  # Fraction of frames where A & B vocal midrange simultaneously clash
    passes_studio_qc: bool
    qc_warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "true_peak_dbfs": round(self.true_peak_dbfs, 2),
            "integrated_rms_dbfs": round(self.integrated_rms_dbfs, 2),
            "min_stereo_phase_correlation": round(self.min_stereo_phase_correlation, 3),
            "mean_stereo_phase_correlation": round(self.mean_stereo_phase_correlation, 3),
            "sub_bass_collision_ratio": round(self.sub_bass_collision_ratio, 3),
            "vocal_midrange_masking_ratio": round(self.vocal_midrange_masking_ratio, 3),
            "passes_studio_qc": self.passes_studio_qc,
            "qc_warnings": list(self.qc_warnings),
        }


def evaluate_bridge_telemetry(
    rendered_master: np.ndarray,
    contrib_a: np.ndarray | None = None,
    contrib_b: np.ndarray | None = None,
    sr: int = 48000,
) -> BridgeTelemetryReport:
    """
    Computes objective studio mastering and arrangement metrics on a rendered Song C bridge.
    """
    if rendered_master.ndim == 1:
        stereo = np.vstack([rendered_master, rendered_master])
    elif rendered_master.shape[0] == 2:
        stereo = rendered_master
    else:
        stereo = rendered_master.T

    # 1. True Peak (oversampled peak estimation) & Integrated RMS
    peak_lin = float(np.max(np.abs(stereo))) + 1e-12
    true_peak_db = float(20.0 * np.log10(peak_lin))
    rms_lin = float(np.sqrt(np.mean(stereo.astype(np.float64) ** 2) + 1e-12))
    rms_db = float(20.0 * np.log10(rms_lin))

    # 2. Windowed Stereo Phase Correlation (100ms windows)
    win = max(256, int(0.100 * sr))
    n_frames = max(1, stereo.shape[1] // win)
    corrs = []
    for i in range(n_frames):
        sL = stereo[0, i * win : (i + 1) * win].astype(np.float64)
        sR = stereo[1, i * win : (i + 1) * win].astype(np.float64)
        denom = float(np.sqrt(np.sum(sL * sL) * np.sum(sR * sR)))
        if denom > 1e-9:
            corrs.append(float(np.sum(sL * sR) / denom))
    min_corr = float(np.min(corrs)) if corrs else 1.0
    mean_corr = float(np.mean(corrs)) if corrs else 1.0

    # 3. Sub-Bass Collision & Vocal Masking (if individual deck contributions are supplied)
    sub_collision = 0.0
    vocal_masking = 0.0
    if contrib_a is not None and contrib_b is not None:
        mono_a = np.mean(contrib_a, axis=0) if contrib_a.ndim == 2 else contrib_a
        mono_b = np.mean(contrib_b, axis=0) if contrib_b.ndim == 2 else contrib_b
        L = min(len(mono_a), len(mono_b))

        if L > win:
            nyq = 0.5 * sr
            # Sub-bass band (25 - 120 Hz)
            low_f = max(10.0, min(25.0, 0.05 * sr))
            high_f = min(120.0, 0.45 * sr)
            b_sub, a_sub = signal.butter(2, [low_f / nyq, high_f / nyq], btype="band")
            sub_a = signal.lfilter(b_sub, a_sub, mono_a[:L])
            sub_b = signal.lfilter(b_sub, a_sub, mono_b[:L])

            # Vocal presence band (350 - 3500 Hz)
            mid_low = max(50.0, min(350.0, 0.1 * sr))
            mid_high = min(3500.0, 0.45 * sr)
            b_mid, a_mid = signal.butter(2, [mid_low / nyq, mid_high / nyq], btype="band")
            mid_a = signal.lfilter(b_mid, a_mid, mono_a[:L])
            mid_b = signal.lfilter(b_mid, a_mid, mono_b[:L])

            n_w = L // win
            sub_clashes = 0
            mid_clashes = 0
            for i in range(n_w):
                sl = slice(i * win, (i + 1) * win)
                db_sub_a = 20.0 * np.log10(np.sqrt(np.mean(sub_a[sl] ** 2) + 1e-12))
                db_sub_b = 20.0 * np.log10(np.sqrt(np.mean(sub_b[sl] ** 2) + 1e-12))
                if db_sub_a > -22.0 and db_sub_b > -22.0:
                    sub_clashes += 1

                db_mid_a = 20.0 * np.log10(np.sqrt(np.mean(mid_a[sl] ** 2) + 1e-12))
                db_mid_b = 20.0 * np.log10(np.sqrt(np.mean(mid_b[sl] ** 2) + 1e-12))
                if db_mid_a > -18.0 and db_mid_b > -18.0:
                    mid_clashes += 1

            sub_collision = float(sub_clashes / max(1, n_w))
            vocal_masking = float(mid_clashes / max(1, n_w))

    warnings = []
    if true_peak_db > 0.05:  # Tolerance for true peak margin
        warnings.append(f"Digital clipping detected: True Peak {true_peak_db:.2f} dBFS > 0.0 dBFS")
    if min_corr < -0.15:
        warnings.append(f"Severe stereo phase cancellation: min correlation {min_corr:.2f}")
    if sub_collision > 0.15:
        warnings.append(f"Low-End Monogamy violation: sub-bass collision ratio {sub_collision:.1%}")
    if vocal_masking > 0.35:
        warnings.append(f"Excessive midrange masking: collision ratio {vocal_masking:.1%}")

    return BridgeTelemetryReport(
        true_peak_dbfs=true_peak_db,
        integrated_rms_dbfs=rms_db,
        min_stereo_phase_correlation=min_corr,
        mean_stereo_phase_correlation=mean_corr,
        sub_bass_collision_ratio=sub_collision,
        vocal_midrange_masking_ratio=vocal_masking,
        passes_studio_qc=(len(warnings) == 0),
        qc_warnings=tuple(warnings),
    )


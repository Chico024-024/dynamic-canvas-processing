"""Shared offline-rendering helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import DEFAULT_CONFIG, RenderConfig
from ..models import AudioBuffer, RenderError


@dataclass(frozen=True)
class RenderResult:
    audio: AudioBuffer
    notes: tuple[str, ...] = ()
    master_gain_db: float = 0.0


def finalize_output(
    samples: np.ndarray,
    sample_rate: int,
    *,
    config: RenderConfig | None = None,
    notes: tuple[str, ...] | list[str] = (),
    label: str = "output",
) -> RenderResult:
    """Validate numbers, apply sample-peak limiting, and build a RenderResult.

    Note: This applies digital sample-peak scaling to enforce ``cfg.peak_ceiling``
    (default 0.999 / -0.01 dBFS) to prevent digital integer clipping. It does
    not perform 4x oversampled ITU-R BS.1770 true-peak reconstruction limiting,
    which is part of the Phase 3 mastering chain.
    """

    cfg = config or DEFAULT_CONFIG.render
    working = np.ascontiguousarray(samples, dtype=np.float64)
    if not np.all(np.isfinite(working)):
        raise RenderError(f"{label} contains NaN or infinite samples")
    if working.ndim != 2:
        raise RenderError(f"{label} must be two-dimensional")

    peak = float(np.max(np.abs(working)))
    master_gain_db = 0.0
    final_notes = list(notes)

    if peak > cfg.peak_ceiling:
        if cfg.prevent_clipping:
            gain = cfg.peak_ceiling / peak
            working = working * gain
            master_gain_db = float(20.0 * np.log10(gain))
            final_notes.append(
                f"Master gain {master_gain_db:.2f} dB was applied to prevent "
                f"clipping (raw peak {peak:.4f})."
            )
        else:
            final_notes.append(
                f"Output peak {peak:.4f} exceeds full scale and clipping is "
                "allowed; the rendered file may clip."
            )

    return RenderResult(
        audio=AudioBuffer(working, sample_rate),
        notes=tuple(final_notes),
        master_gain_db=master_gain_db,
    )


def add_unique_note(notes: list[str], note: str) -> None:
    if note not in notes:
        notes.append(note)

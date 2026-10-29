"""Gain conversion and automation-curve evaluation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .plan import GainPoint


def db_to_gain(
    gain_db: float | np.ndarray, floor_db: float = -120.0
) -> float | np.ndarray:
    """Convert decibels to linear gain, mapping the floor (and below) to zero."""

    values = np.asarray(gain_db, dtype=np.float64)
    clipped = np.maximum(values, floor_db)
    gains = np.power(10.0, clipped / 20.0)
    gains = np.where(values <= floor_db + 1e-9, 0.0, gains)
    if np.ndim(gain_db) == 0:
        return float(gains)
    return gains


def gain_to_db(gain: float, floor_db: float = -120.0) -> float:
    if gain <= 0.0:
        return float(floor_db)
    return float(20.0 * np.log10(gain))


def equal_power_fades(
    frame_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return paired cosine/sine fades whose squared gains sum to one."""

    if frame_count < 1:
        raise ValueError(f"frame_count must be at least 1, got {frame_count}")
    positions = np.linspace(0.0, 1.0, frame_count, endpoint=True)
    fade_out = np.cos(positions * np.pi / 2.0)
    fade_in = np.sin(positions * np.pi / 2.0)
    return fade_out, fade_in


def smoothstep(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def interpolate_gain(
    points: Sequence[GainPoint],
    times_s: np.ndarray,
    *,
    curve: str = "smoothstep",
    floor_db: float = -120.0,
) -> np.ndarray:
    """Evaluate a piecewise gain envelope at the given times.

    Values before the first point and after the last point are held constant.
    """

    if not points:
        raise ValueError("at least one gain point is required")
    times = np.asarray(times_s, dtype=np.float64)
    point_times = np.array([point.time_s for point in points], dtype=np.float64)
    point_gains = np.array([point.gain_db for point in points], dtype=np.float64)

    if np.any(np.diff(point_times) <= 0):
        raise ValueError("gain points must have strictly increasing times")

    result_db = np.full(times.shape, point_gains[-1], dtype=np.float64)
    result_db = np.where(times <= point_times[0], point_gains[0], result_db)

    for index in range(len(points) - 1):
        start_time = point_times[index]
        end_time = point_times[index + 1]
        mask = (times >= start_time) & (times < end_time)
        if not np.any(mask):
            continue
        progress = (times[mask] - start_time) / (end_time - start_time)
        if curve == "smoothstep":
            progress = smoothstep(progress)
        elif curve == "linear":
            progress = np.clip(progress, 0.0, 1.0)
        else:
            raise ValueError(
                f"unsupported curve {curve!r}; expected 'smoothstep' or 'linear'"
            )
        result_db[mask] = (
            point_gains[index]
            + (point_gains[index + 1] - point_gains[index]) * progress
        )

    return np.asarray(db_to_gain(result_db, floor_db), dtype=np.float64)

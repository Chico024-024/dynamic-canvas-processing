"""Objective validation and comparison of rendered audio."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..analysis.basic import amplitude_to_dbfs
from ..config import DEFAULT_CONFIG
from ..models import AudioBuffer


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


@dataclass(frozen=True)
class ValidationReport:
    checks: tuple[CheckResult, ...]
    metrics: Mapping[str, float]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "checks": [check.to_dict() for check in self.checks],
            "metrics": {key: round(value, 6) for key, value in self.metrics.items()},
        }

    def format_text(self) -> str:
        lines = [f"Validation: {'PASS' if self.passed else 'FAIL'}"]
        for check in self.checks:
            status = "PASS" if check.passed else "FAIL"
            lines.append(f"  [{status}] {check.name}: {check.detail}")
        if self.metrics:
            metrics = ", ".join(
                f"{key}={value:.4f}" for key, value in self.metrics.items()
            )
            lines.append(f"  metrics: {metrics}")
        return "\n".join(lines)


def _window_rms_db(mono: np.ndarray, sample_rate: int, window_s: float) -> np.ndarray:
    window = max(1, int(round(window_s * sample_rate)))
    count = mono.shape[0] // window
    if count < 1:
        return np.array([amplitude_to_dbfs(float(np.sqrt(np.mean(np.square(mono)))))])
    trimmed = mono[: count * window].reshape(count, window)
    rms = np.sqrt(np.mean(np.square(trimmed), axis=1))
    return 20.0 * np.log10(np.maximum(rms, 1e-12))


def longest_interior_silence_s(
    buffer: AudioBuffer,
    *,
    threshold_dbfs: float = -70.0,
    window_s: float = 0.05,
) -> float:
    """Longest silent run that does not touch either edge of the file."""

    frame_db = _window_rms_db(buffer.mono(), buffer.sample_rate, window_s)
    silent = frame_db < threshold_dbfs
    frame_count = silent.size
    if frame_count < 1:
        return 0.0
    if bool(np.all(silent)):
        return buffer.duration_s

    longest_frames = 0
    index = 0
    while index < frame_count:
        if not silent[index]:
            index += 1
            continue
        end = index
        while end < frame_count and silent[end]:
            end += 1
        if index > 0 and end < frame_count:
            longest_frames = max(longest_frames, end - index)
        index = end

    window = window_s
    return float(longest_frames * window)


def measure_audio(buffer: AudioBuffer) -> dict[str, float]:
    """Compute objective metrics, ignoring non-finite samples in the values."""

    samples = np.asarray(buffer.samples, dtype=np.float64)
    finite = np.isfinite(samples)
    safe = np.where(finite, samples, 0.0)
    peak = float(np.max(np.abs(safe))) if safe.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(safe)))) if safe.size else 0.0
    crest = 20.0 * np.log10(peak / rms) if peak > 0 and rms > 0 else 0.0
    if safe.shape[1] > 1:
        steps = np.abs(np.diff(safe, axis=1))
        max_step = float(np.max(steps)) if steps.size else 0.0
    else:
        max_step = 0.0
    return {
        "duration_s": buffer.duration_s,
        "peak_dbfs": amplitude_to_dbfs(peak),
        "rms_dbfs": amplitude_to_dbfs(rms),
        "crest_factor_db": float(crest),
        "longest_silence_s": longest_interior_silence_s(buffer),
        "near_full_scale_ratio": float(np.mean(np.abs(safe) >= 0.9999)),
        "max_sample_step": max_step,
        "dc_offset_max": float(np.max(np.abs(np.mean(safe, axis=1)))),
    }


def validate_audio(
    buffer: AudioBuffer,
    *,
    expected_duration_s: float | None = None,
    expected_sample_rate: int | None = None,
    expected_channels: int | None = None,
    max_silence_s: float | None = None,
    silence_threshold_dbfs: float | None = None,
    click_threshold: float | None = None,
) -> ValidationReport:
    """Run structural and numeric checks on a rendered buffer."""

    max_gap = (
        DEFAULT_CONFIG.validation_max_gap_s
        if max_silence_s is None
        else max_silence_s
    )
    silence_db = (
        DEFAULT_CONFIG.validation_silence_dbfs
        if silence_threshold_dbfs is None
        else silence_threshold_dbfs
    )
    click_limit = (
        DEFAULT_CONFIG.validation_click_threshold
        if click_threshold is None
        else click_threshold
    )

    checks: list[CheckResult] = []
    checks.append(
        CheckResult(
            "non_empty",
            buffer.frames > 0,
            f"{buffer.frames} frame(s), {buffer.channels} channel(s)",
        )
    )

    finite = bool(np.all(np.isfinite(buffer.samples)))
    checks.append(
        CheckResult(
            "finite",
            finite,
            "all samples are finite" if finite else "NaN or infinite samples found",
        )
    )

    if expected_duration_s is None:
        checks.append(
            CheckResult(
                "duration",
                True,
                f"{buffer.duration_s:.6f}s (no expected duration supplied)",
            )
        )
    else:
        tolerance = max(0.05, 2.0 / buffer.sample_rate)
        difference = abs(buffer.duration_s - expected_duration_s)
        checks.append(
            CheckResult(
                "duration",
                difference <= tolerance,
                f"{buffer.duration_s:.6f}s vs expected "
                f"{expected_duration_s:.6f}s (tolerance {tolerance:.3f}s)",
            )
        )

    if expected_sample_rate is None:
        checks.append(
            CheckResult(
                "sample_rate", True, f"{buffer.sample_rate} Hz (not constrained)"
            )
        )
    else:
        checks.append(
            CheckResult(
                "sample_rate",
                buffer.sample_rate == expected_sample_rate,
                f"{buffer.sample_rate} Hz vs expected {expected_sample_rate} Hz",
            )
        )

    if expected_channels is None:
        checks.append(
            CheckResult("channels", True, f"{buffer.channels} channel(s) (not constrained)")
        )
    else:
        checks.append(
            CheckResult(
                "channels",
                buffer.channels == expected_channels,
                f"{buffer.channels} vs expected {expected_channels} channel(s)",
            )
        )

    peak = buffer.peak if finite else float("inf")
    peak_ok = peak <= 1.0 + 1e-6
    checks.append(
        CheckResult(
            "peak_level",
            peak_ok,
            f"peak {amplitude_to_dbfs(peak):.3f} dBFS"
            if np.isfinite(peak)
            else "peak unavailable",
        )
    )

    safe = np.where(np.isfinite(buffer.samples), buffer.samples, 0.0)
    near_full_ratio = float(np.mean(np.abs(safe) >= 0.9999))
    clipping_ok = near_full_ratio <= 1e-4
    checks.append(
        CheckResult(
            "clipping",
            clipping_ok,
            f"{near_full_ratio:.6%} of samples at or above -0.0009 dBFS",
        )
    )

    longest_gap = longest_interior_silence_s(
        buffer, threshold_dbfs=silence_db
    )
    silence_ok = longest_gap <= max_gap + 1e-9
    checks.append(
        CheckResult(
            "silence_gaps",
            silence_ok,
            f"longest interior silence {longest_gap:.3f}s "
            f"(limit {max_gap:.3f}s, threshold {silence_db:.0f} dBFS)",
        )
    )

    if buffer.frames > 1:
        steps = np.abs(np.diff(safe, axis=1))
        max_step = float(np.max(steps)) if steps.size else 0.0
        discontinuity_ok = max_step <= click_limit
    else:
        max_step = 0.0
        discontinuity_ok = True
    checks.append(
        CheckResult(
            "discontinuity",
            discontinuity_ok,
            f"max adjacent-sample step {max_step:.4f} "
            f"(heuristic limit {click_limit:.4f})",
        )
    )

    metrics = measure_audio(buffer)
    return ValidationReport(checks=tuple(checks), metrics=metrics)


def compare_audio(
    baseline: AudioBuffer, candidate: AudioBuffer
) -> dict[str, float | str]:
    """Return objective differences. This is not a musical-quality score."""

    base_metrics = measure_audio(baseline)
    candidate_metrics = measure_audio(candidate)
    result: dict[str, float | str] = {
        "note": (
            "Objective metrics only; they do not establish musical quality. "
            "Listening tests are required for perceptual evaluation."
        ),
    }
    for key, base_value in base_metrics.items():
        result[f"baseline_{key}"] = round(base_value, 6)
        result[f"dcp_{key}"] = round(candidate_metrics[key], 6)
        result[f"delta_{key}"] = round(candidate_metrics[key] - base_value, 6)
    return result

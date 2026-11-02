"""Optional librosa-backed tempo and beat analysis.

librosa is imported only when this backend is requested. The default Core
analysis path does not depend on it, and the rest of DCP continues to work when
the optional extra is absent.
"""

from __future__ import annotations

import importlib.util
from dataclasses import replace

import numpy as np

from ..config import DEFAULT_CONFIG, AnalysisConfig
from ..models import AnalysisError, AudioBuffer
from .basic import TrackAnalysis, analyze_track
from .key_detection import estimate_downbeats

_INSTALL_HINT = 'install the optional analysis extra with: pip install -e ".[analysis]"'


def librosa_available() -> bool:
    """Return whether librosa can be imported without importing it eagerly."""

    return importlib.util.find_spec("librosa") is not None


def _beat_regularity(beat_times: tuple[float, ...]) -> float:
    if len(beat_times) < 3:
        return 0.0
    intervals = np.diff(np.asarray(beat_times, dtype=np.float64))
    mean = float(np.mean(intervals))
    if mean <= 0.0:
        return 0.0
    return float(np.clip(1.0 - float(np.std(intervals)) / mean, 0.0, 1.0))


def _cross_checked_confidence(
    base: TrackAnalysis,
    librosa_tempo: float,
    regularity: float,
    beat_count: int,
) -> tuple[float, bool]:
    """Blend beat regularity with agreement against the lightweight estimate.

    The result is a conservative engineering confidence, not a calibrated
    probability. If the two estimators disagree beyond an octave relationship,
    confidence is forced below the planning threshold so rhythmic planning
    falls back instead of trusting a contradictory result.
    """

    if beat_count < 4:
        return 0.0, base.tempo_ambiguous

    if base.tempo_bpm is None or base.tempo_confidence <= 0.0:
        confidence = 0.55 if regularity >= 0.75 and beat_count >= 8 else 0.29
        return confidence, False

    ratio = librosa_tempo / base.tempo_bpm
    direct = abs(ratio - 1.0)
    octave = min(abs(ratio - 2.0), abs(ratio - 0.5))

    if direct <= 0.03:
        return max(base.tempo_confidence, 0.80), base.tempo_ambiguous
    if octave <= 0.03:
        return max(base.tempo_confidence, 0.65), True

    confidence = min(0.29, max(0.0, base.tempo_confidence * 0.4))
    return confidence, base.tempo_ambiguous


def analyze_track_librosa(
    buffer: AudioBuffer, config: AnalysisConfig | None = None
) -> TrackAnalysis:
    """Analyze a track with librosa beat tracking, falling back safely.

    The returned object always contains basic measurements. When beat tracking
    succeeds, it also carries estimated beat positions and a librosa tempo
    estimate. The detector does not identify downbeats or musical phrases.
    """

    cfg = config or DEFAULT_CONFIG.analysis
    base = analyze_track(buffer, cfg)
    if buffer.duration_s < 1.0 or base.silence_ratio >= 0.99:
        return base

    try:
        import librosa
    except ImportError as exc:
        raise AnalysisError(
            f"librosa analysis requested but librosa is not installed; {_INSTALL_HINT}"
        ) from exc

    try:
        mono = np.ascontiguousarray(buffer.mono(), dtype=np.float32)
        tempo_raw, beats_raw = librosa.beat.beat_track(
            y=mono,
            sr=buffer.sample_rate,
            units="time",
            trim=False,
        )
        tempo = float(np.atleast_1d(tempo_raw)[0])
    except Exception as exc:
        raise AnalysisError(f"librosa beat tracking failed: {exc}") from exc

    if not np.isfinite(tempo) or tempo <= 0.0:
        return replace(
            base,
            notes=base.notes
            + ("librosa returned no usable tempo; basic analysis was retained.",),
        )

    raw_beats = np.atleast_1d(beats_raw).astype(np.float64)
    beats = tuple(
        sorted(
            {
                float(t)
                for t in raw_beats
                if np.isfinite(t) and 0.0 <= float(t) <= buffer.duration_s
            }
        )
    )
    if len(beats) < 4:
        return replace(
            base,
            notes=base.notes
            + ("librosa returned too few beats; basic analysis was retained.",),
        )

    intervals = np.diff(np.asarray(beats, dtype=np.float64))
    if float(np.mean(intervals)) <= 0.0:
        return replace(
            base,
            notes=base.notes
            + ("librosa returned an unusable beat interval; basic analysis was retained.",),
        )

    regularity = _beat_regularity(beats)
    confidence, ambiguous = _cross_checked_confidence(
        base, tempo, regularity, len(beats)
    )

    notes = [
        note
        for note in base.notes
        if not note.startswith("Tempo estimate")
        and not note.startswith("Tempo is")
        and not note.startswith("Possible half/double-time")
    ]
    notes.append(
        f"Tempo and {len(beats)} beat positions estimated by librosa "
        f"{librosa.__version__}; beat phase is approximate, downbeats are not "
        "detected."
    )
    if confidence < cfg.tempo_confidence_threshold:
        notes.append(
            f"Cross-checked beat confidence {confidence:.2f} is below the "
            f"reliability threshold ({cfg.tempo_confidence_threshold:.2f}); "
            "rhythm-aware planning will not use it."
        )
    if ambiguous:
        notes.append(
            "Possible half/double-time ambiguity between estimators; verify "
            "before relying on beat alignment."
        )
    if regularity < 0.5:
        notes.append(
            "Detected beat intervals are irregular; beat alignment may be "
            "unstable."
        )

    # Downbeat estimation from detected beats
    downbeat_times: tuple[float, ...] = ()
    if len(beats) >= 4:
        try:
            mono_f64 = np.ascontiguousarray(buffer.mono(), dtype=np.float64)
            downbeat_times = estimate_downbeats(
                mono_f64, buffer.sample_rate, beats
            )
            if downbeat_times:
                notes.append(
                    f"{len(downbeat_times)} downbeats estimated from detected "
                    "beats using spectral energy accents; assumes 4/4 meter."
                )
        except Exception:
            pass  # downbeat estimation is informational

    return TrackAnalysis(
        duration_s=base.duration_s,
        sample_rate=base.sample_rate,
        channels=base.channels,
        peak_dbfs=base.peak_dbfs,
        rms_dbfs=base.rms_dbfs,
        silence_ratio=base.silence_ratio,
        tempo_bpm=tempo,
        tempo_confidence=confidence,
        onset_count=base.onset_count,
        tempo_ambiguous=ambiguous,
        notes=tuple(notes),
        beat_times_s=beats,
        analyzer="librosa",
        key=base.key,
        key_confidence=base.key_confidence,
        camelot_code=base.camelot_code,
        downbeat_times_s=downbeat_times,
    )


def analyze_track_with_backend(
    buffer: AudioBuffer,
    config: AnalysisConfig | None = None,
    *,
    backend: str = "auto",
) -> TrackAnalysis:
    """Analyze with ``basic``, ``librosa``, or ``auto`` selection.

    ``auto`` uses librosa when the optional package is installed and falls back
    to basic analysis if librosa fails. Explicitly requesting librosa without the
    package raises a clear AnalysisError rather than silently changing behavior.
    """

    normalized = backend.strip().lower()
    if normalized == "auto":
        if not librosa_available():
            return analyze_track(buffer, config)
        try:
            return analyze_track_librosa(buffer, config)
        except AnalysisError as exc:
            base = analyze_track(buffer, config)
            return replace(
                base,
                notes=base.notes
                + (
                    "Optional librosa analysis failed; basic analysis was used "
                    f"instead ({exc}).",
                ),
            )
    if normalized == "basic":
        return analyze_track(buffer, config)
    if normalized == "librosa":
        return analyze_track_librosa(buffer, config)
    raise AnalysisError(
        f"unknown analysis backend {backend!r}; expected auto, basic, or librosa"
    )


__all__ = [
    "analyze_track_librosa",
    "analyze_track_with_backend",
    "librosa_available",
]

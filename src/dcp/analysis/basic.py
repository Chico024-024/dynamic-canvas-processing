"""Basic track analysis with a lightweight, dependency-light tempo estimator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, stft

from ..config import DEFAULT_CONFIG, AnalysisConfig
from ..models import AnalysisError, AudioBuffer

_DB_FLOOR = -120.0


def amplitude_to_dbfs(value: float, floor: float = _DB_FLOOR) -> float:
    if value <= 0.0:
        return float(floor)
    return float(20.0 * np.log10(value))


def _amplitude_array_to_dbfs(values: np.ndarray, floor: float = _DB_FLOOR) -> np.ndarray:
    finite = np.maximum(values, 10.0 ** (floor / 20.0))
    return 20.0 * np.log10(finite)


@dataclass(frozen=True)
class TrackAnalysis:
    """Objective, explainable measurements for one track."""

    duration_s: float
    sample_rate: int
    channels: int
    peak_dbfs: float
    rms_dbfs: float
    silence_ratio: float
    tempo_bpm: float | None
    tempo_confidence: float
    onset_count: int
    tempo_ambiguous: bool = False
    notes: tuple[str, ...] = ()
    beat_times_s: tuple[float, ...] = ()
    analyzer: str = "basic"

    def to_dict(self) -> dict[str, object]:
        return {
            "duration_s": round(self.duration_s, 6),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "peak_dbfs": round(self.peak_dbfs, 3),
            "rms_dbfs": round(self.rms_dbfs, 3),
            "silence_ratio": round(self.silence_ratio, 5),
            "tempo_bpm": None if self.tempo_bpm is None else round(self.tempo_bpm, 3),
            "tempo_confidence": round(self.tempo_confidence, 4),
            "onset_count": self.onset_count,
            "tempo_ambiguous": self.tempo_ambiguous,
            "analyzer": self.analyzer,
            "beat_times_s": [round(t, 6) for t in self.beat_times_s],
            "notes": list(self.notes),
        }


def _silence_ratio(
    buffer: AudioBuffer, threshold_dbfs: float, window_s: float
) -> float:
    mono = buffer.mono()
    window = max(1, int(round(window_s * buffer.sample_rate)))
    frame_count = mono.shape[0] // window
    if frame_count < 1:
        return 1.0 if amplitude_to_dbfs(buffer.rms) < threshold_dbfs else 0.0
    trimmed = mono[: frame_count * window].reshape(frame_count, window)
    frame_rms = np.sqrt(np.mean(np.square(trimmed), axis=1))
    frame_db = _amplitude_array_to_dbfs(frame_rms)
    return float(np.mean(frame_db < threshold_dbfs))


def _onset_envelope(
    mono: np.ndarray, sample_rate: int, config: AnalysisConfig
) -> tuple[np.ndarray, int]:
    window = min(config.stft_window, 1 << int(np.floor(np.log2(mono.shape[0]))))
    window = max(64, window)
    hop = max(16, min(config.stft_hop, window // 2))
    _, _, spectrum = stft(
        mono,
        fs=sample_rate,
        window="hann",
        nperseg=window,
        noverlap=window - hop,
        nfft=window,
        boundary=None,
        padded=False,
    )
    magnitude = np.abs(spectrum)
    log_magnitude = np.log1p(20.0 * magnitude)
    difference = np.diff(log_magnitude, axis=1)
    flux = np.sum(np.maximum(difference, 0.0), axis=0)
    if flux.size == 0:
        return flux, hop

    smooth_length = max(1, int(config.tempo_smoothing_frames))
    if smooth_length > 1 and flux.size >= smooth_length:
        kernel = np.hanning(smooth_length + 2)[1:-1]
        kernel = kernel / max(float(np.sum(kernel)), 1e-12)
        flux = np.convolve(flux, kernel, mode="same")
    return flux, hop


def _estimate_tempo(
    mono: np.ndarray, sample_rate: int, config: AnalysisConfig
) -> tuple[float | None, float, int, bool]:
    min_bpm = max(1.0, float(config.tempo_min_bpm))
    max_bpm = max(min_bpm + 1.0, float(config.tempo_max_bpm))
    flux, hop = _onset_envelope(mono, sample_rate, config)
    if flux.size < 16:
        return None, 0.0, 0, False

    normalized = flux - float(np.mean(flux))
    energy = float(np.sqrt(np.mean(np.square(normalized))))
    if energy < 1e-9:
        return None, 0.0, 0, False

    frame_count = normalized.size
    fft_size = 1 << (2 * frame_count - 1).bit_length()
    spectrum = np.fft.rfft(normalized, n=fft_size)
    autocorrelation = np.fft.irfft(spectrum * np.conj(spectrum), n=fft_size)
    autocorrelation = autocorrelation[:frame_count]
    if autocorrelation[0] <= 0:
        return None, 0.0, 0, False
    autocorrelation = autocorrelation / autocorrelation[0]

    frame_hz = sample_rate / hop
    lag_min = int(np.ceil(60.0 * frame_hz / max_bpm))
    lag_max = int(np.floor(60.0 * frame_hz / min_bpm))
    lag_min = max(2, lag_min)
    lag_max = min(frame_count - 2, lag_max)
    if lag_max <= lag_min:
        return None, 0.0, 0, False

    search = autocorrelation[lag_min : lag_max + 1]
    best_offset = int(np.argmax(search))
    best_lag = lag_min + best_offset
    confidence = float(np.clip(autocorrelation[best_lag], 0.0, 1.0))

    ambiguous = False
    for relative_lag in (best_lag / 2.0, best_lag * 2.0):
        candidate = int(round(relative_lag))
        if lag_min <= candidate <= lag_max:
            if autocorrelation[candidate] >= 0.75 * autocorrelation[best_lag]:
                ambiguous = True

    refined_lag = float(best_lag)
    if 0 < best_lag < frame_count - 1:
        left = float(autocorrelation[best_lag - 1])
        center = float(autocorrelation[best_lag])
        right = float(autocorrelation[best_lag + 1])
        denominator = left - 2.0 * center + right
        if abs(denominator) > 1e-12:
            offset = 0.5 * (left - right) / denominator
            if abs(offset) < 1.0:
                refined_lag += offset

    tempo = 60.0 * frame_hz / refined_lag
    if not min_bpm - 2.0 <= tempo <= max_bpm + 2.0:
        return None, confidence, 0, ambiguous

    peak_height = float(np.mean(flux) + np.std(flux))
    peaks, _ = find_peaks(
        flux,
        height=peak_height,
        distance=max(1, int(round(0.08 * frame_hz))),
    )
    return float(tempo), confidence, int(len(peaks)), ambiguous


def analyze_track(
    buffer: AudioBuffer, config: AnalysisConfig | None = None
) -> TrackAnalysis:
    """Analyze a track and return objective measurements.

    Tempo estimation is an onset-envelope autocorrelation estimate on the mono
    mix. It is deterministic but approximate; consumers must check
    ``tempo_confidence`` before relying on it.
    """

    cfg = config or DEFAULT_CONFIG.analysis
    if not np.all(np.isfinite(buffer.samples)):
        raise AnalysisError("cannot analyze audio containing NaN or infinite samples")

    peak = amplitude_to_dbfs(buffer.peak)
    rms = amplitude_to_dbfs(buffer.rms)
    silence_ratio = _silence_ratio(buffer, cfg.silence_dbfs, cfg.silence_window_s)

    tempo: float | None = None
    confidence = 0.0
    onset_count = 0
    ambiguous = False
    notes: list[str] = []

    if buffer.duration_s >= 1.0:
        mono = buffer.mono()
        try:
            tempo, confidence, onset_count, ambiguous = _estimate_tempo(
                mono, buffer.sample_rate, cfg
            )
        except Exception as exc:  # analysis must never prevent the rest of DCP
            raise AnalysisError(f"tempo estimation failed: {exc}") from exc

    if tempo is None:
        notes.append(
            "Tempo estimate unavailable (insufficient rhythmic evidence or "
            "confidence)."
        )
    elif confidence < cfg.tempo_confidence_threshold:
        notes.append(
            f"Tempo estimate {tempo:.1f} BPM is below the reliability threshold "
            f"({confidence:.2f} < {cfg.tempo_confidence_threshold:.2f}); "
            "rhythm-aware planning will not use it."
        )
    else:
        notes.append(
            "Tempo is an onset-envelope autocorrelation estimate, not a full "
            "beat tracker."
        )
        if ambiguous:
            notes.append(
                "Possible half/double-time ambiguity in the tempo estimate; "
                "verify before relying on beat alignment."
            )
    if silence_ratio > 0.5:
        notes.append(f"More than half of the track is near-silent ({silence_ratio:.0%}).")

    return TrackAnalysis(
        duration_s=buffer.duration_s,
        sample_rate=buffer.sample_rate,
        channels=buffer.channels,
        peak_dbfs=peak,
        rms_dbfs=rms,
        silence_ratio=silence_ratio,
        tempo_bpm=tempo,
        tempo_confidence=confidence,
        onset_count=onset_count,
        tempo_ambiguous=ambiguous,
        notes=tuple(notes),
    )

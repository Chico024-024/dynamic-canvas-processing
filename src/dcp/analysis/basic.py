"""Basic track analysis with a lightweight, dependency-light tempo estimator."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util

import numpy as np
from scipy import signal
from scipy.signal import find_peaks, stft

from .key_detection import detect_key, estimate_downbeats

from ..config import DEFAULT_CONFIG, AnalysisConfig
from ..models import AnalysisError, AudioBuffer

_PYLOUDNORM_AVAILABLE = importlib.util.find_spec("pyloudnorm") is not None
_LIBROSA_AVAILABLE = importlib.util.find_spec("librosa") is not None

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
    key: str | None = None
    key_confidence: float = 0.0
    camelot_code: str | None = None
    downbeat_times_s: tuple[float, ...] = ()

    # Loudness descriptors (EBU R128)
    loudness_lufs: float | None = None
    loudness_lra: float | None = None
    loudness_true_peak: float | None = None

    # Timbre descriptors
    spectral_centroid_mean: float | None = None
    spectral_centroid_std: float | None = None
    spectral_bandwidth_mean: float | None = None
    spectral_flatness_mean: float | None = None
    spectral_rolloff_mean: float | None = None
    mfcc_mean: tuple[float, ...] = ()
    mfcc_std: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, object]:
        d = {
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
            "key": self.key,
            "key_confidence": round(self.key_confidence, 4),
            "camelot_code": self.camelot_code,
            "downbeat_times_s": [round(t, 6) for t in self.downbeat_times_s],
            "notes": list(self.notes),
        }
        # Loudness descriptors
        if self.loudness_lufs is not None:
            d["loudness_lufs"] = round(self.loudness_lufs, 2)
        if self.loudness_lra is not None:
            d["loudness_lra"] = round(self.loudness_lra, 2)
        if self.loudness_true_peak is not None:
            d["loudness_true_peak"] = round(self.loudness_true_peak, 2)

        # Timbre descriptors
        if self.spectral_centroid_mean is not None:
            d["spectral_centroid_mean"] = round(self.spectral_centroid_mean, 2)
        if self.spectral_centroid_std is not None:
            d["spectral_centroid_std"] = round(self.spectral_centroid_std, 2)
        if self.spectral_bandwidth_mean is not None:
            d["spectral_bandwidth_mean"] = round(self.spectral_bandwidth_mean, 2)
        if self.spectral_flatness_mean is not None:
            d["spectral_flatness_mean"] = round(self.spectral_flatness_mean, 4)
        if self.spectral_rolloff_mean is not None:
            d["spectral_rolloff_mean"] = round(self.spectral_rolloff_mean, 2)
        if self.mfcc_mean:
            d["mfcc_mean"] = [round(x, 4) for x in self.mfcc_mean]
        if self.mfcc_std:
            d["mfcc_std"] = [round(x, 4) for x in self.mfcc_std]

        return d


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


def _track_beats_and_downbeats(
    flux: np.ndarray,
    frame_hz: float,
    tempo: float,
    lag: float,
) -> tuple[tuple[float, ...], tuple[float, ...], float]:
    """Track beats and 4/4 downbeats via circular vector-sum phase locking on onset flux.

    Returns:
        (beat_times_s, downbeat_times_s, alignment_score)
    """
    n_frames = len(flux)
    if n_frames < 4 or lag <= 1.0:
        return (), (), 0.0

    # 1. Circular vector-sum phase lock at the tempo lag
    indices = np.arange(n_frames, dtype=np.float64)
    angles = 2.0 * np.pi * indices / lag
    weights = np.maximum(0.0, flux - np.mean(flux))
    w_sum = float(np.sum(weights))
    if w_sum <= 1e-9:
        weights = np.maximum(0.0, flux)
        w_sum = float(np.sum(weights))

    if w_sum <= 1e-9:
        return (), (), 0.0

    x = float(np.sum(weights * np.cos(angles)))
    y = float(np.sum(weights * np.sin(angles)))
    phi = float(np.arctan2(y, x)) % (2.0 * np.pi)

    # Offset in frames
    phi_frames = (phi / (2.0 * np.pi)) * lag
    first_beat = phi_frames
    while first_beat >= lag:
        first_beat -= lag
    while first_beat < 0:
        first_beat += lag

    beat_frames = np.arange(first_beat, n_frames, lag)
    if len(beat_frames) == 0:
        return (), (), 0.0

    beat_times = tuple(float(f / frame_hz) for f in beat_frames)

    # 2. Downbeat tracking (4-beat bar phase lock)
    bar_energies = np.zeros(4, dtype=np.float64)
    for phase_idx in range(4):
        p_frames = beat_frames[phase_idx::4]
        valid_p = [int(round(bf)) for bf in p_frames if 0 <= int(round(bf)) < n_frames]
        if valid_p:
            bar_energies[phase_idx] = float(np.sum(flux[valid_p]))

    best_bar_phase = int(np.argmax(bar_energies))
    downbeat_times = tuple(beat_times[best_bar_phase::4])

    # 3. Alignment score: flux at beat positions vs overall flux
    valid_beat_indices = [int(round(bf)) for bf in beat_frames if 0 <= int(round(bf)) < n_frames]
    mean_beat_flux = float(np.mean(flux[valid_beat_indices])) if valid_beat_indices else 0.0
    global_mean_flux = float(np.mean(flux))
    score = 0.0
    if global_mean_flux > 1e-9:
        ratio = mean_beat_flux / global_mean_flux
        score = float(np.clip((ratio - 1.0) / 1.5, 0.0, 1.0))

    return beat_times, downbeat_times, score


def _estimate_tempo(
    mono: np.ndarray, sample_rate: int, config: AnalysisConfig
) -> tuple[float | None, float, int, bool, tuple[float, ...], tuple[float, ...]]:
    min_bpm = max(1.0, float(config.tempo_min_bpm))
    max_bpm = max(min_bpm + 1.0, float(config.tempo_max_bpm))
    flux, hop = _onset_envelope(mono, sample_rate, config)
    if flux.size < 16:
        return None, 0.0, 0, False, (), ()

    normalized = flux - float(np.mean(flux))
    energy = float(np.sqrt(np.mean(np.square(normalized))))
    if energy < 1e-9:
        return None, 0.0, 0, False, (), ()

    frame_count = normalized.size
    fft_size = 1 << (2 * frame_count - 1).bit_length()
    spectrum = np.fft.rfft(normalized, n=fft_size)
    autocorrelation = np.fft.irfft(spectrum * np.conj(spectrum), n=fft_size)
    autocorrelation = autocorrelation[:frame_count]
    if autocorrelation[0] <= 0:
        return None, 0.0, 0, False, (), ()
    autocorrelation = autocorrelation / autocorrelation[0]

    frame_hz = sample_rate / hop
    lag_min = int(np.ceil(60.0 * frame_hz / max_bpm))
    lag_max = int(np.floor(60.0 * frame_hz / min_bpm))
    lag_min = max(2, lag_min)
    lag_max = min(frame_count - 2, lag_max)
    if lag_max <= lag_min:
        return None, 0.0, 0, False, (), ()

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
        return None, confidence, 0, ambiguous, (), ()

    peak_height = float(np.mean(flux) + np.std(flux))
    peaks, _ = find_peaks(
        flux,
        height=peak_height,
        distance=max(1, int(round(0.08 * frame_hz))),
    )

    beat_times, downbeats, beat_score = _track_beats_and_downbeats(flux, frame_hz, float(tempo), refined_lag)
    combined_confidence = float(np.clip(0.6 * confidence + 0.4 * beat_score, 0.0, 1.0))
    if combined_confidence < config.tempo_confidence_threshold or beat_score < 0.15 or len(peaks) < 4:
        beat_times = ()
        downbeats = ()

    return float(tempo), combined_confidence, int(len(peaks)), ambiguous, beat_times, downbeats


def _compute_loudness(
    buffer: AudioBuffer,
) -> tuple[float | None, float | None, float | None]:
    """Compute EBU R128 loudness metrics (LUFS, LRA, True Peak).

    Uses pyloudnorm if available; falls back to an ITU-R BS.1770 K-weighting filter proxy.
    """
    if _PYLOUDNORM_AVAILABLE:
        try:
            import pyloudnorm as pyln

            mono = buffer.mono()
            meter = pyln.Meter(buffer.sample_rate)
            integrated_loudness = meter.integrated_loudness(mono)
            lra = meter.loudness_range(mono)
            true_peak = amplitude_to_dbfs(buffer.peak)

            return float(integrated_loudness), float(lra), float(true_peak)
        except Exception:
            pass

    # Pure NumPy / SciPy ITU-R BS.1770 K-weighting filter proxy
    try:
        sr = buffer.sample_rate
        mono = buffer.mono()
        # Stage 1: High-pass filter (~38 Hz 2nd-order Butterworth)
        f_hp = min(38.0, 0.45 * (sr / 2.0))
        sos_hp = signal.butter(2, f_hp, "highpass", fs=sr, output="sos")
        y = signal.sosfilt(sos_hp, mono.astype(np.float64))

        # Stage 2: High-shelf boost (~1500 Hz, +4 dB)
        f_hs = min(1500.0, 0.45 * (sr / 2.0))
        sos_hs = signal.butter(1, f_hs, "highpass", fs=sr, output="sos")
        y_k = y + 0.585 * signal.sosfilt(sos_hs, y)

        mean_sq = float(np.mean(y_k ** 2))
        if mean_sq > 1e-12:
            lufs = -0.691 + 10.0 * np.log10(mean_sq)
        else:
            lufs = -120.0
        peak = amplitude_to_dbfs(buffer.peak)
        return float(lufs), None, float(peak)
    except Exception:
        return None, None, None


def _compute_timbre(
    buffer: AudioBuffer,
) -> tuple[
    float | None, float | None,
    float | None, float | None,
    float | None,
    tuple[float, ...], tuple[float, ...]
]:
    """Compute timbre descriptors.

    Returns:
        (spectral_centroid_mean, spectral_centroid_std,
         spectral_bandwidth_mean, spectral_flatness_mean,
         spectral_rolloff_mean, mfcc_mean, mfcc_std)
    """
    if not _LIBROSA_AVAILABLE:
        return None, None, None, None, None, (), ()

    try:
        import librosa

        mono = buffer.mono()
        y = mono.astype(np.float32)
        sr = buffer.sample_rate

        # Spectral centroid
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_mean = float(np.mean(centroid))
        centroid_std = float(np.std(centroid))

        # Spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        bandwidth_mean = float(np.mean(bandwidth))

        # Spectral flatness
        flatness = librosa.feature.spectral_flatness(y=y)
        flatness_mean = float(np.mean(flatness))

        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_mean = float(np.mean(rolloff))

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = tuple(float(x) for x in np.mean(mfcc, axis=1))
        mfcc_std = tuple(float(x) for x in np.std(mfcc, axis=1))

        return (centroid_mean, centroid_std, bandwidth_mean,
                flatness_mean, rolloff_mean, mfcc_mean, mfcc_std)
    except Exception:
        return None, None, None, None, None, (), ()


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

    # Compute loudness descriptors (EBU R128)
    loudness_lufs, loudness_lra, loudness_true_peak = _compute_loudness(buffer)

    # Compute timbre descriptors
    (spectral_centroid_mean, spectral_centroid_std,
     spectral_bandwidth_mean, spectral_flatness_mean,
     spectral_rolloff_mean, mfcc_mean, mfcc_std) = _compute_timbre(buffer)

    tempo: float | None = None
    confidence = 0.0
    onset_count = 0
    ambiguous = False
    beat_times: tuple[float, ...] = ()
    downbeat_times: tuple[float, ...] = ()
    notes: list[str] = []
    detected_key: str | None = None
    detected_key_confidence = 0.0
    detected_camelot: str | None = None

    if buffer.duration_s >= 1.0:
        mono = buffer.mono()
        try:
            tempo, confidence, onset_count, ambiguous, beat_times, downbeat_times = _estimate_tempo(
                mono, buffer.sample_rate, cfg
            )
        except Exception as exc:  # analysis must never prevent the rest of DCP
            raise AnalysisError(f"tempo estimation failed: {exc}") from exc

        # Key detection — best-effort, never blocks the pipeline
        try:
            key_name, camelot, _mode, key_conf = detect_key(mono, buffer.sample_rate)
            if key_name != "unknown":
                detected_key = key_name
                detected_key_confidence = key_conf
                detected_camelot = camelot
                notes.append(
                    f"Key estimate: {key_name} (Camelot {camelot}, "
                    f"confidence {key_conf:.2f}). Chroma-based heuristic, "
                    "not a trained classifier."
                )
        except Exception:
            pass  # key detection is informational; never fatal

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

    # Add loudness note if available
    if loudness_lufs is not None:
        lra_part = f" (LRA: {loudness_lra:.1f} LU)" if loudness_lra is not None else ""
        tp_part = f", True Peak: {loudness_true_peak:.1f} dBTP" if loudness_true_peak is not None else ""
        notes.append(f"Integrated loudness: {loudness_lufs:.1f} LUFS{lra_part}{tp_part}.")

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
        beat_times_s=beat_times,
        downbeat_times_s=downbeat_times,
        notes=tuple(notes),
        key=detected_key,
        key_confidence=detected_key_confidence,
        camelot_code=detected_camelot,
        # Loudness descriptors
        loudness_lufs=loudness_lufs,
        loudness_lra=loudness_lra,
        loudness_true_peak=loudness_true_peak,
        # Timbre descriptors
        spectral_centroid_mean=spectral_centroid_mean,
        spectral_centroid_std=spectral_centroid_std,
        spectral_bandwidth_mean=spectral_bandwidth_mean,
        spectral_flatness_mean=spectral_flatness_mean,
        spectral_rolloff_mean=spectral_rolloff_mean,
        mfcc_mean=mfcc_mean,
        mfcc_std=mfcc_std,
    )

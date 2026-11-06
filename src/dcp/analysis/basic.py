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


def _comb_phase_and_beats(
    flux: np.ndarray,
    lam: float,
    hop: int,
    sample_rate: int,
    window: int | None = None,
) -> tuple[float, float, tuple[float, ...]]:
    """Compute circular vector-sum comb strength, beat phase, and locally refined beat times."""
    n_frames = flux.size
    if n_frames < 8 or lam <= 1.0:
        return 0.0, 0.0, ()

    w = np.maximum(flux - float(np.mean(flux)), 0.0)
    w_sum = float(np.sum(w))
    if w_sum <= 1e-9:
        return 0.0, 0.0, ()

    phase_x = np.mod(np.arange(n_frames, dtype=np.float64) / lam, 1.0)
    z = np.sum(w * np.exp(2j * np.pi * phase_x))
    phi = float(np.angle(z) / (2.0 * np.pi)) % 1.0
    comb = float(np.abs(z) / w_sum)

    # Generate candidate beat frames k * lam + phi * lam and snap +-2 frames to local flux peak
    k_max = int(np.floor((n_frames - 1 - phi * lam) / lam))
    if k_max < 0:
        return comb, phi, ()

    raw_frames = np.arange(k_max + 1, dtype=np.float64) * lam + phi * lam
    refined_frames: list[int] = []
    for rf in raw_frames:
        center = int(round(float(rf)))
        lo = max(0, center - 2)
        hi = min(n_frames - 1, center + 2)
        if hi >= lo:
            best_f = lo + int(np.argmax(flux[lo : hi + 1]))
            if not refined_frames or best_f > refined_frames[-1]:
                refined_frames.append(best_f)

    # STFT without boundary padding centers frame 0 at window / 2, where window = 4 * hop (default 1024/256)
    win = window if window is not None else 4 * hop
    beat_times = tuple(float((f * hop + (win // 2)) / sample_rate) for f in refined_frames)
    return comb, phi, beat_times


def _estimate_tempo(
    mono: np.ndarray, sample_rate: int, config: AnalysisConfig
) -> tuple[float | None, float, int, bool]:
    min_bpm = max(1.0, float(config.tempo_min_bpm))
    max_bpm = max(min_bpm + 1.0, float(config.tempo_max_bpm))
    flux, hop = _onset_envelope(mono, sample_rate, config)
    if flux.size < 16:
        return None, 0.0, 0, False

    mean_flux = float(np.mean(flux))
    std_flux = float(np.std(flux))
    normalized = flux - mean_flux
    energy = float(np.sqrt(np.mean(np.square(normalized))))
    if energy < 1e-9 or mean_flux <= 1e-9:
        return None, 0.0, 0, False

    # Guard against stationary white noise / unmodulated signals (low coefficient of variation)
    cv = std_flux / max(mean_flux, 1e-9)
    if cv < 0.22:
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
    autocorr_peak = float(np.clip(autocorrelation[best_lag], 0.0, 1.0))

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

    coarse_tempo = 60.0 * frame_hz / refined_lag
    if not min_bpm - 2.0 <= coarse_tempo <= max_bpm + 2.0:
        return None, 0.0, 0, False

    # R1(b): Octave resolution BEFORE phase-locking at L, 2L (half-tempo), and L/2 (double-tempo)
    # Weight comb strength with local autocorrelation support at the candidate lag so subharmonics
    # (which always match a subset of 2x impulses in a circular sum) only win when backed by pulse periodicity.
    candidate_lags: list[float] = [refined_lag]
    if 60.0 * frame_hz / (2.0 * refined_lag) >= min_bpm:
        candidate_lags.append(2.0 * refined_lag)
    if 60.0 * frame_hz / (0.5 * refined_lag) <= max_bpm:
        candidate_lags.append(0.5 * refined_lag)

    cand_results: list[tuple[float, float, float]] = []
    for cand_lam in candidate_lags:
        c_comb, _, _ = _comb_phase_and_beats(flux, cand_lam, hop, sample_rate)
        idx_lam = int(round(cand_lam))
        ac_val = (
            float(np.clip(autocorrelation[idx_lam], 0.0, 1.0))
            if 0 < idx_lam < frame_count
            else 0.0
        )
        # Effective octave score combines circular comb lock with autocorrelation periodicity at lag
        eff_score = 0.55 * c_comb + 0.45 * ac_val
        cand_results.append((eff_score, c_comb, cand_lam))

    cand_results.sort(key=lambda item: item[0], reverse=True)
    winner_eff, winner_comb, winner_lam = cand_results[0]

    ambiguous = False
    if len(cand_results) > 1:
        runner_up_eff = cand_results[1][0]
        if winner_eff > 1e-9 and runner_up_eff >= 0.85 * winner_eff:
            ambiguous = True

    tempo = 60.0 * frame_hz / winner_lam

    peak_height = float(mean_flux + std_flux)
    peaks, _ = find_peaks(
        flux,
        height=peak_height,
        distance=max(1, int(round(0.08 * frame_hz))),
    )

    # R3: Calibrated blend of autocorrelation peak and circular comb vector strength
    confidence = float(np.clip(0.25 * autocorr_peak + 0.75 * winner_comb, 0.0, 1.0))
    if len(peaks) < 4:
        confidence = min(confidence, 0.15)

    return float(tempo), confidence, int(len(peaks)), ambiguous


def _build_beat_and_downbeat_grids(
    mono: np.ndarray,
    sample_rate: int,
    duration_s: float,
    tempo: float | None,
    confidence: float,
    config: AnalysisConfig,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Construct phase-locked beat and 4/4 downbeat grids when rhythmic evidence is sufficient."""
    if (
        tempo is None
        or tempo <= 0.0
        or confidence < config.tempo_confidence_threshold
    ):
        return (), ()

    flux, hop = _onset_envelope(mono, sample_rate, config)
    if flux.size < 16:
        return (), ()

    frame_hz = sample_rate / hop
    lam = 60.0 * frame_hz / tempo
    stft_win = max(64, min(config.stft_window, 1 << int(np.floor(np.log2(mono.shape[0])))))
    comb, _, raw_beats = _comb_phase_and_beats(flux, lam, hop, sample_rate, window=stft_win)
    if comb <= 1e-6 or not raw_beats:
        return (), ()

    # Trim beats to [0, duration_s] and require >= 8 beats
    beat_times = tuple(t for t in raw_beats if 0.0 <= t <= duration_s)
    if len(beat_times) < 8:
        return (), ()

    downbeats = estimate_downbeats(mono, sample_rate, beat_times, beats_per_bar=4)
    downbeat_times = tuple(t for t in downbeats if 0.0 <= t <= duration_s)
    if not downbeat_times:
        return (), ()

    return beat_times, downbeat_times


def _rbj_high_shelf_sos(
    fc: float, gain_db: float, q: float, fs: int
) -> np.ndarray:
    """Design a 2nd-order RBJ audio EQ cookbook high-shelf biquad in SOS format."""
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * min(fc, 0.475 * fs) / fs
    cos_w0 = float(np.cos(w0))
    sin_w0 = float(np.sin(w0))
    alpha = sin_w0 / (2.0 * q)
    two_sqrt_a_alpha = 2.0 * np.sqrt(a) * alpha

    b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + two_sqrt_a_alpha)
    b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0)
    b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - two_sqrt_a_alpha)
    a0 = (a + 1.0) - (a - 1.0) * cos_w0 + two_sqrt_a_alpha
    a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0)
    a2 = (a + 1.0) - (a - 1.0) * cos_w0 - two_sqrt_a_alpha

    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)


def _compute_loudness(
    buffer: AudioBuffer,
) -> tuple[float | None, float | None, float | None]:
    """Compute EBU R128 loudness metrics (LUFS, LRA, True Peak).

    Uses pyloudnorm when installed; otherwise uses a pure NumPy/SciPy
    ITU BS.1770 approximation, calibrated vs pyloudnorm.
    """
    if _PYLOUDNORM_AVAILABLE:
        try:
            import pyloudnorm as pyln

            mono = buffer.mono()
            meter = pyln.Meter(buffer.sample_rate)
            integrated_loudness = meter.integrated_loudness(mono)
            lra = meter.loudness_range(mono)
            true_peak = amplitude_to_dbfs(buffer.peak)

            if np.isfinite(integrated_loudness) and np.isfinite(lra):
                return float(integrated_loudness), float(lra), float(true_peak)
        except Exception:
            pass

    try:
        sr = int(buffer.sample_rate)
        mono = buffer.mono().astype(np.float64, copy=False)
        if mono.size == 0 or not np.all(np.isfinite(mono)):
            return None, None, None

        # Stage 1: 2nd-order Butterworth highpass at 38 Hz
        f_hp = min(38.0, 0.45 * (sr / 2.0))
        sos_hp = signal.butter(2, f_hp, btype="highpass", fs=sr, output="sos")

        # Stage 2: RBJ high-shelf (+4.0 dB, ~1.5 kHz, Q=0.7)
        sos_hs = _rbj_high_shelf_sos(fc=1500.0, gain_db=4.0, q=0.7, fs=sr)
        sos_k = np.vstack([sos_hp, sos_hs])

        y_k = signal.sosfilt(sos_k, mono)

        # 400 ms non-overlapping RMS blocks (fallback to whole buffer if shorter than 400 ms)
        block_samples = max(1, int(round(0.400 * sr)))
        n_blocks = y_k.size // block_samples
        if n_blocks >= 1:
            blocks = y_k[: n_blocks * block_samples].reshape(n_blocks, block_samples)
            block_powers = np.mean(np.square(blocks), axis=1)
        else:
            block_powers = np.array([ float(np.mean(np.square(y_k))) ], dtype=np.float64)

        # Filter out absolute silence blocks (> -70 LUFS)
        abs_thresh_power = 10.0 ** ((-70.0 + 0.691) / 10.0)
        active_powers = block_powers[block_powers > abs_thresh_power]
        if active_powers.size == 0:
            return None, None, None

        # One-pass relative gating: drop blocks > 10 LU below mean active block loudness
        mean_un_lufs = -0.691 + 10.0 * np.log10(float(np.mean(active_powers)))
        rel_gate_lufs = mean_un_lufs - 10.0
        block_lufs = -0.691 + 10.0 * np.log10(active_powers)
        gated_powers = active_powers[block_lufs >= rel_gate_lufs]
        if gated_powers.size == 0:
            gated_powers = active_powers

        integrated_lufs = float(-0.691 + 10.0 * np.log10(float(np.mean(gated_powers))))
        gated_lufs = -0.691 + 10.0 * np.log10(gated_powers)
        lra = (
            float(np.percentile(gated_lufs, 95.0) - np.percentile(gated_lufs, 10.0))
            if gated_lufs.size >= 2
            else 0.0
        )
        true_peak = amplitude_to_dbfs(buffer.peak)
        return integrated_lufs, lra, float(true_peak)
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
            tempo, confidence, onset_count, ambiguous = _estimate_tempo(
                mono, buffer.sample_rate, cfg
            )
            beat_times, downbeat_times = _build_beat_and_downbeat_grids(
                mono,
                buffer.sample_rate,
                buffer.duration_s,
                tempo,
                confidence,
                cfg,
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
        analyzer="basic",
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


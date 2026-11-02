"""Neural beat & downbeat tracker backend using TCN (Temporal Convolutional Network).

This module provides a drop-in replacement for librosa-based beat tracking
using a neural network architecture. It integrates with the existing
analyze_track_with_backend abstraction.

Architecture: TCN with dilated causal convolutions for beat/downbeat probability
estimation, followed by peak picking and tempo estimation.

Training data: GTZAN, Ballroom, Hainsworth datasets + synthetic augmentation.
"""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from typing import Optional, TYPE_CHECKING

import numpy as np

from ..config import DEFAULT_CONFIG, AnalysisConfig
from ..models import AnalysisError, AudioBuffer
from .basic import TrackAnalysis, analyze_track
from .key_detection import estimate_downbeats
from .model_registry import get_model_path, MODEL_REGISTRY

if TYPE_CHECKING:
    import torch
    import torchaudio


_INSTALL_HINT = (
    "install PyTorch and torchaudio: pip install torch torchaudio\n"
    "Model weights will be downloaded on first use to ~/.dcp/models/"
)

try:
    import torch
    _TORCH_AVAILABLE = True
except (ImportError, OSError):
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False

try:
    import torchaudio
    _TORCHAUDIO_AVAILABLE = True
except (ImportError, OSError):
    torchaudio = None  # type: ignore
    _TORCHAUDIO_AVAILABLE = False


def neural_beat_available() -> bool:
    """Return whether the neural beat tracker dependencies are available."""
    return _TORCH_AVAILABLE and _TORCHAUDIO_AVAILABLE


# Only define the model class if torch is available
if _TORCH_AVAILABLE and torch is not None:
    class BeatTrackerNet(torch.nn.Module):
        """TCN-based beat and downbeat tracker.

        Architecture:
        - Input: Log-mel spectrogram (64 bands, 100 fps)
        - Frontend: 2 conv layers for feature extraction
        - Backbone: 8-layer TCN with dilated convolutions (dilation 1,2,4,8,16,32,64,128)
        - Heads: Beat probability, Downbeat probability, Tempo regression
        """

        def __init__(
            self,
            n_mels: int = 64,
            n_channels: int = 64,
            n_layers: int = 8,
            kernel_size: int = 3,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.n_mels = n_mels

            # Frontend feature extraction
            self.frontend = torch.nn.Sequential(
                torch.nn.Conv1d(n_mels, n_channels, kernel_size=7, padding=3),
                torch.nn.BatchNorm1d(n_channels),
                torch.nn.ReLU(),
                torch.nn.Conv1d(n_channels, n_channels, kernel_size=3, padding=1),
                torch.nn.BatchNorm1d(n_channels),
                torch.nn.ReLU(),
            )

            # TCN backbone with dilated convolutions
            self.tcn_layers = torch.nn.ModuleList()
            for i in range(n_layers):
                dilation = 2 ** i
                padding = (kernel_size - 1) * dilation
                self.tcn_layers.append(
                    torch.nn.Sequential(
                        torch.nn.Conv1d(
                            n_channels,
                            n_channels,
                            kernel_size=kernel_size,
                            padding=padding,
                            dilation=dilation,
                        ),
                        torch.nn.BatchNorm1d(n_channels),
                        torch.nn.ReLU(),
                        torch.nn.Dropout(dropout),
                        torch.nn.Conv1d(n_channels, n_channels, kernel_size=1),
                        torch.nn.BatchNorm1d(n_channels),
                    )
                )

            # Residual connections
            self.residual_projs = torch.nn.ModuleList([
                torch.nn.Conv1d(n_channels, n_channels, kernel_size=1)
                for _ in range(n_layers)
            ])

            # Output heads
            self.beat_head = torch.nn.Conv1d(n_channels, 1, kernel_size=1)
            self.downbeat_head = torch.nn.Conv1d(n_channels, 1, kernel_size=1)
            self.tempo_head = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool1d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(n_channels, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(128, 1),
            )

        def forward(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            """Forward pass.

            Args:
                x: Log-mel spectrogram [B, n_mels, T]

            Returns:
                beat_probs: Beat probability per frame [B, 1, T]
                downbeat_probs: Downbeat probability per frame [B, 1, T]
                tempo: Tempo estimate in BPM [B, 1]
            """
            x = self.frontend(x)

            for i, layer in enumerate(self.tcn_layers):
                residual = x
                x = layer(x)
                # Crop residual to match output length (due to padding)
                if x.shape[-1] != residual.shape[-1]:
                    diff = residual.shape[-1] - x.shape[-1]
                    residual = residual[..., diff // 2 : -(diff - diff // 2)]
                x = x + self.residual_projs[i](residual)

            beat_probs = torch.sigmoid(self.beat_head(x))
            downbeat_probs = torch.sigmoid(self.downbeat_head(x))
            tempo = torch.relu(self.tempo_head(x)) + 60.0  # Minimum 60 BPM

            return beat_probs, downbeat_probs, tempo

else:
    # Placeholder when torch is not available
    class BeatTrackerNet:
        def __init__(self, *args, **kwargs):
            raise AnalysisError(f"Neural beat tracker requires PyTorch; {_INSTALL_HINT}")


def _load_mel_spectrogram(
    buffer: AudioBuffer,
    n_mels: int = 64,
    n_fft: int = 1024,
    hop_length: int = 256,
    fmin: float = 30.0,
    fmax: float = 8000.0,
) -> "torch.Tensor":
    """Compute log-mel spectrogram from audio buffer.

    Returns tensor of shape [1, n_mels, T] for model input.
    """
    if not _TORCHAUDIO_AVAILABLE:
        raise AnalysisError("torchaudio required for mel spectrogram computation")

    import torchaudio.transforms as T

    mono = buffer.mono()
    waveform = torch.from_numpy(mono).unsqueeze(0).float()  # [1, samples]

    mel_transform = T.MelSpectrogram(
        sample_rate=buffer.sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=fmin,
        f_max=fmax,
        power=2.0,
        normalized=False,
    )
    mel_spec = mel_transform(waveform)  # [1, n_mels, T]
    log_mel = torch.log(torch.clamp(mel_spec, min=1e-10))

    return log_mel


def _peaks_from_probs(
    probs: np.ndarray,
    threshold: float = 0.5,
    min_distance: int = 2,
) -> np.ndarray:
    """Extract peak positions from probability sequence."""
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(probs, height=threshold, distance=min_distance)
    return peaks


def _tempo_from_beats(beat_times: np.ndarray, sr: int, hop_length: int) -> tuple[float, float]:
    """Estimate tempo from beat times with confidence."""
    if len(beat_times) < 2:
        return 120.0, 0.0

    intervals = np.diff(beat_times)
    mean_interval = float(np.mean(intervals))
    if mean_interval <= 0:
        return 120.0, 0.0

    tempo = 60.0 / mean_interval
    # Confidence based on interval regularity
    cv = float(np.std(intervals) / mean_interval) if mean_interval > 0 else 1.0
    confidence = float(np.clip(1.0 - cv, 0.0, 1.0))

    return tempo, confidence


def _run_neural_inference(
    buffer: AudioBuffer,
    model: "BeatTrackerNet",
    device: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run neural inference on audio buffer.

    Returns:
        beat_times: Beat times in seconds
        downbeat_times: Downbeat times in seconds
        tempo: Estimated tempo in BPM
    """
    if not _TORCH_AVAILABLE:
        raise AnalysisError(f"Neural beat tracker requires PyTorch; {_INSTALL_HINT}")

    import torch

    model.eval()
    with torch.no_grad():
        mel = _load_mel_spectrogram(buffer).to(device)  # [1, 64, T]
        beat_probs, downbeat_probs, tempo_pred = model(mel)

        beat_probs = beat_probs.squeeze().cpu().numpy()  # [T]
        downbeat_probs = downbeat_probs.squeeze().cpu().numpy()
        tempo = float(tempo_pred.item())

    # Convert frame indices to time
    hop_length = 256
    frame_times = np.arange(len(beat_probs)) * hop_length / buffer.sample_rate

    # Extract peaks
    beat_frames = _peaks_from_probs(beat_probs, threshold=0.4, min_distance=max(1, int(0.3 * buffer.sample_rate / hop_length)))
    downbeat_frames = _peaks_from_probs(downbeat_probs, threshold=0.35, min_distance=max(1, int(1.0 * buffer.sample_rate / hop_length)))

    beat_times = frame_times[beat_frames]
    downbeat_times = frame_times[downbeat_frames]

    # Filter to valid range
    beat_times = beat_times[(beat_times >= 0) & (beat_times <= buffer.duration_s)]
    downbeat_times = downbeat_times[(downbeat_times >= 0) & (downbeat_times <= buffer.duration_s)]

    return beat_times, downbeat_times, tempo


def analyze_track_neural(
    buffer: AudioBuffer,
    config: AnalysisConfig | None = None,
) -> TrackAnalysis:
    """Analyze a track using the neural beat tracker.

    This function follows the same interface as analyze_track_librosa
    and returns a TrackAnalysis with neural beat/downbeat estimates.
    """
    if not neural_beat_available():
        raise AnalysisError(
            f"Neural beat tracker requires PyTorch and torchaudio; {_INSTALL_HINT}"
        )

    cfg = config or DEFAULT_CONFIG.analysis

    # Get basic measurements first
    base = analyze_track(buffer, cfg)

    if buffer.duration_s < 1.0 or base.silence_ratio >= 0.99:
        return replace(base, analyzer="neural")

    try:
        # Load model
        model_path = get_model_path("beat_tracker_v1")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = BeatTrackerNet()
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)

        # Run inference
        beat_times, downbeat_times, tempo = _run_neural_inference(buffer, model, device)

        if len(beat_times) < 4:
            return replace(
                base,
                analyzer="neural",
                notes=base.notes + ("Neural tracker returned too few beats; basic analysis retained.",),
            )

        # Compute confidence from beat regularity
        intervals = np.diff(beat_times)
        mean_interval = float(np.mean(intervals))
        if mean_interval <= 0:
            return replace(
                base,
                analyzer="neural",
                notes=base.notes + ("Neural tracker returned invalid intervals; basic analysis retained.",),
            )

        tempo_from_beats = 60.0 / mean_interval
        cv = float(np.std(intervals) / mean_interval) if mean_interval > 0 else 1.0
        confidence = float(np.clip(1.0 - cv, 0.0, 1.0))

        # Cross-check with basic tempo estimate
        if base.tempo_bpm is not None and base.tempo_confidence > 0:
            ratio = tempo_from_beats / base.tempo_bpm
            if abs(ratio - 1.0) <= 0.05:
                confidence = max(confidence, 0.85)
            elif min(abs(ratio - 2.0), abs(ratio - 0.5)) <= 0.05:
                confidence = max(confidence, 0.7)

        # Detect ambiguous tempo
        ambiguous = False
        if base.tempo_bpm is not None:
            ratio = tempo_from_beats / base.tempo_bpm
            if min(abs(ratio - 2.0), abs(ratio - 0.5)) <= 0.03:
                ambiguous = True

        # Build notes
        notes = [
            note
            for note in base.notes
            if not note.startswith("Tempo estimate")
            and not note.startswith("Tempo is")
            and not note.startswith("Possible half/double-time")
        ]
        notes.append(
            f"Tempo and {len(beat_times)} beat positions estimated by neural TCN tracker; "
            "beat phase is approximate."
        )
        if confidence < cfg.tempo_confidence_threshold:
            notes.append(
                f"Neural beat confidence {confidence:.2f} is below the "
                f"reliability threshold ({cfg.tempo_confidence_threshold:.2f}); "
                "rhythm-aware planning will not use it."
            )
        if ambiguous:
            notes.append(
                "Possible half/double-time ambiguity with basic estimator; verify "
                "before relying on beat alignment."
            )

        if len(intervals) > 0 and np.std(intervals) / mean_interval > 0.15:
            notes.append(
                "Detected beat intervals are irregular; beat alignment may be "
                "unstable."
            )

        return TrackAnalysis(
            duration_s=base.duration_s,
            sample_rate=base.sample_rate,
            channels=base.channels,
            peak_dbfs=base.peak_dbfs,
            rms_dbfs=base.rms_dbfs,
            silence_ratio=base.silence_ratio,
            tempo_bpm=tempo_from_beats,
            tempo_confidence=confidence,
            onset_count=base.onset_count,
            tempo_ambiguous=ambiguous,
            notes=tuple(notes),
            beat_times_s=tuple(sorted(float(t) for t in beat_times)),
            analyzer="neural",
            key=base.key,
            key_confidence=base.key_confidence,
            camelot_code=base.camelot_code,
            downbeat_times_s=tuple(sorted(float(t) for t in downbeat_times)),
        )

    except Exception as exc:
        raise AnalysisError(f"Neural beat tracking failed: {exc}") from exc


# Re-export librosa_available for backward compatibility
from .librosa_backend import librosa_available


def analyze_track_with_backend(
    buffer: AudioBuffer,
    config: AnalysisConfig | None = None,
    *,
    backend: str = "auto",
) -> TrackAnalysis:
    """Analyze with 'basic', 'librosa', 'neural', or 'auto' selection.

    'auto' priority: neural > librosa > basic
    """
    from dataclasses import replace

    normalized = backend.strip().lower()
    if normalized == "auto":
        # Try neural first, then librosa, then basic
        if neural_beat_available():
            try:
                return analyze_track_neural(buffer, config)
            except AnalysisError as exc:
                # Fall through to librosa
                pass
        if librosa_available():
            try:
                from .librosa_backend import analyze_track_librosa
                return analyze_track_librosa(buffer, config)
            except AnalysisError as exc:
                # Fall through to basic with note
                base = analyze_track(buffer, config)
                return replace(
                    base,
                    notes=base.notes
                    + (
                        f"Optional librosa analysis failed; basic analysis was used instead ({exc}).",
                    ),
                )
        return analyze_track(buffer, config)

    if normalized == "basic":
        return analyze_track(buffer, config)
    if normalized == "librosa":
        from .librosa_backend import analyze_track_librosa
        return analyze_track_librosa(buffer, config)
    if normalized == "neural":
        return analyze_track_neural(buffer, config)

    raise AnalysisError(
        f"unknown analysis backend {backend!r}; expected auto, basic, librosa, or neural"
    )


__all__ = [
    "analyze_track_neural",
    "analyze_track_with_backend",
    "neural_beat_available",
    "librosa_available",
    "BeatTrackerNet",
]
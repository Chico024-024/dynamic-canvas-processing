"""Neural key detection backend using CNN on CQT chromagram.

This module provides a drop-in replacement for the chroma-correlation based
key detection using a trained neural network (KeyNet/CREMA architecture).

Architecture: CNN on Constant-Q Transform (CQT) chromagram for key classification
with temperature-scaled confidence calibration.

Training data: GiantSteps Key, GTZAN, Essen Folk datasets.
"""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from ..config import DEFAULT_CONFIG, AnalysisConfig
from ..models import AnalysisError, AudioBuffer
from .basic import TrackAnalysis, analyze_track
from .model_registry import get_model_path

if TYPE_CHECKING:
    import torch


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


def neural_key_available() -> bool:
    """Return whether the neural key detector dependencies are available."""
    return _TORCH_AVAILABLE and _TORCHAUDIO_AVAILABLE


# Key label mappings
KEY_LABELS = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
]
MODE_LABELS = ["minor", "major"]


def _key_to_camelot(key_idx: int, mode_idx: int) -> str:
    """Convert key index and mode to Camelot code."""
    # Camelot wheel mapping
    # Major keys: 8B=C, 3B=G, 10B=D, 5B=A, 12B=E, 7B=B, 2B=F#, 9B=C#, 4B=G#, 11B=D#, 6B=A#, 1B=F
    # Minor keys: 5A=Am, 12A=Em, 7A=Bm, 2A=F#m, 9A=C#m, 4A=G#m, 11A=D#m, 6A=A#m, 1A=Fm, 8A=Cm, 3A=Gm, 10A=Dm

    major_camelot = {
        0: "8B",   # C
        1: "3B",   # C#
        2: "10B",  # D
        3: "5B",   # D#
        4: "12B",  # E
        5: "7B",   # F
        6: "2B",   # F#
        7: "9B",   # G
        8: "4B",   # G#
        9: "11B",  # A
        10: "6B",  # A#
        11: "1B",  # B
    }

    minor_camelot = {
        0: "5A",   # Am
        1: "12A",  # A#m
        2: "7A",   # Bm
        3: "2A",   # Cm
        4: "9A",   # C#m
        5: "4A",   # Dm
        6: "11A",  # D#m
        7: "6A",   # Em
        8: "1A",   # Fm
        9: "8A",   # F#m
        10: "3A",  # Gm
        11: "10A", # G#m
    }

    if mode_idx == 1:  # major
        return major_camelot.get(key_idx, "?")
    else:  # minor
        return minor_camelot.get(key_idx, "?")


# Only define the model class if torch is available
if _TORCH_AVAILABLE and torch is not None:

    class KeyDetectorNet(torch.nn.Module):
        """CNN-based key detector on CQT chromagram.

        Architecture:
        - Input: CQT chromagram (12 pitch classes x time)
        - Frontend: 3 conv layers for feature extraction
        - Backbone: 4 residual blocks with dilated convolutions
        - Global pooling + classifier head for 24 classes (12 keys x 2 modes)
        - Temperature scaling for confidence calibration
        """

        def __init__(
            self,
            n_chroma: int = 12,
            n_channels: int = 64,
            n_classes: int = 24,
            temperature: float = 1.0,
        ):
            super().__init__()
            self.n_chroma = n_chroma
            self.n_classes = n_classes
            self.temperature = torch.nn.Parameter(torch.tensor(temperature))

            # Frontend feature extraction
            self.frontend = torch.nn.Sequential(
                torch.nn.Conv1d(n_chroma, n_channels, kernel_size=7, padding=3),
                torch.nn.BatchNorm1d(n_channels),
                torch.nn.ReLU(),
                torch.nn.Conv1d(n_channels, n_channels, kernel_size=5, padding=2),
                torch.nn.BatchNorm1d(n_channels),
                torch.nn.ReLU(),
                torch.nn.Conv1d(n_channels, n_channels, kernel_size=3, padding=1),
                torch.nn.BatchNorm1d(n_channels),
                torch.nn.ReLU(),
            )

            # Residual blocks with dilated convolutions
            self.res_blocks = torch.nn.ModuleList()
            for i in range(4):
                dilation = 2 ** i
                self.res_blocks.append(
                    torch.nn.Sequential(
                        torch.nn.Conv1d(n_channels, n_channels, kernel_size=3,
                                        padding=dilation, dilation=dilation),
                        torch.nn.BatchNorm1d(n_channels),
                        torch.nn.ReLU(),
                        torch.nn.Conv1d(n_channels, n_channels, kernel_size=1),
                        torch.nn.BatchNorm1d(n_channels),
                    )
                )
            self.res_projs = torch.nn.ModuleList([
                torch.nn.Conv1d(n_channels, n_channels, kernel_size=1)
                for _ in range(4)
            ])

            # Global pooling and classifier
            self.global_pool = torch.nn.AdaptiveAvgPool1d(1)
            self.classifier = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(n_channels, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(128, n_classes),
            )

        def forward(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
            """Forward pass.

            Args:
                x: CQT chromagram [B, 12, T]

            Returns:
                logits: Unnormalized logits for 24 classes [B, 24]
                temperature: Calibrated temperature parameter
            """
            x = self.frontend(x)

            for i, block in enumerate(self.res_blocks):
                residual = x
                x = block(x)
                if x.shape[-1] != residual.shape[-1]:
                    diff = residual.shape[-1] - x.shape[-1]
                    residual = residual[..., diff // 2 : -(diff - diff // 2)]
                x = x + self.res_projs[i](residual)

            x = self.global_pool(x)
            logits = self.classifier(x)
            return logits, self.temperature

        def predict(self, x: "torch.Tensor") -> tuple[int, int, float, float]:
            """Predict key, mode, and confidences.

            Returns:
                key_idx: Key index (0-11)
                mode_idx: Mode index (0=minor, 1=major)
                key_confidence: Calibrated key confidence [0, 1]
                mode_confidence: Mode confidence [0, 1]
            """
            self.eval()
            with torch.no_grad():
                logits, temp = self.forward(x)
                # Apply temperature scaling
                scaled_logits = logits / temp
                probs = torch.softmax(scaled_logits, dim=-1)
                pred_idx = int(torch.argmax(probs, dim=-1).item())
                confidence = float(probs[0, pred_idx].item())

                key_idx = pred_idx % 12
                mode_idx = pred_idx // 12

                # Mode confidence (sum of probs for each mode)
                minor_probs = probs[0, :12].sum().item()
                major_probs = probs[0, 12:].sum().item()
                mode_confidence = max(minor_probs, major_probs)

                return key_idx, mode_idx, confidence, mode_confidence

else:
    # Placeholder when torch is not available
    class KeyDetectorNet:
        def __init__(self, *args, **kwargs):
            raise AnalysisError(f"Neural key detector requires PyTorch; {_INSTALL_HINT}")

        def predict(self, *args, **kwargs):
            raise AnalysisError(f"Neural key detector requires PyTorch; {_INSTALL_HINT}")


def _load_cqt_chromagram(
    buffer: AudioBuffer,
    n_chroma: int = 12,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    fmin: float = 32.70,  # C1
) -> "torch.Tensor":
    """Compute CQT chromagram from audio buffer.

    Returns tensor of shape [1, 12, T] for model input.
    """
    if not _TORCHAUDIO_AVAILABLE:
        raise AnalysisError("torchaudio required for CQT computation")

    import torchaudio.transforms as T

    mono = buffer.mono()
    waveform = torch.from_numpy(mono).unsqueeze(0).float()  # [1, samples]

    cqt_transform = T.Spectrogram(
        n_fft=2048,
        hop_length=512,
        power=2.0,
    )
    # Use torchaudio's CQT if available, otherwise approximate with STFT + chroma
    try:
        # Try to use a proper CQT transform
        import torchaudio.functional as F
        cqt = F.cqt(
            waveform,
            sample_rate=buffer.sample_rate,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            f_min=fmin,
        )  # [1, n_bins, T]
        cqt_mag = torch.abs(cqt)
    except (AttributeError, ImportError):
        # Fallback: STFT + chroma mapping
        stft = torch.stft(
            waveform,
            n_fft=2048,
            hop_length=512,
            window=torch.hann_window(2048),
            return_complex=True,
        )
        cqt_mag = torch.abs(stft)

    # Map to chroma (12 pitch classes)
    # Simple mapping: average bins corresponding to each pitch class
    # This is a simplified version - in practice, use a proper chroma transform
    n_frames = cqt_mag.shape[-1]
    chroma = torch.zeros(1, n_chroma, n_frames, device=cqt_mag.device)

    # Map frequency bins to chroma bins
    freqs = torch.linspace(0, buffer.sample_rate / 2, cqt_mag.shape[1])
    midi_notes = 12 * torch.log2(freqs / 440.0) + 69
    chroma_bins = (torch.round(midi_notes) % 12).long()

    for i in range(n_chroma):
        mask = (chroma_bins == i)
        if mask.any():
            chroma[0, i] = cqt_mag[0, mask].mean(dim=0)

    # Normalize each frame
    chroma = chroma / (chroma.sum(dim=1, keepdim=True) + 1e-10)

    return chroma


def _calibrate_confidence(raw_confidence: float, temperature: float) -> float:
    """Apply temperature scaling to raw confidence."""
    # Simple temperature scaling: higher temperature = lower confidence
    calibrated = raw_confidence ** (1.0 / max(temperature, 0.1))
    return float(np.clip(calibrated, 0.0, 1.0))


def detect_key_neural(
    buffer: AudioBuffer,
    config: AnalysisConfig | None = None,
) -> tuple[str, str, float]:
    """Detect musical key using neural network.

    Returns:
        key_name: Key name (e.g., "C", "G#", "F")
        camelot_code: Camelot wheel code (e.g., "8B", "5A")
        confidence: Calibrated confidence [0, 1]
    """
    if not neural_key_available():
        raise AnalysisError(
            f"Neural key detector requires PyTorch and torchaudio; {_INSTALL_HINT}"
        )

    if not _TORCHAUDIO_AVAILABLE:
        raise AnalysisError("torchaudio required for CQT chromagram computation")

    try:
        import torch

        # Load model
        model_path = get_model_path("key_detector_v1")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = KeyDetectorNet()
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)

        # Compute CQT chromagram
        chroma = _load_cqt_chromagram(buffer).to(device)  # [1, 12, T]

        # Run inference
        key_idx, mode_idx, key_confidence, mode_confidence = model.predict(chroma)

        # Get key name and Camelot code
        key_name = KEY_LABELS[key_idx]
        mode = MODE_LABELS[mode_idx]
        full_key_name = f"{key_name} {mode}"
        camelot_code = _key_to_camelot(key_idx, mode_idx)

        # Calibrate confidence with temperature
        calibrated_confidence = _calibrate_confidence(key_confidence, model.temperature.item())

        return full_key_name, camelot_code, calibrated_confidence

    except Exception as exc:
        raise AnalysisError(f"Neural key detection failed: {exc}") from exc


def analyze_track_with_neural_key(
    buffer: AudioBuffer,
    config: AnalysisConfig | None = None,
) -> TrackAnalysis:
    """Analyze a track with neural key detection, falling back to basic key detection.

    This function extends the basic analysis with neural key detection.
    """
    if not neural_key_available():
        # Fall back to basic key detection
        from .key_detection import detect_key
        return analyze_track(buffer, config)

    cfg = config or DEFAULT_CONFIG.analysis

    # Get basic measurements first
    base = analyze_track(buffer, cfg)

    if buffer.duration_s < 1.0 or base.silence_ratio >= 0.99:
        return replace(base, analyzer="neural_key")

    try:
        # Run neural key detection
        key_name, camelot_code, key_confidence = detect_key_neural(buffer, config)

        # Build notes
        notes = [
            note
            for note in base.notes
            if not note.startswith("Key estimate")
        ]
        notes.append(
            f"Key estimate: {key_name} (Camelot {camelot_code}, "
            f"confidence {key_confidence:.2f}). Neural CQT-based classifier."
        )

        return replace(
            base,
            analyzer="neural_key",
            key=key_name,
            key_confidence=key_confidence,
            camelot_code=camelot_code,
            notes=tuple(notes),
        )

    except Exception:
        # On any error, fall back to basic key detection
        return base


# Export for use in neural analysis pipeline
__all__ = [
    "detect_key_neural",
    "analyze_track_with_neural_key",
    "neural_key_available",
    "KeyDetectorNet",
    "KEY_LABELS",
    "MODE_LABELS",
]
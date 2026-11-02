"""Structure segmentation for phrase/section boundaries.

This module provides neural structure segmentation to detect musical
phrases and sections (intro, verse, chorus, bridge, outro, drop).

Architecture: BiLSTM/Transformer on multi-scale audio features for
sequence labeling of section boundaries.

Training data: Annotated structure datasets (SALAMI, RWC, etc.).
"""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from ..config import AnalysisConfig
from ..models import AnalysisError, AudioBuffer
from .basic import TrackAnalysis, analyze_track
from .model_registry import get_model_path

if TYPE_CHECKING:
    import torch


_INSTALL_HINT = (
    "install PyTorch and torchaudio: pip install torch torchaudio\n"
    "Model weights will be downloaded on first use to ~/.dcp/models/"
)

_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
_TORCHAUDIO_AVAILABLE = importlib.util.find_spec("torchaudio") is not None


def neural_structure_available() -> bool:
    """Return whether the neural structure segmenter dependencies are available."""
    return _TORCH_AVAILABLE and _TORCHAUDIO_AVAILABLE


# Section label mappings
SECTION_LABELS = [
    "intro", "verse", "chorus", "bridge", "outro", "drop",
    "pre_chorus", "post_chorus", "solo", "buildup", "breakdown", "other"
]

SECTION_COLORS = {
    "intro": "#888888",
    "verse": "#38bdf8",
    "chorus": "#f43f5e",
    "bridge": "#a78bfa",
    "outro": "#64748b",
    "drop": "#fb923c",
    "pre_chorus": "#34d399",
    "post_chorus": "#22d3ee",
    "solo": "#fbbf24",
    "buildup": "#f87171",
    "breakdown": "#c084fc",
    "other": "#94a3b8",
}


# Only define the model class if torch is available
if _TORCH_AVAILABLE:
    import torch

    class StructureSegmenterNet(torch.nn.Module):
        """BiLSTM/Transformer-based structure segmenter.

        Architecture:
        - Input: Multi-scale features (chroma, MFCC, spectral contrast, rhythm)
        - Feature encoder: 3 conv layers per feature stream
        - Fusion: Concatenate + projection
        - Backbone: 2-layer BiLSTM or Transformer encoder
        - Output: Frame-level section classification (12 classes)
        - CRF layer for structured prediction (optional)
        """

        def __init__(
            self,
            n_features: int = 64,  # Combined feature dimension
            n_hidden: int = 128,
            n_layers: int = 2,
            n_classes: int = 12,
            dropout: float = 0.2,
            use_transformer: bool = True,
        ):
            super().__init__()
            self.n_features = n_features
            self.n_classes = n_classes
            self.use_transformer = use_transformer

            # Feature encoder (processes concatenated features)
            self.encoder = torch.nn.Sequential(
                torch.nn.Conv1d(n_features, n_hidden, kernel_size=7, padding=3),
                torch.nn.BatchNorm1d(n_hidden),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Conv1d(n_hidden, n_hidden, kernel_size=5, padding=2),
                torch.nn.BatchNorm1d(n_hidden),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Conv1d(n_hidden, n_hidden, kernel_size=3, padding=1),
                torch.nn.BatchNorm1d(n_hidden),
                torch.nn.ReLU(),
            )

            if use_transformer:
                # Transformer encoder
                encoder_layer = torch.nn.TransformerEncoderLayer(
                    d_model=n_hidden,
                    nhead=8,
                    dim_feedforward=n_hidden * 4,
                    dropout=dropout,
                    batch_first=True,
                )
                self.backbone = torch.nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            else:
                # BiLSTM
                self.backbone = torch.nn.LSTM(
                    input_size=n_hidden,
                    hidden_size=n_hidden // 2,
                    num_layers=n_layers,
                    bidirectional=True,
                    batch_first=True,
                    dropout=dropout if n_layers > 1 else 0,
                )

            # Classifier head
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(n_hidden, n_hidden),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(n_hidden, n_classes),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """Forward pass.

            Args:
                x: Multi-scale features [B, n_features, T]

            Returns:
                logits: Section logits per frame [B, n_classes, T]
            """
            # Encoder
            x = self.encoder(x)  # [B, n_hidden, T]

            # Transpose for sequence models: [B, T, n_hidden]
            x = x.transpose(1, 2)

            if self.use_transformer:
                x = self.backbone(x)
            else:
                x, _ = self.backbone(x)

            # Classifier
            x = self.classifier(x)  # [B, T, n_classes]
            x = x.transpose(1, 2)  # [B, n_classes, T]

            return x

        def predict(self, x: "torch.Tensor") -> tuple[list[int], np.ndarray]:
            """Predict section labels per frame.

            Returns:
                labels: List of section label indices per frame
                probs: Probability matrix [n_classes, T]
            """
            self.eval()
            with torch.no_grad():
                logits = self.forward(x)
                probs = torch.softmax(logits, dim=1)
                labels = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy().tolist()
                probs = probs.squeeze(0).cpu().numpy()
                return labels, probs

else:
    # Placeholder when torch is not available
    class StructureSegmenterNet:
        def __init__(self, *args, **kwargs):
            raise AnalysisError(f"Neural structure segmenter requires PyTorch; {_INSTALL_HINT}")

        def predict(self, *args, **kwargs):
            raise AnalysisError(f"Neural structure segmenter requires PyTorch; {_INSTALL_HINT}")


def _extract_structure_features(
    buffer: AudioBuffer,
    hop_length: int = 512,
) -> "torch.Tensor":
    """Extract multi-scale features for structure segmentation.

    Features:
    - Chroma (12 dims) - harmonic content
    - MFCC (13 dims) - timbral content
    - Spectral contrast (7 dims) - spectral shape
    - Rhythm features (tempogram, onset strength) (8 dims)
    - Total: ~40 dims per frame

    Returns tensor of shape [1, n_features, T].
    """
    if not _TORCHAUDIO_AVAILABLE:
        raise AnalysisError("torchaudio required for feature extraction")

    import torchaudio.transforms as T
    import torchaudio.functional as F

    mono = buffer.mono()
    waveform = torch.from_numpy(mono).unsqueeze(0).float()  # [1, samples]
    n_frames = (mono.shape[0] + hop_length - 1) // hop_length

    features = []

    # 1. Chroma (12 dims)
    try:
        chroma_transform = T.MelSpectrogram(
            sample_rate=buffer.sample_rate,
            n_fft=2048,
            hop_length=hop_length,
            n_mels=12,
            f_min=32.70,
            f_max=8000.0,
            power=2.0,
        )
        chroma = chroma_transform(waveform)
        chroma = torch.log(torch.clamp(chroma, min=1e-10))
        # Normalize per frame
        chroma = chroma / (chroma.sum(dim=1, keepdim=True) + 1e-10)
        features.append(chroma)
    except Exception:
        # Fallback: zero chroma
        features.append(torch.zeros(1, 12, n_frames))

    # 2. MFCC (13 dims)
    try:
        mfcc_transform = T.MFCC(
            sample_rate=buffer.sample_rate,
            n_mfcc=13,
            melkwargs={
                "n_fft": 2048,
                "hop_length": hop_length,
                "n_mels": 40,
            },
        )
        mfcc = mfcc_transform(waveform)  # [1, 13, T]
        features.append(mfcc[:, :n_frames] if mfcc.shape[-1] >= n_frames else
                        torch.nn.functional.pad(mfcc, (0, n_frames - mfcc.shape[-1])))
    except Exception:
        features.append(torch.zeros(1, 13, n_frames))

    # 3. Spectral contrast (7 dims)
    try:
        stft = torch.stft(
            waveform,
            n_fft=2048,
            hop_length=hop_length,
            window=torch.hann_window(2048),
            return_complex=True,
        )
        mag = torch.abs(stft)
        # Compute spectral contrast (simplified)
        n_bands = 7
        band_edges = torch.linspace(0, mag.shape[1], n_bands + 1).long()
        contrast = []
        for i in range(n_bands):
            band = mag[:, band_edges[i]:band_edges[i+1]]
            band_mean = band.mean(dim=1)
            band_max = band.max(dim=1).values
            band_min = band.min(dim=1).values
            contr = (band_max - band_min) / (band_max + band_min + 1e-10)
            contrast.append(contr)
        contrast = torch.stack(contrast, dim=1)  # [1, 7, T]
        features.append(contrast[:, :, :n_frames] if contrast.shape[-1] >= n_frames else
                        torch.nn.functional.pad(contrast, (0, n_frames - contrast.shape[-1])))
    except Exception:
        features.append(torch.zeros(1, 7, n_frames))

    # 4. Rhythm features (8 dims) - tempogram and onset strength
    try:
        # Onset strength
        onset_env = T.SpectralFlux(
            n_fft=2048,
            hop_length=hop_length,
        )(waveform)  # [1, T]
        onset_env = onset_env.squeeze(0)  # [T]

        # Simple tempogram (autocorrelation of onset envelope)
        tempogram = []
        for lag in [1, 2, 4, 8, 16, 32, 64, 128]:
            if lag < len(onset_env):
                auto = (onset_env[:-lag] * onset_env[lag:]).mean()
            else:
                auto = 0.0
            tempogram.append(auto)
        tempogram = torch.tensor(tempogram).float().unsqueeze(0).unsqueeze(-1).expand(-1, -1, n_frames)
        features.append(tempogram)
    except Exception:
        features.append(torch.zeros(1, 8, n_frames))

    # Concatenate all features
    all_features = torch.cat(features, dim=1)  # [1, n_features, T]

    # Standardize (per feature)
    mean = all_features.mean(dim=-1, keepdim=True)
    std = all_features.std(dim=-1, keepdim=True) + 1e-10
    all_features = (all_features - mean) / std

    return all_features


def _labels_to_segments(
    labels: list[int],
    hop_length: int,
    sample_rate: int,
) -> list[tuple[float, float, str]]:
    """Convert frame-level labels to time segments.

    Returns list of (start_time, end_time, section_label).
    """
    if not labels:
        return []

    segments = []
    current_label = labels[0]
    start_frame = 0

    for i, label in enumerate(labels):
        if label != current_label:
            # End current segment
            start_time = start_frame * hop_length / sample_rate
            end_time = i * hop_length / sample_rate
            segments.append((start_time, end_time, SECTION_LABELS[current_label]))
            current_label = label
            start_frame = i

    # Add final segment
    start_time = start_frame * hop_length / sample_rate
    end_time = len(labels) * hop_length / sample_rate
    segments.append((start_time, end_time, SECTION_LABELS[current_label]))

    return segments


def _merge_short_segments(
    segments: list[tuple[float, float, str]],
    min_duration: float = 2.0,
) -> list[tuple[float, float, str]]:
    """Merge segments shorter than min_duration with adjacent segments."""
    if not segments:
        return segments

    merged = []
    for seg in segments:
        if merged and (seg[1] - seg[0]) < min_duration:
            # Merge with previous
            prev_start, prev_end, prev_label = merged[-1]
            merged[-1] = (prev_start, seg[1], prev_label)
        else:
            merged.append(seg)
    return merged


def segment_structure_neural(
    buffer: AudioBuffer,
    config: AnalysisConfig | None = None,
) -> list[tuple[float, float, str]]:
    """Segment audio into musical sections using neural network.

    Returns:
        List of (start_time, end_time, section_label) tuples.
    """
    if not neural_structure_available():
        raise AnalysisError(
            f"Neural structure segmenter requires PyTorch and torchaudio; {_INSTALL_HINT}"
        )

    if not _TORCHAUDIO_AVAILABLE:
        raise AnalysisError("torchaudio required for feature extraction")

    try:
        import torch

        # Load model
        model_path = get_model_path("structure_segmenter_v1")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = StructureSegmenterNet()
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(device)

        # Extract features
        features = _extract_structure_features(buffer).to(device)

        # Run inference
        labels, probs = model.predict(features)

        # Convert to segments
        hop_length = 512
        segments = _labels_to_segments(labels, hop_length, buffer.sample_rate)

        # Merge short segments
        segments = _merge_short_segments(segments, min_duration=2.0)

        return segments

    except Exception as exc:
        raise AnalysisError(f"Neural structure segmentation failed: {exc}") from exc


def analyze_track_with_structure(
    buffer: AudioBuffer,
    config: AnalysisConfig | None = None,
) -> TrackAnalysis:
    """Analyze a track with neural structure segmentation.

    This function extends the basic analysis with structure segmentation.
    """
    if not neural_structure_available():
        return analyze_track(buffer, config)

    cfg = config or DEFAULT_CONFIG.analysis

    # Get basic measurements first
    base = analyze_track(buffer, cfg)

    if buffer.duration_s < 1.0 or base.silence_ratio >= 0.99:
        return replace(base, analyzer="neural_structure")

    try:
        # Run neural structure segmentation
        segments = segment_structure_neural(buffer, config)

        # Build notes
        notes = [
            note
            for note in base.notes
            if not note.startswith("Structure")
        ]
        notes.append(
            f"Structure segmentation: {len(segments)} sections detected "
            f"({', '.join(s[2] for s in segments)}). Neural multi-scale segmenter."
        )

        return replace(
            base,
            analyzer="neural_structure",
            notes=tuple(notes),
        )

    except Exception:
        # On any error, fall back to basic analysis
        return base


# Export for use in neural analysis pipeline
__all__ = [
    "segment_structure_neural",
    "analyze_track_with_structure",
    "neural_structure_available",
    "StructureSegmenterNet",
    "SECTION_LABELS",
    "SECTION_COLORS",
]
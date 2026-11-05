"""Hierarchical 3-Tier Stem Separation Pipeline.

Tier 1: BS-RoFormer (SOTA Studio Quality, ~12.97 dB SDR)
Tier 2: Meta Demucs (Classic Neural, ~9.0 dB SDR)
Tier 3: Complementary DSP Filterbank (Bit-exact, zero external dependencies)
"""

from __future__ import annotations

import logging
from typing import Literal

from ..models import AudioBuffer, StemSet
from .demucs_adapter import DemucsAdapter, demucs_available
from .dsp_splitter import decompose_audio_to_stemset
from .roformer_adapter import RoformerAdapter, roformer_available

logger = logging.getLogger(__name__)

BackendType = Literal["roformer", "demucs", "dsp", "auto"]


def get_available_backends() -> list[str]:
    """Return available stem separation backends in priority order."""
    backends: list[str] = []
    if roformer_available():
        backends.append("roformer")
    if demucs_available():
        backends.append("demucs")
    backends.append("dsp")  # DSP is always available
    return backends


def separate_hierarchical(
    audio: AudioBuffer,
    *,
    label: str = "",
    preferred_backend: BackendType = "auto",
) -> tuple[StemSet, str]:
    """Separate an AudioBuffer into 4 stems using the 3-tier hierarchy.

    Parameters
    ----------
    audio : AudioBuffer
        Mix to separate.
    label : str
        Source label for tracking (e.g. 'A' or 'B').
    preferred_backend : 'auto', 'roformer', 'demucs', or 'dsp'
        Specific backend to prioritize, or 'auto' for best available.

    Returns
    -------
    tuple[StemSet, str]
        (separated_stem_set, backend_used)
    """
    # 1. Tier 1: BS-RoFormer
    if preferred_backend in ("auto", "roformer") and roformer_available():
        try:
            adapter = RoformerAdapter()
            stems = adapter.separate(audio)
            return stems, "roformer"
        except Exception as exc:
            logger.warning("BS-RoFormer separation failed, falling back: %s", exc)
            if preferred_backend == "roformer":
                raise

    # 2. Tier 2: Demucs
    if preferred_backend in ("auto", "demucs") and demucs_available():
        try:
            adapter = DemucsAdapter()
            stems = adapter.separate(audio)
            return stems, "demucs"
        except Exception as exc:
            logger.warning("Demucs separation failed, falling back: %s", exc)
            if preferred_backend == "demucs":
                raise

    # 3. Tier 3: Zero-dependency DSP Splitter
    stems = decompose_audio_to_stemset(audio, label=label or "dsp")
    return stems, "dsp"


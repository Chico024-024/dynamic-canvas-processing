"""Optional Demucs-based stem separation adapter.

This module provides a concrete ``StemSeparator`` implementation that wraps
Meta's Demucs model for local, CPU or GPU-based source separation. It is an
optional backend that requires ``demucs`` and ``torch`` to be installed.

DCP Core works fully without this adapter. Users can always supply stems
manually. This adapter exists to provide a turnkey separation experience
when the dependencies are available and their licensing terms are acceptable.

Licensing note: Demucs source code is MIT-licensed, but pretrained model
weights may carry separate terms depending on the training data used. Users
should verify weight licensing before commercial deployment. DCP does not
download or include any model weights; the user must install demucs and accept
its terms independently.

Content-hash caching is provided: once a track is separated, results are
cached locally under ``.dcp_cache/stems/`` by SHA-256 content hash so that
repeated runs do not re-separate the same audio.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..models import AudioBuffer, DCPError, StemSet
from .separation import StemSeparationUnavailable, StemSeparator

if TYPE_CHECKING:
    pass

# Demucs stem names → DCP canonical names
_DEMUCS_NAME_MAP: dict[str, str] = {
    "drums": "drums",
    "bass": "bass",
    "other": "other",
    "vocals": "vocals",
}

_DEFAULT_CACHE_DIR = ".dcp_cache/stems"


def demucs_available() -> bool:
    """Return whether demucs and torch can be imported."""
    try:
        import demucs
        import torch
        return True
    except (ImportError, OSError):
        return False


def _content_hash(audio: AudioBuffer) -> str:
    """Compute a deterministic hash of audio content for cache keying."""
    h = hashlib.sha256()
    h.update(audio.samples.tobytes())
    h.update(str(audio.sample_rate).encode())
    return h.hexdigest()[:16]


class DemucsAdapter:
    """Concrete ``StemSeparator`` wrapping the Demucs neural source separator.

    This adapter loads a Demucs model on first use and separates audio into
    four stems: drums, bass, other, and vocals. Results are cached locally
    by content hash to avoid re-separation of already-processed tracks.

    Parameters
    ----------
    model_name : str
        Demucs model to use. Default ``"htdemucs"`` is the hybrid transformer
        model. Other options include ``"htdemucs_ft"`` (fine-tuned) and
        ``"mdx_extra"``.
    device : str or None
        PyTorch device. ``None`` auto-selects CUDA if available, otherwise CPU.
    cache_dir : str or Path or None
        Directory for caching separated stems. Set to ``None`` to disable.
    """

    name = "demucs"

    def __init__(
        self,
        *,
        model_name: str = "htdemucs",
        device: str | None = None,
        cache_dir: str | Path | None = _DEFAULT_CACHE_DIR,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._separator = None  # lazy-loaded

    def is_available(self) -> bool:
        """Return True when demucs and torch can be imported."""
        return demucs_available()

    def _load_separator(self):
        """Lazy-load the demucs Separator on first use."""
        if self._separator is not None:
            return self._separator

        if not self.is_available():
            raise StemSeparationUnavailable(
                "Demucs stem separation requires the 'demucs' and 'torch' "
                "packages. Install them with: pip install demucs torch"
            )

        try:
            from demucs.api import Separator
        except ImportError as exc:
            raise StemSeparationUnavailable(
                f"Failed to import demucs.api: {exc}. "
                "Install with: pip install demucs"
            ) from exc

        try:
            self._separator = Separator(
                model=self._model_name,
                device=self._device,
            )
        except Exception as exc:
            raise DCPError(
                f"Failed to load Demucs model '{self._model_name}': {exc}"
            ) from exc

        return self._separator

    def _cache_path(self, content_hash: str) -> Path | None:
        """Return the cache directory for a given content hash, or None."""
        if self._cache_dir is None:
            return None
        return self._cache_dir / content_hash

    def _load_from_cache(self, audio: AudioBuffer) -> StemSet | None:
        """Try to load previously separated stems from the cache."""
        if self._cache_dir is None:
            return None

        cache_key = _content_hash(audio)
        cache_path = self._cache_path(cache_key)
        if cache_path is None or not cache_path.is_dir():
            return None

        try:
            from ..audio_io import load_audio

            stems: dict[str, AudioBuffer] = {}
            for canonical_name in ("drums", "bass", "other", "vocals"):
                wav_path = cache_path / f"{canonical_name}.wav"
                if wav_path.exists():
                    stems[canonical_name] = load_audio(str(wav_path))

            if len(stems) >= 4:
                return StemSet(stems, source_label="demucs_cached")
        except Exception:
            pass  # cache miss or corrupt cache

        return None

    def _save_to_cache(
        self, audio: AudioBuffer, stem_set: StemSet
    ) -> None:
        """Save separated stems to the content-hash cache."""
        if self._cache_dir is None:
            return

        cache_key = _content_hash(audio)
        cache_path = self._cache_path(cache_key)
        if cache_path is None:
            return

        try:
            from ..audio_io import save_audio

            cache_path.mkdir(parents=True, exist_ok=True)
            for name, stem_audio in stem_set.items():
                save_audio(cache_path / f"{name}.wav", stem_audio)
        except Exception:
            pass  # cache write failure is non-fatal

    def separate(
        self, audio: AudioBuffer, *, timeout_s: float | None = None
    ) -> StemSet:
        """Separate audio into drums, bass, other, and vocals stems.

        Parameters
        ----------
        audio : AudioBuffer
            The full mix to separate. Must be mono or stereo.
        timeout_s : float or None
            Not enforced by demucs; reserved for future use.

        Returns
        -------
        StemSet
            A StemSet with canonical stem names.
        """
        # Try cache first
        cached = self._load_from_cache(audio)
        if cached is not None:
            return cached

        separator = self._load_separator()

        try:
            import torch
        except ImportError as exc:
            raise StemSeparationUnavailable(
                f"torch is required for demucs separation: {exc}"
            ) from exc

        # Prepare input tensor: demucs expects (channels, samples) float32
        samples = audio.samples.astype(np.float32)
        if samples.shape[0] == 1:
            # Demucs expects stereo; duplicate mono to stereo
            samples = np.vstack([samples, samples])

        wav_tensor = torch.from_numpy(samples)

        try:
            # demucs.api.Separator.separate_tensor returns
            # (original, {source_name: tensor})
            _origin, separated = separator.separate_tensor(wav_tensor)
        except Exception as exc:
            raise DCPError(
                f"Demucs separation failed: {exc}. "
                "Check that the model is properly installed and audio is valid."
            ) from exc

        # Convert demucs output to DCP StemSet
        stems: dict[str, AudioBuffer] = {}
        for demucs_name, canonical_name in _DEMUCS_NAME_MAP.items():
            if demucs_name in separated:
                stem_tensor = separated[demucs_name]
                stem_np = stem_tensor.numpy().astype(np.float64)
                # Ensure correct shape: (channels, samples)
                if stem_np.ndim == 1:
                    stem_np = stem_np.reshape(1, -1)
                elif stem_np.ndim == 3:
                    # Remove batch dimension if present
                    stem_np = stem_np.squeeze(0)
                # Match original channel count
                if audio.channels == 1 and stem_np.shape[0] == 2:
                    stem_np = np.mean(stem_np, axis=0, keepdims=True)
                stems[canonical_name] = AudioBuffer(
                    samples=stem_np, sample_rate=audio.sample_rate
                )

        if not stems:
            raise DCPError(
                "Demucs returned no usable stems. Available sources: "
                + ", ".join(separated.keys())
            )

        result = StemSet(stems, source_label="demucs")

        # Cache for future use
        self._save_to_cache(audio, result)

        return result


def get_demucs_separator(
    *,
    model_name: str = "htdemucs",
    device: str | None = None,
    cache_dir: str | Path | None = _DEFAULT_CACHE_DIR,
) -> DemucsAdapter:
    """Create a configured DemucsAdapter instance.

    Raises StemSeparationUnavailable if demucs is not installed.
    """
    adapter = DemucsAdapter(
        model_name=model_name, device=device, cache_dir=cache_dir
    )
    if not adapter.is_available():
        raise StemSeparationUnavailable(
            "Demucs is not installed. Install with: pip install demucs torch\n"
            "Note: Demucs model weights may have separate licensing terms."
        )
    return adapter


__all__ = [
    "DemucsAdapter",
    "demucs_available",
    "get_demucs_separator",
]

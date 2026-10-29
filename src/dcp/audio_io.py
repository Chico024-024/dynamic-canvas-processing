"""Audio file loading, saving, and format adaptation."""

from __future__ import annotations

import re
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .models import AudioBuffer, AudioFormatError, StemSet

_SUPPORTED_CHANNELS = (1, 2)
_SAFE_STEM_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def load_audio(
    path: str | Path,
    *,
    target_sample_rate: int | None = None,
    target_channels: int | None = None,
) -> AudioBuffer:
    """Load an audio file as a de-interleaved float64 AudioBuffer.

    The original sample rate and channel count are preserved unless a target
    is given. DCP supports mono and stereo files in Core.
    """

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"audio file not found: {file_path}")
    if not file_path.is_file():
        raise AudioFormatError(f"audio path is not a file: {file_path}")

    try:
        samples, sample_rate = sf.read(
            str(file_path), always_2d=True, dtype="float64"
        )
    except (sf.LibsndfileError, RuntimeError, ValueError) as exc:
        raise AudioFormatError(
            f"could not read audio file '{file_path}': unsupported or corrupt "
            f"file ({exc})"
        ) from exc

    if samples.size == 0:
        raise AudioFormatError(f"audio file contains no samples: {file_path}")
    if not np.all(np.isfinite(samples)):
        raise AudioFormatError(
            f"audio file contains NaN or infinite samples: {file_path}"
        )
    if int(samples.shape[1]) not in _SUPPORTED_CHANNELS:
        raise AudioFormatError(
            "DCP Core supports mono or stereo audio only; "
            f"'{file_path}' has {samples.shape[1]} channels"
        )

    buffer = AudioBuffer(np.ascontiguousarray(samples.T), int(sample_rate))
    if target_sample_rate is not None:
        buffer = resample_audio(buffer, target_sample_rate)
    if target_channels is not None:
        buffer = convert_channels(buffer, target_channels)
    return buffer


def save_audio(
    path: str | Path,
    buffer: AudioBuffer,
    *,
    subtype: str | None = "PCM_24",
    allow_clipping: bool = False,
) -> Path:
    """Write an AudioBuffer to an audio file and return the path."""

    file_path = Path(path)
    if buffer.channels not in _SUPPORTED_CHANNELS:
        raise AudioFormatError(
            "DCP Core supports mono or stereo output only; "
            f"buffer has {buffer.channels} channels"
        )
    if not np.all(np.isfinite(buffer.samples)):
        raise AudioFormatError(
            f"refusing to write non-finite samples to '{file_path}'"
        )
    peak = buffer.peak
    if peak > 1.0 and not allow_clipping:
        raise AudioFormatError(
            f"refusing to write '{file_path}' with peak {peak:.4f} above 1.0; "
            "reduce the level or pass allow_clipping=True"
        )

    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        sf.write(
            str(file_path),
            buffer.samples.T,
            buffer.sample_rate,
            subtype=subtype,
            format="WAV" if file_path.suffix.lower() == ".wav" else None,
        )
    except (sf.LibsndfileError, RuntimeError, ValueError) as exc:
        raise AudioFormatError(
            f"could not write audio file '{file_path}': {exc}"
        ) from exc
    return file_path


def resample_audio(buffer: AudioBuffer, target_sample_rate: int) -> AudioBuffer:
    """Resample with a polyphase filter, preserving channel layout."""

    if target_sample_rate <= 0:
        raise AudioFormatError(
            f"target sample rate must be positive, got {target_sample_rate}"
        )
    if buffer.sample_rate == target_sample_rate:
        return buffer

    divisor = gcd(buffer.sample_rate, target_sample_rate)
    up = target_sample_rate // divisor
    down = buffer.sample_rate // divisor
    resampled = resample_poly(buffer.samples, up, down, axis=1, padtype="constant")
    return AudioBuffer(np.ascontiguousarray(resampled), target_sample_rate)


def convert_channels(buffer: AudioBuffer, target_channels: int) -> AudioBuffer:
    """Convert mono/stereo audio to the requested mono or stereo layout."""

    if target_channels not in _SUPPORTED_CHANNELS:
        raise AudioFormatError(
            "DCP Core supports mono or stereo audio only; "
            f"requested {target_channels} channels"
        )
    if buffer.channels == target_channels:
        return buffer
    if buffer.channels == 1 and target_channels == 2:
        stereo = np.repeat(buffer.samples, 2, axis=0)
        return AudioBuffer(np.ascontiguousarray(stereo), buffer.sample_rate)
    if buffer.channels == 2 and target_channels == 1:
        mono = np.mean(buffer.samples, axis=0, keepdims=True)
        return AudioBuffer(np.ascontiguousarray(mono), buffer.sample_rate)
    raise AudioFormatError(
        "DCP Core supports mono-to-stereo and stereo-to-mono conversion; "
        f"cannot convert {buffer.channels} channels to {target_channels}"
    )


def match_format(
    buffer: AudioBuffer, reference: AudioBuffer
) -> tuple[AudioBuffer, tuple[str, ...]]:
    """Adapt *buffer* to the sample rate and channel layout of *reference*."""

    notes: list[str] = []
    converted = buffer
    if converted.sample_rate != reference.sample_rate:
        converted = resample_audio(converted, reference.sample_rate)
        notes.append(
            f"resampled from {buffer.sample_rate} Hz to {reference.sample_rate} Hz"
        )
    if converted.channels != reference.channels:
        converted = convert_channels(converted, reference.channels)
        notes.append(
            f"converted from {buffer.channels} to {reference.channels} channel(s)"
        )
    return converted, tuple(notes)


def load_stem_set(directory: str | Path) -> StemSet:
    """Load every *.wav file in a directory as a named stem set."""

    stem_dir = Path(directory)
    if not stem_dir.exists():
        raise FileNotFoundError(f"stem directory not found: {stem_dir}")
    if not stem_dir.is_dir():
        raise AudioFormatError(f"stem path is not a directory: {stem_dir}")

    files = sorted(
        path for path in stem_dir.iterdir() if path.is_file() and path.suffix.lower() == ".wav"
    )
    if not files:
        raise AudioFormatError(f"no WAV stems found in directory: {stem_dir}")

    stems: dict[str, AudioBuffer] = {}
    for path in files:
        name = path.stem.strip().lower()
        if not name:
            raise AudioFormatError(f"invalid empty stem name for file: {path}")
        if name in stems:
            raise AudioFormatError(
                f"duplicate stem name '{name}' in directory: {stem_dir}"
            )
        stems[name] = load_audio(path)

    try:
        return StemSet(stems, source_label=str(stem_dir))
    except (TypeError, ValueError) as exc:
        raise AudioFormatError(f"invalid stem set in '{stem_dir}': {exc}") from exc


def save_stem_set(
    directory: str | Path,
    stems: StemSet,
    *,
    subtype: str = "PCM_24",
) -> dict[str, Path]:
    """Write each stem to ``<directory>/<name>.wav``."""

    stem_dir = Path(directory)
    stem_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, audio in stems.items():
        if not _SAFE_STEM_NAME.match(name):
            raise AudioFormatError(
                f"stem name '{name}' cannot be used as a safe filename"
            )
        written[name] = save_audio(
            stem_dir / f"{name}.wav", audio, subtype=subtype
        )
    return written

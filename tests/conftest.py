"""Shared deterministic fixtures for the DCP test suite."""

from __future__ import annotations

import numpy as np
import pytest

from dcp.analysis import TrackAnalysis
from dcp.models import AudioBuffer, StemSet
from dcp.synthetic import SyntheticTrack, generate_synthetic_track

SR = 22050
STEM_FREQUENCIES = {
    "drums": (110.0, 0.35),
    "bass": (220.0, 0.30),
    "other": (440.0, 0.22),
    "vocals": (880.0, 0.25),
}


def make_tone(
    *,
    duration_s: float = 4.0,
    frequency: float = 440.0,
    amplitude: float = 0.5,
    sample_rate: int = SR,
    channels: int = 2,
    phase: float = 0.0,
) -> AudioBuffer:
    frames = int(round(duration_s * sample_rate))
    time = np.arange(frames, dtype=np.float64) / sample_rate
    mono = amplitude * np.sin(2.0 * np.pi * frequency * time + phase)
    if channels == 1:
        samples = mono[np.newaxis, :]
    else:
        samples = np.repeat(mono[np.newaxis, :], channels, axis=0)
    return AudioBuffer(np.ascontiguousarray(samples), sample_rate)


def make_constant(
    value: float,
    *,
    duration_s: float = 4.0,
    sample_rate: int = SR,
    channels: int = 2,
) -> AudioBuffer:
    frames = int(round(duration_s * sample_rate))
    samples = np.full((channels, frames), float(value), dtype=np.float64)
    return AudioBuffer(samples, sample_rate)


def make_silence(
    *, duration_s: float = 4.0, sample_rate: int = SR, channels: int = 2
) -> AudioBuffer:
    return make_constant(0.0, duration_s=duration_s, sample_rate=sample_rate, channels=channels)


def make_stems(
    *, duration_s: float = 8.0, sample_rate: int = SR, channels: int = 2
) -> StemSet:
    stems = {
        name: make_tone(
            duration_s=duration_s,
            frequency=frequency,
            amplitude=amplitude,
            sample_rate=sample_rate,
            channels=channels,
        )
        for name, (frequency, amplitude) in STEM_FREQUENCIES.items()
    }
    return StemSet(stems, source_label="test-stems")


def make_analysis(
    buffer: AudioBuffer,
    *,
    tempo_bpm: float | None = 120.0,
    confidence: float = 0.9,
    ambiguous: bool = False,
    beat_times_s: tuple[float, ...] = (),
    analyzer: str = "basic",
    key: str | None = None,
    camelot_code: str | None = None,
    key_confidence: float = 0.0,
    downbeat_times_s: tuple[float, ...] = (),
) -> TrackAnalysis:
    return TrackAnalysis(
        duration_s=buffer.duration_s,
        sample_rate=buffer.sample_rate,
        channels=buffer.channels,
        peak_dbfs=-6.0,
        rms_dbfs=-12.0,
        silence_ratio=0.0,
        tempo_bpm=tempo_bpm,
        tempo_confidence=confidence if tempo_bpm is not None else 0.0,
        onset_count=32,
        tempo_ambiguous=ambiguous,
        notes=(),
        beat_times_s=tuple(beat_times_s),
        analyzer=analyzer,
        key=key,
        camelot_code=camelot_code,
        key_confidence=key_confidence,
        downbeat_times_s=tuple(downbeat_times_s),
    )


@pytest.fixture(scope="session")
def synthetic_pair() -> tuple[SyntheticTrack, SyntheticTrack]:
    """Two 16-second, 22.05 kHz synthetic songs with four stems each."""

    song_a = generate_synthetic_track(
        bpm=120.0,
        duration_s=16.0,
        sample_rate=SR,
        seed=7,
        root_midi=57,
        mode="minor",
        vocal_lead_in_s=0.4,
        vocal_lead_out_s=0.8,
    )
    song_b = generate_synthetic_track(
        bpm=124.0,
        duration_s=16.0,
        sample_rate=SR,
        seed=8,
        root_midi=60,
        mode="major",
        vocal_lead_in_s=1.0,
        vocal_lead_out_s=0.4,
    )
    return song_a, song_b

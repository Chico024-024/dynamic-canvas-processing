"""Deterministic synthetic music generation for demos and tests.

The generator produces four stylized stems (drums, bass, other, vocals) and a
mix that is exactly their sum. It is not a substitute for real recordings; it
exists so that the full DCP workflow can be reproduced without copyrighted
audio, a GPU, or model downloads.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import AudioBuffer, StemSet

KEY_NAMES: tuple[str, ...] = (
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
)

_SCALE_MINOR = (0, 2, 3, 5, 7, 8, 10)
_SCALE_MAJOR = (0, 2, 4, 5, 7, 9, 11)
_MINOR_PROGRESSION = ((0, "minor"), (-4, "major"), (3, "major"), (-2, "major"))
_MAJOR_PROGRESSION = ((0, "major"), (-3, "minor"), (-7, "major"), (-5, "major"))
_CHORD_INTERVALS = {"minor": (0, 3, 7), "major": (0, 4, 7)}
_STEM_TARGET_PEAKS = {
    "drums": 0.55,
    "bass": 0.40,
    "other": 0.28,
    "vocals": 0.32,
}


@dataclass(frozen=True)
class SyntheticTrack:
    """A generated mix, its stems, and the parameters used to build it."""

    mix: AudioBuffer
    stems: StemSet
    bpm: float
    duration_s: float
    sample_rate: int
    channels: int
    seed: int
    root_midi: int
    mode: str
    vocal_lead_in_s: float
    vocal_lead_out_s: float

    @property
    def key_name(self) -> str:
        return f"{KEY_NAMES[self.root_midi % 12]} {self.mode}"

    def metadata(self) -> dict[str, object]:
        return {
            "bpm": self.bpm,
            "key": self.key_name,
            "root_midi": self.root_midi,
            "mode": self.mode,
            "duration_s": round(self.duration_s, 6),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "seed": self.seed,
            "stems": list(self.stems.names),
            "vocal_lead_in_s": self.vocal_lead_in_s,
            "vocal_lead_out_s": self.vocal_lead_out_s,
            "synthetic": True,
        }


def midi_to_hz(midi: float) -> float:
    return float(440.0 * (2.0 ** ((float(midi) - 69.0) / 12.0)))


def _equal_power_pan(pan: float) -> tuple[float, float]:
    angle = (float(np.clip(pan, -1.0, 1.0)) + 1.0) * np.pi / 4.0
    return float(np.cos(angle)), float(np.sin(angle))


def _tail_fade(signal: np.ndarray, sample_rate: int, length_s: float = 0.006) -> np.ndarray:
    n = signal.shape[0]
    fade_n = min(n, max(1, int(round(length_s * sample_rate))))
    ramp = np.linspace(1.0, 0.0, fade_n, endpoint=True)
    signal = signal.copy()
    signal[-fade_n:] *= ramp
    return signal


def _edge_fade(samples: np.ndarray, sample_rate: int, length_s: float = 0.015) -> None:
    fade_n = min(samples.shape[1] // 2, max(1, int(round(length_s * sample_rate))))
    if fade_n <= 0:
        return
    ramp = np.linspace(0.0, 1.0, fade_n, endpoint=True)
    samples[:, :fade_n] *= ramp
    samples[:, -fade_n:] *= ramp[::-1]


def _add_stereo(
    target: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    start_s: float,
    sample_rate: int,
    gain: float = 1.0,
) -> None:
    start = int(round(start_s * sample_rate))
    if start >= target.shape[1]:
        return
    start = max(0, start)
    count = min(left.shape[0], target.shape[1] - start)
    if count <= 0:
        return
    target[0, start : start + count] += left[:count] * gain
    target[1, start : start + count] += right[:count] * gain


def _add_mono(
    target: np.ndarray,
    signal: np.ndarray,
    start_s: float,
    sample_rate: int,
    gain: float = 1.0,
    pan: float = 0.0,
) -> None:
    left_gain, right_gain = _equal_power_pan(pan)
    _add_stereo(
        target,
        signal * left_gain,
        signal * right_gain,
        start_s,
        sample_rate,
        gain,
    )


def _adsr(
    n: int,
    sample_rate: int,
    *,
    attack_s: float,
    decay_s: float,
    sustain: float,
    release_s: float,
) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / sample_rate
    total_s = n / sample_rate
    attack = max(attack_s, 1e-4)
    decay = max(decay_s, 1e-4)
    release = max(release_s, 1e-4)

    attack_env = np.clip(t / attack, 0.0, 1.0)
    attack_env = np.sin(attack_env * np.pi / 2.0) ** 2
    decay_env = sustain + (1.0 - sustain) * np.exp(
        -np.maximum(t - attack, 0.0) / decay
    )
    env = attack_env * decay_env

    release_start = max(0.0, total_s - release)
    release_env = np.clip((t - release_start) / release, 0.0, 1.0)
    env *= np.cos(release_env * np.pi / 2.0) ** 2
    return env


def _kick(sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    n = max(1, int(round(0.28 * sample_rate)))
    t = np.arange(n, dtype=np.float64) / sample_rate
    frequency = 46.0 + 108.0 * np.exp(-t / 0.028)
    phase = 2.0 * np.pi * np.cumsum(frequency) / sample_rate
    body = np.sin(phase) * np.exp(-t / 0.085)
    click = rng.standard_normal(n) * np.exp(-t / 0.003) * 0.30
    signal = np.tanh(1.7 * (body + click))
    return _tail_fade(signal, sample_rate, 0.012)


def _snare(sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    n = max(1, int(round(0.20 * sample_rate)))
    t = np.arange(n, dtype=np.float64) / sample_rate
    noise = rng.standard_normal(n)
    noise = np.diff(noise, prepend=0.0)
    noise *= np.exp(-t / 0.055)
    tone = 0.40 * np.sin(2.0 * np.pi * 185.0 * t) * np.exp(-t / 0.040)
    signal = 0.55 * noise + tone
    signal = signal / max(float(np.max(np.abs(signal))), 1e-9)
    return _tail_fade(signal, sample_rate, 0.008)


def _hat(sample_rate: int, rng: np.random.Generator, *, open_hat: bool) -> np.ndarray:
    length_s = 0.14 if open_hat else 0.045
    n = max(1, int(round(length_s * sample_rate)))
    t = np.arange(n, dtype=np.float64) / sample_rate
    noise = rng.standard_normal(n)
    noise = np.diff(np.diff(noise, prepend=0.0), prepend=0.0)
    decay = 0.045 if open_hat else 0.012
    signal = noise * np.exp(-t / decay)
    signal = signal / max(float(np.max(np.abs(signal))), 1e-9)
    return _tail_fade(signal, sample_rate, 0.004)


def _drums(
    *, bpm: float, duration_s: float, sample_rate: int, rng: np.random.Generator
) -> np.ndarray:
    n = max(1, int(round(duration_s * sample_rate)))
    out = np.zeros((2, n), dtype=np.float64)
    beat_s = 60.0 / bpm
    total_beats = int(np.floor(duration_s / beat_s + 1e-9))

    kicks = [_kick(sample_rate, rng) for _ in range(4)]
    snares = [_snare(sample_rate, rng) for _ in range(4)]
    closed_hats = [_hat(sample_rate, rng, open_hat=False) for _ in range(2)]
    open_hats = [_hat(sample_rate, rng, open_hat=True) for _ in range(2)]

    for beat_index in range(total_beats):
        beat_in_bar = beat_index % 4
        beat_start = beat_index * beat_s
        if beat_in_bar in (0, 2):
            _add_mono(
                out,
                kicks[beat_index % len(kicks)],
                beat_start,
                sample_rate,
                gain=0.95,
            )
        if beat_in_bar in (1, 3):
            _add_mono(
                out,
                snares[beat_index % len(snares)],
                beat_start,
                sample_rate,
                gain=0.78,
            )

        for half_index, offset_beats in enumerate((0.0, 0.5)):
            start = beat_start + offset_beats * beat_s
            if start >= duration_s:
                continue
            is_open = beat_in_bar == 3 and half_index == 1
            hat = (
                open_hats[beat_index % len(open_hats)]
                if is_open
                else closed_hats[(beat_index + half_index) % len(closed_hats)]
            )
            pan = 0.32 if (beat_index + half_index) % 2 == 0 else -0.32
            _add_mono(
                out,
                hat,
                start,
                sample_rate,
                gain=0.55 if is_open else 0.30,
                pan=pan,
            )
    return out


def _bass_note(midi: float, length_s: float, sample_rate: int) -> np.ndarray:
    n = max(1, int(round(length_s * sample_rate)))
    t = np.arange(n, dtype=np.float64) / sample_rate
    frequency = midi_to_hz(midi)
    phase = 2.0 * np.pi * frequency * t
    signal = (
        np.sin(phase) + 0.28 * np.sin(2.0 * phase) + 0.08 * np.sin(3.0 * phase)
    )
    env = _adsr(
        n,
        sample_rate,
        attack_s=0.008,
        decay_s=0.07,
        sustain=0.75,
        release_s=0.035,
    )
    return np.tanh(1.2 * signal * env)


def _bass(
    *,
    bpm: float,
    duration_s: float,
    sample_rate: int,
    root_midi: int,
    mode: str,
) -> np.ndarray:
    n = max(1, int(round(duration_s * sample_rate)))
    out = np.zeros((2, n), dtype=np.float64)
    beat_s = 60.0 / bpm
    eighth_s = beat_s / 2.0
    bar_s = beat_s * 4.0
    progression = _MINOR_PROGRESSION if mode == "minor" else _MAJOR_PROGRESSION
    pattern = (0, 0, 7, 0, 0, 12, 7, 0)
    total_bars = int(np.ceil(duration_s / bar_s))

    for bar_index in range(total_bars):
        chord_offset = progression[bar_index % len(progression)][0]
        bass_root = root_midi - 12 + chord_offset
        for step in range(8):
            start = bar_index * bar_s + step * eighth_s
            if start >= duration_s:
                break
            midi = bass_root + pattern[step]
            note = _bass_note(midi, eighth_s * 0.92, sample_rate)
            _add_mono(out, note, start, sample_rate, gain=0.9)
    return out


def _pad_note(
    midi: float, length_s: float, sample_rate: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    n = max(1, int(round(length_s * sample_rate)))
    t = np.arange(n, dtype=np.float64) / sample_rate
    frequency = midi_to_hz(midi)
    env = _adsr(
        n,
        sample_rate,
        attack_s=0.35,
        decay_s=0.4,
        sustain=0.85,
        release_s=min(0.5, max(0.1, length_s * 0.25)),
    )
    tremolo = 1.0 + 0.04 * np.sin(2.0 * np.pi * 0.13 * t + rng.uniform(0, 2 * np.pi))

    channels = []
    for detune_cents in (-3.0, 3.0):
        detuned = frequency * (2.0 ** (detune_cents / 1200.0))
        phase_offset = rng.uniform(0.0, 2.0 * np.pi)
        signal = (
            np.sin(2.0 * np.pi * detuned * t + phase_offset)
            + 0.16 * np.sin(4.0 * np.pi * detuned * t + 0.6 + phase_offset)
            + 0.07 * np.sin(6.0 * np.pi * detuned * t + 1.1 + phase_offset)
        )
        channels.append(signal * env * tremolo)
    return channels[0], channels[1]


def _other(
    *,
    bpm: float,
    duration_s: float,
    sample_rate: int,
    root_midi: int,
    mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    n = max(1, int(round(duration_s * sample_rate)))
    out = np.zeros((2, n), dtype=np.float64)
    beat_s = 60.0 / bpm
    bar_s = beat_s * 4.0
    progression = _MINOR_PROGRESSION if mode == "minor" else _MAJOR_PROGRESSION
    total_bars = int(np.ceil(duration_s / bar_s))

    for bar_index in range(total_bars):
        chord_offset, quality = progression[bar_index % len(progression)]
        chord_root = root_midi + chord_offset
        start = bar_index * bar_s
        length_s = min(bar_s + 0.35, duration_s - start)
        if length_s <= 0.05:
            continue
        for interval in _CHORD_INTERVALS[quality]:
            midi = chord_root + interval
            left, right = _pad_note(midi, length_s, sample_rate, rng)
            _add_stereo(out, left, right, start, sample_rate, gain=0.22)
    return out


def _vocal_note(
    midi: float,
    length_s: float,
    sample_rate: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = max(1, int(round(length_s * sample_rate)))
    t = np.arange(n, dtype=np.float64) / sample_rate
    frequency = midi_to_hz(midi)

    vibrato_start = np.clip(t / 0.18, 0.0, 1.0)
    vibrato = 0.18 * np.sin(2.0 * np.pi * 5.2 * t + rng.uniform(0, 2 * np.pi))
    instantaneous = frequency * (2.0 ** (vibrato * vibrato_start / 12.0))
    phase = 2.0 * np.pi * np.cumsum(instantaneous) / sample_rate

    tone = (
        np.sin(phase)
        + 0.36 * np.sin(2.0 * phase + 0.2)
        + 0.14 * np.sin(3.0 * phase + 0.7)
        + 0.05 * np.sin(5.0 * phase + 1.2)
    )
    env = _adsr(
        n,
        sample_rate,
        attack_s=0.035,
        decay_s=0.10,
        sustain=0.84,
        release_s=min(0.16, max(0.06, length_s * 0.25)),
    )
    breath = np.diff(rng.standard_normal(n), prepend=0.0)
    breath = breath / max(float(np.max(np.abs(breath))), 1e-9)
    signal = (tone + 0.018 * breath) * env
    return signal


def _vocals(
    *,
    bpm: float,
    duration_s: float,
    sample_rate: int,
    root_midi: int,
    mode: str,
    rng: np.random.Generator,
    lead_in_s: float,
    lead_out_s: float,
) -> np.ndarray:
    n = max(1, int(round(duration_s * sample_rate)))
    out = np.zeros((2, n), dtype=np.float64)
    beat_s = 60.0 / bpm
    scale = _SCALE_MINOR if mode == "minor" else _SCALE_MAJOR
    phrase_bars = 4
    phrase_beats = 8.0
    total_bars = int(np.ceil(duration_s / (beat_s * 4.0)))
    usable_end = duration_s - lead_out_s
    durations = (0.5, 0.5, 1.0, 1.0, 1.5, 2.0)

    for phrase_start_bar in range(0, total_bars, phrase_bars):
        phrase_start = phrase_start_bar * 4.0 * beat_s
        phrase_end = min(phrase_start + phrase_beats * beat_s, usable_end)
        cursor = max(phrase_start, lead_in_s)
        if cursor >= phrase_end:
            continue

        degree = int(rng.integers(0, 5))
        events = 0
        while cursor < phrase_end - 0.05 and events < 24:
            remaining = phrase_end - cursor
            if rng.random() < 0.18 and remaining > 0.5 * beat_s:
                cursor += 0.5 * beat_s
                continue

            duration_beats = float(rng.choice(durations))
            length_s = min(duration_beats * beat_s, remaining)
            if length_s < 0.12:
                break

            step = int(rng.integers(-2, 3))
            degree = int(np.clip(degree + step, 0, len(scale) - 1))
            midi = root_midi + 12 + scale[degree]
            note = _vocal_note(midi, length_s * 0.92, sample_rate, rng)
            _add_mono(
                out,
                note,
                cursor,
                sample_rate,
                gain=float(rng.uniform(0.82, 1.0)),
            )
            cursor += length_s + min(0.04, 0.08 * beat_s)
            events += 1
    return out


def generate_synthetic_track(
    *,
    bpm: float = 120.0,
    duration_s: float = 16.0,
    sample_rate: int = 44100,
    channels: int = 2,
    seed: int = 1,
    root_midi: int = 57,
    mode: str = "minor",
    vocal_lead_in_s: float = 0.4,
    vocal_lead_out_s: float = 0.4,
) -> SyntheticTrack:
    """Generate a reproducible synthetic track with four stems."""

    if not 40.0 <= float(bpm) <= 240.0:
        raise ValueError(f"bpm must be between 40 and 240, got {bpm}")
    if float(duration_s) < 2.0:
        raise ValueError(f"duration_s must be at least 2 seconds, got {duration_s}")
    if int(sample_rate) < 8000:
        raise ValueError(f"sample_rate must be at least 8000, got {sample_rate}")
    if int(channels) not in (1, 2):
        raise ValueError(f"channels must be 1 or 2, got {channels}")
    if not 0 <= int(root_midi) <= 127:
        raise ValueError(f"root_midi must be within 0..127, got {root_midi}")
    if mode not in ("minor", "major"):
        raise ValueError(f"mode must be 'minor' or 'major', got {mode!r}")
    if vocal_lead_in_s < 0 or vocal_lead_out_s < 0:
        raise ValueError("vocal lead-in/out times must be non-negative")
    if vocal_lead_in_s + vocal_lead_out_s >= float(duration_s):
        raise ValueError("vocal lead-in plus lead-out must be shorter than the track")

    duration = float(duration_s)
    sample_rate = int(sample_rate)
    channels = int(channels)

    stems: dict[str, np.ndarray] = {
        "drums": _drums(
            bpm=float(bpm),
            duration_s=duration,
            sample_rate=sample_rate,
            rng=np.random.default_rng(seed + 1),
        ),
        "bass": _bass(
            bpm=float(bpm),
            duration_s=duration,
            sample_rate=sample_rate,
            root_midi=int(root_midi),
            mode=mode,
        ),
        "other": _other(
            bpm=float(bpm),
            duration_s=duration,
            sample_rate=sample_rate,
            root_midi=int(root_midi),
            mode=mode,
            rng=np.random.default_rng(seed + 2),
        ),
        "vocals": _vocals(
            bpm=float(bpm),
            duration_s=duration,
            sample_rate=sample_rate,
            root_midi=int(root_midi),
            mode=mode,
            rng=np.random.default_rng(seed + 3),
            lead_in_s=float(vocal_lead_in_s),
            lead_out_s=float(vocal_lead_out_s),
        ),
    }

    for name, samples in stems.items():
        samples -= np.mean(samples, axis=1, keepdims=True)
        _edge_fade(samples, sample_rate, 0.015)
        peak = float(np.max(np.abs(samples)))
        if peak > 0:
            samples *= _STEM_TARGET_PEAKS[name] / peak

    mix = np.sum(np.stack([stems[name] for name in stems], axis=0), axis=0)
    mix_peak = float(np.max(np.abs(mix)))
    if mix_peak > 0:
        master_scale = 0.65 / mix_peak
        for name in stems:
            stems[name] *= master_scale
        mix *= master_scale

    if channels == 1:
        stems = {
            name: np.mean(samples, axis=0, keepdims=True)
            for name, samples in stems.items()
        }
        mix = np.sum(np.stack([stems[name] for name in stems], axis=0), axis=0)

    mix_buffer = AudioBuffer(np.ascontiguousarray(mix), sample_rate)
    stem_set = StemSet(
        {
            name: AudioBuffer(np.ascontiguousarray(samples), sample_rate)
            for name, samples in stems.items()
        },
        source_label=f"synthetic-seed-{seed}",
    )
    return SyntheticTrack(
        mix=mix_buffer,
        stems=stem_set,
        bpm=float(bpm),
        duration_s=duration,
        sample_rate=sample_rate,
        channels=channels,
        seed=int(seed),
        root_midi=int(root_midi),
        mode=mode,
        vocal_lead_in_s=float(vocal_lead_in_s),
        vocal_lead_out_s=float(vocal_lead_out_s),
    )

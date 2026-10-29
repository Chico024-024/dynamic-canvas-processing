"""Offline per-stem effects.

The initial Core effect is a time-varying low-pass sweep implemented as two
cascaded one-pole filters. Cutoff changes are evaluated in short chunks, which
keeps the filter stable while avoiding a per-sample Python loop.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import lfilter

from ..config import DEFAULT_CONFIG
from ..models import AudioBuffer, PlanValidationError
from ..transitions.plan import EffectSpec

_MIN_CUTOFF_HZ = 20.0


def _cutoff_for_index(
    index: int,
    start_index: int,
    end_index: int,
    start_hz: float,
    end_hz: float,
) -> float:
    if index <= start_index:
        return start_hz
    if index >= end_index:
        return end_hz
    progress = (index - start_index) / float(end_index - start_index)
    return float(start_hz * ((end_hz / start_hz) ** progress))


def apply_lowpass_sweep(
    audio: AudioBuffer,
    effect: EffectSpec,
    *,
    transition_start_s: float,
    transition_duration_s: float,
    chunk_samples: int | None = None,
) -> AudioBuffer:
    """Apply a low-pass sweep to a copy of *audio*.

    ``transition_start_s`` is the position of the transition start within this
    audio buffer's own timeline. Effect times are relative to that transition
    start.
    """

    if effect.kind != "lowpass_sweep":
        raise PlanValidationError(f"unsupported effect kind {effect.kind!r}")
    if audio.sample_rate <= 0:
        raise PlanValidationError("audio sample rate must be positive")
    if not math.isfinite(effect.start_hz) or not math.isfinite(effect.end_hz):
        raise PlanValidationError("effect cutoffs must be finite")
    if effect.start_hz <= _MIN_CUTOFF_HZ or effect.end_hz <= _MIN_CUTOFF_HZ:
        raise PlanValidationError(
            f"effect cutoffs must be greater than {_MIN_CUTOFF_HZ:.0f} Hz"
        )

    nyquist = audio.sample_rate / 2.0
    if effect.start_hz >= nyquist or effect.end_hz >= nyquist:
        raise PlanValidationError(
            f"effect cutoff must stay below Nyquist ({nyquist:.0f} Hz) for "
            f"{audio.sample_rate} Hz audio"
        )

    end_relative_s = (
        transition_duration_s if effect.end_s is None else float(effect.end_s)
    )
    start_local_s = float(transition_start_s) + float(effect.start_s)
    end_local_s = float(transition_start_s) + end_relative_s

    sample_rate = audio.sample_rate
    frames = audio.frames
    start_index = int(round(start_local_s * sample_rate))
    end_index = int(round(end_local_s * sample_rate))
    start_index = max(0, min(frames, start_index))
    end_index = max(0, min(frames, end_index))
    if start_index >= frames or end_index <= start_index:
        return AudioBuffer(audio.samples.copy(), audio.sample_rate)

    chunk = int(chunk_samples or DEFAULT_CONFIG.render.effect_chunk_samples)
    chunk = max(32, chunk)
    start_hz = min(float(effect.start_hz), nyquist * 0.98)
    end_hz = min(float(effect.end_hz), nyquist * 0.98)

    output = audio.samples.copy()
    for channel in range(audio.channels):
        source = audio.samples[channel]
        processed = np.empty(frames - start_index, dtype=np.float64)
        previous_stage_one = float(source[start_index - 1]) if start_index > 0 else 0.0
        previous_stage_two = previous_stage_one
        write = 0
        position = start_index

        while position < frames:
            stop = min(position + chunk, frames)
            cutoff = _cutoff_for_index(
                (position + stop) // 2,
                start_index,
                end_index,
                start_hz,
                end_hz,
            )
            cutoff = float(np.clip(cutoff, _MIN_CUTOFF_HZ, nyquist * 0.98))
            coefficient = 1.0 - math.exp(-2.0 * math.pi * cutoff / sample_rate)
            numerator = [coefficient]
            denominator = [1.0, -(1.0 - coefficient)]

            state_one = np.array([(1.0 - coefficient) * previous_stage_one])
            state_two = np.array([(1.0 - coefficient) * previous_stage_two])
            stage_one, state_one = lfilter(
                numerator,
                denominator,
                source[position:stop],
                zi=state_one,
            )
            stage_two, state_two = lfilter(
                numerator,
                denominator,
                stage_one,
                zi=state_two,
            )
            previous_stage_one = float(stage_one[-1])
            previous_stage_two = float(stage_two[-1])
            count = stop - position
            processed[write : write + count] = stage_two
            write += count
            position = stop

        output[channel, start_index:] = processed

    return AudioBuffer(output, audio.sample_rate)

"""Conventional equal-power crossfade baseline."""

from __future__ import annotations

import math

import numpy as np

from ..audio_io import match_format
from ..config import DEFAULT_CONFIG, DCPConfig
from ..models import AudioBuffer, RenderError
from ..transitions.curves import equal_power_fades
from .common import RenderResult, add_unique_note, finalize_output


def render_crossfade(
    song_a: AudioBuffer,
    song_b: AudioBuffer,
    *,
    overlap_s: float,
    config: DCPConfig | None = None,
) -> RenderResult:
    """Render a conventional equal-power crossfade of two full mixes.

    Song B starts ``overlap_s`` before Song A ends. Song B is adapted to Song
    A's sample rate and channel layout; the adaptation is reported in notes.
    """

    cfg = config or DEFAULT_CONFIG
    if not math.isfinite(overlap_s) or overlap_s <= 0:
        raise RenderError(
            f"overlap_s must be a positive finite number, got {overlap_s}"
        )
    maximum = min(song_a.duration_s, song_b.duration_s)
    if overlap_s > maximum + 1e-9:
        raise RenderError(
            f"overlap {overlap_s:.3f}s exceeds the shorter track duration "
            f"({maximum:.3f}s)"
        )

    adapted_b, format_notes = match_format(song_b, song_a)
    sample_rate = song_a.sample_rate
    channels = song_a.channels

    overlap_frames = int(round(overlap_s * sample_rate))
    overlap_frames = max(1, min(overlap_frames, song_a.frames, adapted_b.frames))
    start_frame = song_a.frames - overlap_frames
    total_frames = max(song_a.frames, start_frame + adapted_b.frames)

    output = np.zeros((channels, total_frames), dtype=np.float64)
    output[:, :start_frame] = song_a.samples[:, :start_frame]

    fade_out, fade_in = equal_power_fades(overlap_frames)
    output[:, start_frame : start_frame + overlap_frames] = (
        song_a.samples[:, start_frame : start_frame + overlap_frames] * fade_out
        + adapted_b.samples[:, :overlap_frames] * fade_in
    )

    tail_start = start_frame + overlap_frames
    if tail_start < total_frames:
        b_tail = adapted_b.samples[:, overlap_frames:]
        count = min(b_tail.shape[1], total_frames - tail_start)
        if count > 0:
            output[:, tail_start : tail_start + count] += b_tail[:, :count]

    notes: list[str] = [
        f"Equal-power crossfade over {overlap_frames / sample_rate:.3f}s "
        "using the full mixes."
    ]
    for note in format_notes:
        add_unique_note(notes, f"Song B: {note}.")
    return finalize_output(
        output,
        sample_rate,
        config=cfg.render,
        notes=notes,
        label="crossfade output",
    )

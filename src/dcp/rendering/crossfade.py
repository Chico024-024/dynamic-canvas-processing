"""Conventional equal-power crossfade baseline."""

from __future__ import annotations

import math

import numpy as np

from ..audio_io import match_format
from ..config import DEFAULT_CONFIG, DCPConfig
from ..models import AudioBuffer, RenderError
from ..transitions.curves import crossfade_envelopes, equal_power_fades
from .common import RenderResult, add_unique_note, finalize_output


def _trim_track_edges(
    song_a: AudioBuffer,
    song_b: AudioBuffer,
    overlap_s: float,
    threshold_dbfs: float = -42.0,
) -> tuple[AudioBuffer, AudioBuffer, list[str]]:
    """Optionally trim trailing silence from Song A and leading silence from Song B."""
    notes: list[str] = []
    thresh_lin = 10.0 ** (threshold_dbfs / 20.0)

    # Trim trailing silence on Song A
    mono_a = song_a.mono()
    hop_a = max(1, int(0.05 * song_a.sample_rate))
    n_hops_a = len(mono_a) // hop_a
    if n_hops_a > 2:
        end_frame = song_a.frames
        for i in range(n_hops_a - 1, -1, -1):
            chunk = mono_a[i * hop_a : (i + 1) * hop_a]
            if float(np.sqrt(np.mean(chunk * chunk))) >= thresh_lin:
                end_frame = min(song_a.frames, (i + 1) * hop_a + int(0.25 * song_a.sample_rate))
                break
        min_keep = int(round((overlap_s + 0.5) * song_a.sample_rate))
        if min_keep < end_frame < song_a.frames - int(0.5 * song_a.sample_rate):
            trimmed_s = (song_a.frames - end_frame) / song_a.sample_rate
            song_a = AudioBuffer(np.ascontiguousarray(song_a.samples[:, :end_frame]), song_a.sample_rate)
            notes.append(f"Song A: trimmed {trimmed_s:.2f}s of trailing silence.")

    # Trim leading silence on Song B
    mono_b = song_b.mono()
    hop_b = max(1, int(0.05 * song_b.sample_rate))
    n_hops_b = len(mono_b) // hop_b
    if n_hops_b > 2:
        start_frame = 0
        for i in range(n_hops_b):
            chunk = mono_b[i * hop_b : (i + 1) * hop_b]
            if float(np.sqrt(np.mean(chunk * chunk))) >= thresh_lin:
                start_frame = max(0, i * hop_b - int(0.05 * song_b.sample_rate))
                break
        min_keep_b = int(round((overlap_s + 0.5) * song_b.sample_rate))
        if start_frame > int(0.25 * song_b.sample_rate) and (song_b.frames - start_frame) > min_keep_b:
            trimmed_s = start_frame / song_b.sample_rate
            song_b = AudioBuffer(np.ascontiguousarray(song_b.samples[:, start_frame:]), song_b.sample_rate)
            notes.append(f"Song B: trimmed {trimmed_s:.2f}s of leading silence.")

    return song_a, song_b, notes


def render_crossfade(
    song_a: AudioBuffer,
    song_b: AudioBuffer,
    *,
    overlap_s: float | None = None,
    curve: str | None = None,
    trim_silence: bool | None = None,
    config: DCPConfig | None = None,
) -> RenderResult:
    """Render a conventional full-mix volume crossfade of two tracks.

    Song B starts ``overlap_s`` before Song A ends. Song B is adapted to Song
    A's sample rate and channel layout; the adaptation is reported in notes.
    Supports ``curve`` in ``("equal_power", "s_curve", "linear")`` and optional
    edge silence trimming while remaining strictly separate from Automix and DCP.
    """

    cfg = config or DEFAULT_CONFIG
    effective_overlap = cfg.crossfade.default_overlap_s if overlap_s is None else overlap_s
    effective_curve = (curve or cfg.crossfade.curve or "equal_power").strip().lower()
    effective_trim = cfg.crossfade.trim_silence if trim_silence is None else bool(trim_silence)

    if not math.isfinite(effective_overlap) or effective_overlap <= 0:
        raise RenderError(
            f"overlap_s must be a positive finite number, got {effective_overlap}"
        )

    trim_notes: list[str] = []
    if effective_trim:
        song_a, song_b, trim_notes = _trim_track_edges(
            song_a,
            song_b,
            effective_overlap,
            threshold_dbfs=cfg.crossfade.silence_threshold_dbfs,
        )

    maximum = min(song_a.duration_s, song_b.duration_s)
    if effective_overlap > maximum + 1e-9:
        raise RenderError(
            f"overlap {effective_overlap:.3f}s exceeds the shorter track duration "
            f"({maximum:.3f}s)"
        )

    adapted_b, format_notes = match_format(song_b, song_a)
    sample_rate = song_a.sample_rate
    channels = song_a.channels

    overlap_frames = int(round(effective_overlap * sample_rate))
    overlap_frames = max(1, min(overlap_frames, song_a.frames, adapted_b.frames))
    start_frame = song_a.frames - overlap_frames
    total_frames = max(song_a.frames, start_frame + adapted_b.frames)

    output = np.zeros((channels, total_frames), dtype=np.float64)
    output[:, :start_frame] = song_a.samples[:, :start_frame]

    try:
        fade_out, fade_in = crossfade_envelopes(overlap_frames, curve=effective_curve)
    except ValueError as exc:
        raise RenderError(str(exc)) from exc

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

    curve_label = {
        "equal_power": "Equal-power crossfade",
        "s_curve": "S-curve constant-power crossfade",
        "smoothstep": "S-curve constant-power crossfade",
        "scurve": "S-curve constant-power crossfade",
        "linear": "Linear amplitude crossfade",
    }.get(effective_curve, f"Crossfade ({effective_curve})")

    notes: list[str] = [
        f"{curve_label} over {overlap_frames / sample_rate:.3f}s "
        "using the full mixes."
    ]
    for note in trim_notes:
        add_unique_note(notes, note)
    for note in format_notes:
        add_unique_note(notes, f"Song B: {note}.")
    return finalize_output(
        output,
        sample_rate,
        config=cfg.render,
        notes=notes,
        label="crossfade output",
    )

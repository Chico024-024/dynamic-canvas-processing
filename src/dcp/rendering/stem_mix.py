"""Offline rendering of a validated transition plan."""

from __future__ import annotations

import numpy as np

from ..audio_io import match_format
from ..config import DEFAULT_CONFIG, DCPConfig
from ..models import (
    MIX_STEM,
    AudioBuffer,
    PlanValidationError,
    RenderError,
    StemSet,
)
from ..transitions.plan import StemAutomation, TransitionPlan, validate_plan
from ..transitions.curves import interpolate_gain
from .common import RenderResult, add_unique_note, finalize_output
from .effects import apply_lowpass_sweep


def rendered_duration_s(
    song_a: AudioBuffer, song_b: AudioBuffer, plan: TransitionPlan
) -> float:
    """Duration of the output timeline for a plan."""
    cue_s = float(plan.metadata.get("cue_offset_b_s", 0.0) or 0.0)
    remaining_b = max(0.0, song_b.duration_s - max(0.0, cue_s))
    return max(song_a.duration_s, plan.start_time_s + remaining_b)


def render_plan(
    song_a: AudioBuffer,
    song_b: AudioBuffer,
    plan: TransitionPlan,
    *,
    stems_a: StemSet | None = None,
    stems_b: StemSet | None = None,
    config: DCPConfig | None = None,
) -> RenderResult:
    """Render a transition plan from full mixes and/or user-provided stems."""

    cfg = config or DEFAULT_CONFIG
    validate_plan(plan, cfg, sample_rate=song_a.sample_rate)

    tolerance = 1.0 / song_a.sample_rate
    if plan.start_time_s > song_a.duration_s + tolerance:
        raise RenderError(
            f"transition starts at {plan.start_time_s:.3f}s, after Song A ends "
            f"at {song_a.duration_s:.3f}s"
        )
    if plan.end_time_s > song_a.duration_s + tolerance:
        raise RenderError(
            f"transition ends at {plan.end_time_s:.3f}s, after Song A ends at "
            f"{song_a.duration_s:.3f}s; shorten the transition"
        )

    sample_rate = song_a.sample_rate
    cue_offset_b_s = float(plan.metadata.get("cue_offset_b_s", 0.0) or 0.0)
    if cue_offset_b_s < -1e-6:
        raise RenderError(
            f"Song B cue offset ({cue_offset_b_s:.3f}s) cannot be negative"
        )
    cue_start_frames = int(round(max(0.0, cue_offset_b_s) * sample_rate))

    # Determine maximum available frames for Song B across mix and all stems
    b_max_frames = song_b.frames
    if stems_b is not None:
        b_max_frames = max(b_max_frames, *(buf.frames for buf in stems_b.stems.values()))

    remaining_b_frames = max(0, b_max_frames - cue_start_frames)
    total_frames = max(
        song_a.frames,
        int(round(plan.start_time_s * sample_rate)) + remaining_b_frames,
    )
    output = np.zeros((song_a.channels, total_frames), dtype=np.float64)
    notes: list[str] = []

    sources: dict[str, tuple[AudioBuffer, StemSet | None]] = {
        "a": (song_a, stems_a),
        "b": (song_b, stems_b),
    }
    by_source: dict[str, list[StemAutomation]] = {"a": [], "b": []}
    for automation in plan.automations:
        by_source[automation.source].append(automation)

    for source, automations in by_source.items():
        mix, stems = sources[source]
        used_stems = {
            automation.stem
            for automation in automations
            if automation.stem != MIX_STEM
        }
        uses_mix = any(automation.stem == MIX_STEM for automation in automations)
        if used_stems and stems is None:
            raise RenderError(
                f"plan uses stems {sorted(used_stems)} for Song "
                f"{source.upper()}, but no stems were provided"
            )
        if stems is not None and used_stems:
            missing = [name for name in stems.names if name not in used_stems]
            if missing:
                raise RenderError(
                    f"plan does not cover provided stem(s) {missing} for Song "
                    f"{source.upper()}; every provided stem must be automated "
                    "or the plan must use the full mix"
                )
        if stems is not None and uses_mix and not used_stems:
            add_unique_note(
                notes,
                f"Song {source.upper()}: stems were provided but the plan uses "
                "the full mix, so those stems were not rendered.",
            )

    cache: dict[tuple[str, str], tuple[AudioBuffer, tuple[str, ...]]] = {}

    def resolve(source: str, stem: str) -> AudioBuffer:
        key = (source, stem)
        if key not in cache:
            mix, stems = sources[source]
            if stem == MIX_STEM:
                base = mix
            else:
                if stems is None:
                    raise RenderError(
                        f"plan references stem {stem!r} for Song {source.upper()}, "
                        "but no stems were provided"
                    )
                if stem not in stems:
                    raise RenderError(
                        f"stem {stem!r} was requested for Song {source.upper()} "
                        f"but only {list(stems.names)} are available"
                    )
                base = stems[stem]
            converted, format_notes = match_format(base, song_a)
            labelled = tuple(
                f"Song {source.upper()}"
                + (f" stem '{stem}'" if stem != MIX_STEM else "")
                + f": {note}."
                for note in format_notes
            )
            cache[key] = (converted, labelled)
        return cache[key][0]

    for automation in plan.automations:
        audio = resolve(automation.source, automation.stem)
        for note in cache[(automation.source, automation.stem)][1]:
            add_unique_note(notes, note)

        # For Song B, apply phase lock cue offset if available
        # This slices Song B's audio starting from cue_offset_b_s seconds,
        # so its anchor downbeat aligns with Song A's at the transition start
        if automation.source == "a":
            offset_s = 0.0
            cue_start_s = 0.0
            cue_start_frames = 0
            audio_to_process = audio.samples
        else:
            offset_s = plan.start_time_s
            cue_start_s = cue_offset_b_s
            cue_start_frames = int(round(cue_start_s * sample_rate))
            if cue_start_frames >= audio.frames:
                raise RenderError(
                    f"Song B cue offset ({cue_start_s:.3f}s) is beyond track duration "
                    f"({audio.duration_s:.3f}s) for stem '{automation.stem}'"
                )
            transition_frames = int(round(plan.duration_s * sample_rate))
            remaining_frames = audio.frames - cue_start_frames
            # Tolerance of 1 frame for floating-point rounding
            if remaining_frames + 1 < transition_frames:
                remaining_s = remaining_frames / sample_rate
                raise RenderError(
                    f"Song B stem '{automation.stem}' has only {remaining_s:.3f}s remaining "
                    f"after cue offset {cue_start_s:.3f}s, which cannot cover the planned "
                    f"transition duration ({plan.duration_s:.3f}s)"
                )
            audio_to_process = audio.samples[:, cue_start_frames:]

        offset_frames = int(round(offset_s * sample_rate))
        transition_start_in_audio_s = plan.start_time_s - offset_s

        processed = audio_to_process
        for effect in automation.effects:
            processed = apply_lowpass_sweep(
                AudioBuffer(processed, audio.sample_rate),
                effect,
                transition_start_s=transition_start_in_audio_s,
                transition_duration_s=plan.duration_s,
                chunk_samples=cfg.render.effect_chunk_samples,
            ).samples

        # Time axis for gain interpolation should be relative to the sliced audio
        local_times = np.arange(audio_to_process.shape[1], dtype=np.float64) / sample_rate
        relative_times = local_times - transition_start_in_audio_s
        gains = interpolate_gain(
            automation.points,
            relative_times,
            curve=automation.curve,
            floor_db=cfg.transition.fade_floor_db,
        )

        available = min(audio_to_process.shape[1], total_frames - offset_frames)
        if available <= 0:
            continue
        output[
            :, offset_frames : offset_frames + available
        ] += processed[:, :available] * gains[np.newaxis, :available]

    add_unique_note(
        notes,
        f"Rendered {len(plan.automations)} stem automation(s) with strategy "
        f"'{plan.strategy}'.",
    )
    if any(automation.stem != MIX_STEM for automation in plan.automations):
        add_unique_note(
            notes,
            "Stem-level rendering uses the provided stems; the full mixes are "
            "not mixed in for sources driven by stems.",
        )

    return finalize_output(
        output,
        sample_rate,
        config=cfg.render,
        notes=notes,
        label="transition output",
    )

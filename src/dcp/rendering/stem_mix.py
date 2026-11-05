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
from ..transitions.plan import EffectSpec, StemAutomation, TransitionPlan, validate_plan
from ..transitions.curves import interpolate_gain
from .common import RenderResult, add_unique_note, finalize_output
from .effects import apply_lowpass_sweep
from .dsp_fallback import (
    process_zdf_svf_sweep_numpy,
    process_sidechain_compressor_numpy,
    process_fdn_freeze_numpy,
    time_stretch_phase_vocoder_numpy,
    process_master_bus_glue_numpy,
)
from .song_c import (
    generate_noise_riser,
    generate_reverse_crash,
    generate_sub_bass_808,
)
from .vocal_chop import generate_vocal_chops
from .drum_regroove import regroove_drums_half_time


def apply_effect_spec_to_stem(
    stem_audio: np.ndarray,
    effect: Any,
    sr: int,
    bpm: float = 126.0,
    sidechain_key_audio: np.ndarray | None = None,
    transition_start_s: float = 0.0,
    transition_duration_s: float | None = None,
) -> np.ndarray:
    """
    Slices [start_sample : end_sample] from `stem_audio` and executes the EffectSpec.
    """
    n_total = stem_audio.shape[-1]
    eff_end = transition_duration_s if getattr(effect, "end_s", None) is None else float(effect.end_s)
    s_sec = float(transition_start_s) + float(getattr(effect, "start_s", 0.0))
    e_sec = float(transition_start_s) + eff_end

    s_idx = int(np.clip(round(s_sec * sr), 0, n_total))
    e_idx = int(np.clip(round(e_sec * sr), s_idx, n_total))
    if e_idx <= s_idx:
        return stem_audio

    out = stem_audio.copy()
    segment = out[..., s_idx:e_idx]
    kind = getattr(effect, "kind", getattr(effect, "effect_type", "")).lower()
    params = getattr(effect, "params", {}) or {}

    if kind in (
        "hpf_sweep", "lpf_sweep", "bpf_sweep", "zdf_sweep",
        "zdf_highpass_sweep", "zdf_lowpass_sweep", "zdf_bandpass_sweep",
        "zdf_highpass", "zdf_lowpass", "zdf_bandpass",
        "highpass_sweep", "bandpass_sweep",
    ):
        if "highpass" in kind or "hpf" in kind:
            mode = "hpf"
        elif "bandpass" in kind or "bpf" in kind:
            mode = "bpf"
        elif "lowpass" in kind or "lpf" in kind:
            mode = "lpf"
        else:
            mode = params.get("mode", "hpf")

        f_start = float(getattr(effect, "start_hz", params.get("start_hz", 30.0)))
        f_end = float(getattr(effect, "end_hz", params.get("end_hz", 800.0)))
        q_val = float(params.get("resonance", params.get("resonance_q", 1.0)))

        out[..., s_idx:e_idx] = process_zdf_svf_sweep_numpy(
            segment, sr=sr, start_cutoff_hz=f_start, end_cutoff_hz=f_end, resonance_q=q_val, mode=mode
        )

    elif kind in ("reverb_freeze", "fdn_freeze", "spatial_freeze"):
        freeze_on_s = params.get("freeze_engage_s", params.get("freeze_start_s", getattr(effect, "start_s", 0.0)))
        freeze_off_s = params.get("freeze_release_s", params.get("freeze_end_s", eff_end))
        freeze_mode = bool(params.get("freeze_mode", True))

        if freeze_mode and freeze_on_s is not None:
            local_f_start = max(0, int(round((transition_start_s + float(freeze_on_s) - s_sec) * sr)))
        else:
            local_f_start = None

        if freeze_mode and freeze_off_s is not None:
            local_f_end = max(0, int(round((transition_start_s + float(freeze_off_s) - s_sec) * sr)))
        else:
            local_f_end = None

        feedback = float(params.get("feedback", 0.88))
        wet_mix = float(params.get("wet_mix_start", params.get("wet", 0.55)))
        dry_mix = float(params.get("dry", 1.0))

        out[..., s_idx:e_idx] = process_fdn_freeze_numpy(
            segment,
            sr=sr,
            freeze_start_sample=local_f_start,
            freeze_end_sample=local_f_end,
            feedback_gain=feedback,
            wet_mix=wet_mix,
            dry_mix=dry_mix,
        )

    elif kind == "sidechain_duck" and sidechain_key_audio is not None:
        key_total = sidechain_key_audio.shape[-1]
        k_start = min(s_idx, key_total)
        k_end = min(e_idx, key_total)
        if k_end > k_start:
            key_seg = sidechain_key_audio[..., k_start:k_end]
            if key_seg.shape[-1] < segment.shape[-1]:
                pad_len = segment.shape[-1] - key_seg.shape[-1]
                key_seg = np.pad(key_seg, ((0, 0), (0, pad_len)) if key_seg.ndim == 2 else (0, pad_len))

            out[..., s_idx:e_idx] = process_sidechain_compressor_numpy(
                segment,
                key_seg,
                sr=sr,
                bpm=bpm,
                attack_ms=float(params.get("attack_ms", 1.5)),
                release_beat_fraction=float(params.get("release_beats", params.get("release_beat_fraction", 0.25))),
                threshold_db=float(params.get("threshold_db", -18.0)),
                ratio=float(params.get("ratio", 4.0)),
            )

    elif kind == "stereo_width":
        w_start = float(params.get("width_start", 1.0))
        w_end = float(params.get("width_end", 1.8))
        if segment.shape[0] == 2:
            seg_len = segment.shape[1]
            w_traj = np.linspace(w_start, w_end, seg_len, dtype=np.float32)
            mid = 0.5 * (segment[0] + segment[1])
            side = 0.5 * (segment[0] - segment[1])
            out[0, s_idx:e_idx] = mid + side * w_traj
            out[1, s_idx:e_idx] = mid - side * w_traj

    elif kind == "dub_delay_send":
        delay_beats = float(params.get("delay_note", 0.75))
        feedback = float(params.get("feedback", 0.45))
        send_lvl = float(np.power(10.0, float(params.get("send_level", -6.0)) / 20.0))
        delay_sec = delay_beats * (60.0 / max(40.0, bpm))
        delay_samples = int(round(delay_sec * sr))
        if delay_samples > 0 and delay_samples < segment.shape[-1]:
            delayed = np.zeros_like(segment)
            seg_len = segment.shape[-1]
            for ch in range(segment.shape[0]):
                in_ch = segment[ch] * send_lvl
                out_ch = np.zeros(seg_len, dtype=np.float32)
                for i in range(delay_samples, seg_len):
                    out_ch[i] = in_ch[i - delay_samples] + feedback * out_ch[i - delay_samples]
                delayed[ch] = out_ch
            out[..., s_idx:e_idx] = np.clip(segment + delayed, -2.0, 2.0)

    return out


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
    if isinstance(stems_a, dict):
        stems_a = StemSet(stems=stems_a, source_label="A")
    if isinstance(stems_b, dict):
        stems_b = StemSet(stems=stems_b, source_label="B")

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

    # List to collect wet freeze ghost tails for post-glue re-addition
    ghost_tails: list[tuple[np.ndarray, int, int]] = []

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
            warp_ratio_b = float(plan.metadata.get("warp_ratio_b", 1.0) or 1.0)
            if abs(warp_ratio_b - 1.0) > 0.005:
                audio_to_process = time_stretch_phase_vocoder_numpy(audio_to_process, warp_ratio_b)

        offset_frames = int(round(offset_s * sample_rate))
        transition_start_in_audio_s = plan.start_time_s - offset_s

        # Resolve sidechain key audio (Song B drums or mix) if needed
        key_audio_samples = None
        has_duck = any(getattr(e, "kind", "").lower() == "sidechain_duck" for e in automation.effects)
        if has_duck and automation.source == "a":
            try:
                b_stems = sources["b"][1]
                if b_stems and "drums" in b_stems:
                    raw_key = resolve("b", "drums")
                else:
                    raw_key = resolve("b", MIX_STEM)
                cue_f = int(round(cue_offset_b_s * sample_rate))
                sliced_key = raw_key.samples[:, cue_f:]
                # Lowpass filter sidechain key to 150 Hz so hi-hats do not trigger ducking
                from scipy import signal
                sos_key = signal.butter(2, min(150.0, sample_rate * 0.45) / (0.5 * sample_rate), btype="lowpass", output="sos")
                sliced_key = signal.sosfilt(sos_key, sliced_key.astype(np.float64), axis=1).astype(np.float32)
                start_pad_frames = int(round(plan.start_time_s * sample_rate))
                key_audio_samples = np.pad(sliced_key, ((0, 0), (start_pad_frames, 0)))
            except Exception:
                key_audio_samples = None

        bpm_val = float(plan.metadata.get("tempo_b", plan.metadata.get("tempo_a", 126.0)) or 126.0)

        # Check if stem contains an FDN spatial freeze effect
        freeze_eff = next((e for e in automation.effects if getattr(e, "kind", "").lower() in ("spatial_freeze", "fdn_freeze", "reverb_freeze")), None)
        wet_ghost_tail = None
        if freeze_eff is not None:
            # Generate pure wet freeze reverberant ghost tail (dry=0.0)
            p_dict = dict(getattr(freeze_eff, "params", {}) or {})
            p_dict["dry"] = 0.0
            wet_freeze_spec = EffectSpec(
                kind=freeze_eff.kind,
                start_hz=freeze_eff.start_hz,
                end_hz=freeze_eff.end_hz,
                start_s=freeze_eff.start_s,
                end_s=freeze_eff.end_s,
                params=p_dict,
            )
            wet_ghost_tail = apply_effect_spec_to_stem(
                audio_to_process,
                wet_freeze_spec,
                sr=sample_rate,
                bpm=bpm_val,
                sidechain_key_audio=key_audio_samples,
                transition_start_s=transition_start_in_audio_s,
                transition_duration_s=plan.duration_s,
            )
            if wet_ghost_tail is not None:
                available = min(audio_to_process.shape[1], total_frames - offset_frames)
                ghost_tails.append((wet_ghost_tail, offset_frames, available))
            # Exclude the freeze effect from the dry stem so the dry path receives gain automation
            effs_to_run = [e for e in automation.effects if e != freeze_eff]
        else:
            effs_to_run = list(automation.effects)

        processed = audio_to_process
        for effect in effs_to_run:
            if effect.kind == "lowpass_sweep":
                processed = apply_lowpass_sweep(
                    AudioBuffer(processed, audio.sample_rate),
                    effect,
                    transition_start_s=transition_start_in_audio_s,
                    transition_duration_s=plan.duration_s,
                    chunk_samples=cfg.render.effect_chunk_samples,
                ).samples
            else:
                processed = apply_effect_spec_to_stem(
                    processed,
                    effect,
                    sr=sample_rate,
                    bpm=bpm_val,
                    sidechain_key_audio=key_audio_samples,
                    transition_start_s=transition_start_in_audio_s,
                    transition_duration_s=plan.duration_s,
                )

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

        # Add the isolated wet reverberant ghost tail to the mix
        if wet_ghost_tail is not None:
            w_avail = min(wet_ghost_tail.shape[1], total_frames - offset_frames)
            if w_avail > 0:
                output[:, offset_frames : offset_frames + w_avail] += wet_ghost_tail[:, :w_avail]

    # Inject bespoke synthesized transitional elements for Studio Remix (Song C)
    if plan.metadata.get("studio_remix"):
        t_impact_rel = float(plan.metadata.get("t_impact_s", plan.duration_s * (2.0 / 3.0)))
        t_intro_rel = float(plan.metadata.get("t_intro_s", t_impact_rel * 0.5))
        t_breath_rel = float(plan.metadata.get("t_breath_s", max(0.0, t_impact_rel - 0.25)))
        impact_frame = int(round((plan.start_time_s + t_impact_rel) * sample_rate))

        # 1. Reverse Crash Cymbal Swell into impact (BOOSTED)
        try:
            rev_crash = generate_reverse_crash(duration_s=min(1.5, t_impact_rel), sr=sample_rate, level_db=-6.0)
            crash_len = rev_crash.shape[1]
            crash_start = max(0, impact_frame - crash_len)
            crash_avail = min(crash_len, total_frames - crash_start)
            if crash_avail > 0:
                output[:, crash_start : crash_start + crash_avail] += rev_crash[:, :crash_avail]
        except Exception:
            pass

        # 2. Sub-Bass 808 Slam at impact downbeat (BOOSTED)
        try:
            root_pc = int(plan.metadata.get("bass_root_b", 2))
            sub_808 = generate_sub_bass_808(duration_s=1.8, sr=sample_rate, root_pitch_class=root_pc, level_db=-3.0)
            sub_len = sub_808.shape[1]
            sub_avail = min(sub_len, total_frames - impact_frame)
            if sub_avail > 0 and impact_frame >= 0:
                output[:, impact_frame : impact_frame + sub_avail] += sub_808[:, :sub_avail]
        except Exception:
            pass

        # 3. Noise Riser during build-up (BOOSTED)
        try:
            riser_dur = max(0.5, t_breath_rel - t_intro_rel)
            riser = generate_noise_riser(duration_s=riser_dur, sr=sample_rate, level_db=-8.0, sidechain_bpm=bpm_val)
            riser_start = int(round((plan.start_time_s + t_intro_rel) * sample_rate))
            riser_avail = min(riser.shape[1], total_frames - riser_start)
            if riser_avail > 0 and riser_start >= 0:
                output[:, riser_start : riser_start + riser_avail] += riser[:, :riser_avail]
        except Exception:
            pass

        # 4. Syncopated Vocal Chops from RESOLVING PHRASE ONLY (not full stem)
        try:
            a_stems = sources["a"][1]
            if a_stems and "vocals" in a_stems:
                raw_voc = a_stems["vocals"].samples
                # Use only the vocal tail from launchpad phrase to end
                # launchpad is at t_impact_rel in transition coords
                launchpad_frame = int(round((plan.start_time_s + t_impact_rel) * sample_rate))
                vocal_tail_start = max(0, launchpad_frame - int(0.5 * sample_rate))  # 0.5s before launchpad
                vocal_tail = raw_voc[:, vocal_tail_start:]
                res_dur = max(0.5, plan.duration_s - t_impact_rel)
                chops = generate_vocal_chops(vocal_tail, sr=sample_rate, bpm=bpm_val, duration_s=res_dur, level_db=-6.0)
                chops_avail = min(chops.shape[1], total_frames - impact_frame)
                if chops_avail > 0 and impact_frame >= 0:
                    output[:, impact_frame : impact_frame + chops_avail] += chops[:, :chops_avail]
        except Exception:
            pass

        # 5. Half-Time Drum Re-grooving pocket during post-impact bridge (BOOSTED + HPF)
        try:
            b_stems = sources["b"][1]
            if b_stems and "drums" in b_stems:
                raw_drums = b_stems["drums"].samples
                pocket_dur = min(8.0, max(1.0, plan.duration_s - t_impact_rel))
                regrooved = regroove_drums_half_time(raw_drums, sr=sample_rate, bpm=bpm_val, duration_s=pocket_dur)
                # High-pass at 100Hz to avoid kick clash with main drums
                from scipy import signal
                sos_hp = signal.butter(2, min(100.0, sample_rate * 0.45) / (0.5 * sample_rate), btype="highpass", output="sos")
                regrooved = signal.sosfilt(sos_hp, regrooved.astype(np.float64), axis=1).astype(np.float32)
                pocket_avail = min(regrooved.shape[1], total_frames - impact_frame)
                if pocket_avail > 0 and impact_frame >= 0:
                    output[:, impact_frame : impact_frame + pocket_avail] += regrooved[:, :pocket_avail] * 1.0
        except Exception:
            pass

        # 6. Apply Master Bus VCA Glue Compressor & Brickwall Limiter
        try:
            output = process_master_bus_glue_numpy(output, sr=sample_rate, threshold_db=-12.0, ratio=1.5, ceiling_dbfs=-0.5)
        except Exception:
            pass

        # 7. Post-Glue Ghost Bus: re-add wet freeze tails so they survive limiting
        for wet_tail, offset_f, avail in ghost_tails:
            if avail > 0:
                tail_avail = min(wet_tail.shape[1], avail)
                if tail_avail > 0:
                    output[:, offset_f : offset_f + tail_avail] += wet_tail[:, :tail_avail] * 0.7

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

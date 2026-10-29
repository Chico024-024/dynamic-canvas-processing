"""DCP command-line interface."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from dataclasses import replace
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Sequence

from . import __version__
from .advanced import build_planner_with_advanced, discover_advanced_strategies
from .analysis import (
    TrackAnalysis,
    analyze_track_with_backend,
)
from .audio_io import load_audio, load_stem_set, save_audio, save_stem_set
from .config import DCPConfig, DEFAULT_CONFIG
from .demo import run_demo
from .models import DCPError
from .rendering import render_crossfade, render_plan, rendered_duration_s
from .synthetic import generate_synthetic_track
from .transitions import TransitionPlan
from .validation import compare_audio, validate_audio


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, allow_nan=False))


def _write_json(path: str | Path, payload: object) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return target


def _version_of(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "not installed"


def _config_for_cli(args: argparse.Namespace) -> DCPConfig:
    prevent = not bool(getattr(args, "no_limit", False))
    render = replace(DEFAULT_CONFIG.render, prevent_clipping=prevent)
    return replace(DEFAULT_CONFIG, render=render)


def _describe_analysis(analysis: TrackAnalysis) -> str:
    tempo = (
        "unavailable"
        if analysis.tempo_bpm is None
        else f"{analysis.tempo_bpm:.2f} BPM "
        f"(confidence {analysis.tempo_confidence:.2f}"
        + (", ambiguous octave" if analysis.tempo_ambiguous else "")
        + ")"
    )
    lines = [
        f"Duration:       {analysis.duration_s:.3f} s",
        f"Sample rate:    {analysis.sample_rate} Hz",
        f"Channels:       {analysis.channels}",
        f"Peak:           {analysis.peak_dbfs:.2f} dBFS",
        f"RMS:            {analysis.rms_dbfs:.2f} dBFS",
        f"Silence ratio:  {analysis.silence_ratio:.1%}",
        f"Tempo estimate: {tempo}",
        f"Analyzer:       {analysis.analyzer}",
        f"Detected beats: {len(analysis.beat_times_s)}",
        f"Onsets:         {analysis.onset_count}",
    ]
    if analysis.notes:
        lines.append("Notes:")
        lines.extend(f"  - {note}" for note in analysis.notes)
    return "\n".join(lines)


def _format_plan(plan: TransitionPlan) -> str:
    lines = [
        f"Strategy:       {plan.strategy} "
        f"(requested: {plan.requested_strategy})",
        f"Transition:     {plan.duration_s:.3f}s at "
        f"{plan.start_time_s:.3f}s..{plan.end_time_s:.3f}s",
        f"Automations:    {len(plan.automations)}",
    ]
    for automation in plan.automations:
        points = " ".join(
            f"{point.time_s:.2f}s={point.gain_db:.1f}dB"
            for point in automation.points
        )
        line = f"  {automation.source}/{automation.stem} [{automation.curve}]: {points}"
        if automation.effects:
            effects = ", ".join(
                f"{effect.kind} {effect.start_hz:.0f}->{effect.end_hz:.0f}Hz"
                for effect in automation.effects
            )
            line += f" effects: {effects}"
        lines.append(line)
    if plan.reasoning:
        lines.append("Reasoning:")
        lines.extend(f"  - {reason}" for reason in plan.reasoning)
    if plan.warnings:
        lines.append("Warnings:")
        lines.extend(f"  - {warning}" for warning in plan.warnings)
    return "\n".join(lines)


def _cmd_analyze(args: argparse.Namespace) -> int:
    audio = load_audio(args.path)
    analysis = analyze_track_with_backend(audio, backend=args.analyzer)
    if args.json:
        _print_json({"file": str(Path(args.path)), **analysis.to_dict()})
        return 0
    print(f"File:           {Path(args.path)}")
    print(_describe_analysis(analysis))
    return 0


def _cmd_synthesize(args: argparse.Namespace) -> int:
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)

    track_a = generate_synthetic_track(
        bpm=args.bpm_a,
        duration_s=args.duration_a,
        sample_rate=args.sample_rate,
        channels=2,
        seed=args.seed,
        root_midi=57,
        mode="minor",
        vocal_lead_out_s=min(0.8, args.duration_a * 0.1),
    )
    track_b = generate_synthetic_track(
        bpm=args.bpm_b,
        duration_s=args.duration_b,
        sample_rate=args.sample_rate,
        channels=2,
        seed=args.seed + 1,
        root_midi=60,
        mode="major",
        vocal_lead_in_s=min(1.0, args.duration_b * 0.1),
    )

    song_a = save_audio(output / "song_a.wav", track_a.mix)
    song_b = save_audio(output / "song_b.wav", track_b.mix)
    stems_a = save_stem_set(output / "stems" / "a", track_a.stems)
    stems_b = save_stem_set(output / "stems" / "b", track_b.stems)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "engine_version": __version__,
        "a": {
            "mix": "song_a.wav",
            "stems_dir": "stems/a",
            **track_a.metadata(),
        },
        "b": {
            "mix": "song_b.wav",
            "stems_dir": "stems/b",
            **track_b.metadata(),
        },
        "files": {
            "song_a": str(song_a),
            "song_b": str(song_b),
            "stems_a": sorted(str(path) for path in stems_a.values()),
            "stems_b": sorted(str(path) for path in stems_b.values()),
        },
    }
    manifest_path = _write_json(output / "manifest.json", manifest)
    print(f"Generated synthetic songs in {output}")
    print(f"  Song A:  {song_a}")
    print(f"  Song B:  {song_b}")
    print(f"  Stems A: {output / 'stems' / 'a'}")
    print(f"  Stems B: {output / 'stems' / 'b'}")
    print(f"  Manifest: {manifest_path}")
    print("")
    print("Try:")
    print(
        f"  dcp crossfade {song_a} {song_b} {output / 'baseline.wav'} --overlap 8"
    )
    print(
        f"  dcp render {song_a} {song_b} {output / 'dcp.wav'} "
        f"--stems-a {output / 'stems' / 'a'} "
        f"--stems-b {output / 'stems' / 'b'} --strategy stem_level"
    )
    return 0


def _cmd_crossfade(args: argparse.Namespace) -> int:
    config = _config_for_cli(args)
    song_a = load_audio(args.song_a)
    song_b = load_audio(args.song_b)
    result = render_crossfade(
        song_a, song_b, overlap_s=args.overlap, config=config
    )
    save_audio(
        args.output,
        result.audio,
        subtype=config.render.wav_subtype,
        allow_clipping=not config.render.prevent_clipping,
    )
    expected = song_a.duration_s + song_b.duration_s - args.overlap
    report = validate_audio(
        result.audio,
        expected_duration_s=expected,
        expected_sample_rate=song_a.sample_rate,
        expected_channels=song_a.channels,
    )
    print(f"Wrote conventional crossfade: {args.output}")
    print(f"Duration: {result.audio.duration_s:.3f}s (expected {expected:.3f}s)")
    for note in result.notes:
        print(f"Note: {note}")
    print(report.format_text())
    return 0 if report.passed else 1


def _cmd_render(args: argparse.Namespace) -> int:
    config = _config_for_cli(args)
    song_a = load_audio(args.song_a)
    song_b = load_audio(args.song_b)
    stems_a = load_stem_set(args.stems_a) if args.stems_a else None
    stems_b = load_stem_set(args.stems_b) if args.stems_b else None
    analysis_a = analyze_track_with_backend(
        song_a, config.analysis, backend=args.analyzer
    )
    analysis_b = analyze_track_with_backend(
        song_b, config.analysis, backend=args.analyzer
    )

    planner, advanced_status = build_planner_with_advanced(config)
    for error in advanced_status.errors:
        print(f"warning: Advanced plugin issue: {error}", file=sys.stderr)

    plan = planner.plan(
        song_a,
        song_b,
        stems_a=stems_a,
        stems_b=stems_b,
        strategy=args.strategy,
        duration_s=args.transition,
        analysis_a=analysis_a,
        analysis_b=analysis_b,
    )
    render = render_plan(
        song_a,
        song_b,
        plan,
        stems_a=stems_a,
        stems_b=stems_b,
        config=config,
    )
    save_audio(
        args.output,
        render.audio,
        subtype=config.render.wav_subtype,
        allow_clipping=not config.render.prevent_clipping,
    )
    expected = rendered_duration_s(song_a, song_b, plan)
    report = validate_audio(
        render.audio,
        expected_duration_s=expected,
        expected_sample_rate=song_a.sample_rate,
        expected_channels=song_a.channels,
    )

    print(_format_plan(plan))
    print("")
    print(f"Wrote DCP transition: {args.output}")
    print(f"Duration: {render.audio.duration_s:.3f}s (expected {expected:.3f}s)")
    for note in render.notes:
        print(f"Note: {note}")
    print(report.format_text())

    if args.plan_json:
        path = _write_json(args.plan_json, plan.to_dict())
        print(f"Plan JSON: {path}")
    if args.report_json:
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "engine_version": __version__,
            "input": {
                "song_a": str(Path(args.song_a)),
                "song_b": str(Path(args.song_b)),
                "stems_a": str(Path(args.stems_a)) if args.stems_a else None,
                "stems_b": str(Path(args.stems_b)) if args.stems_b else None,
                "strategy": args.strategy,
                "transition_s": args.transition,
                "analyzer": args.analyzer,
            },
            "analyses": {
                "a": analysis_a.to_dict(),
                "b": analysis_b.to_dict(),
            },
            "plan": plan.to_dict(),
            "render": {
                "file": str(Path(args.output)),
                "notes": list(render.notes),
                "master_gain_db": round(render.master_gain_db, 4),
                "validation": report.to_dict(),
            },
            "advanced_plugin": advanced_status.to_dict(),
        }
        path = _write_json(args.report_json, payload)
        print(f"Report JSON: {path}")
    return 0 if report.passed else 1


def _cmd_validate(args: argparse.Namespace) -> int:
    audio = load_audio(args.path)
    report = validate_audio(
        audio,
        expected_duration_s=args.expected_duration,
        max_silence_s=args.max_silence,
    )
    if args.json:
        _print_json(
            {
                "file": str(Path(args.path)),
                **report.to_dict(),
            }
        )
    else:
        print(f"File: {Path(args.path)}")
        print(report.format_text())
    return 0 if report.passed else 1


def _cmd_compare(args: argparse.Namespace) -> int:
    baseline = load_audio(args.baseline)
    candidate = load_audio(args.candidate)
    comparison = compare_audio(baseline, candidate)
    if args.json:
        _print_json(comparison)
        return 0

    metric_order = (
        ("duration_s", "Duration (s)"),
        ("peak_dbfs", "Peak (dBFS)"),
        ("rms_dbfs", "RMS (dBFS)"),
        ("crest_factor_db", "Crest factor (dB)"),
        ("longest_silence_s", "Longest silence (s)"),
        ("near_full_scale_ratio", "Near-full-scale ratio"),
        ("max_sample_step", "Max sample step"),
        ("dc_offset_max", "Max DC offset"),
    )
    print(f"{'Metric':26s} {'Baseline':>14s} {'DCP':>14s} {'Delta':>12s}")
    for key, label in metric_order:
        base_key = f"baseline_{key}"
        dcp_key = f"dcp_{key}"
        delta_key = f"delta_{key}"
        print(
            f"{label:26s} {comparison[base_key]:>14.6f} "
            f"{comparison[dcp_key]:>14.6f} {comparison[delta_key]:>12.6f}"
        )
    print("")
    print(str(comparison["note"]))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    result = run_demo(
        output_dir=args.out,
        duration_s=args.duration,
        transition_s=args.transition,
        sample_rate=args.sample_rate,
        seed=args.seed,
    )
    return (
        0
        if result.baseline_report.passed and result.dcp_report.passed
        else 1
    )


def _optional_module_status(module: str, distribution: str | None = None) -> str:
    spec = importlib.util.find_spec(module)
    if spec is None:
        return "not installed"
    if distribution is None:
        return "installed"
    return _version_of(distribution)


def _cmd_doctor(_args: argparse.Namespace) -> int:
    import numpy
    import soundfile
    import scipy

    print(f"DCP Core {__version__}")
    print(f"Python:     {platform.python_version()} ({sys.platform})")
    print(f"Executable: {sys.executable}")
    print("")
    print("Required dependencies:")
    print(f"  numpy      {numpy.__version__}")
    print(f"  scipy      {scipy.__version__}")
    try:
        libsndfile = soundfile.__libsndfile_version__
    except AttributeError:
        libsndfile = "unknown"
    print(f"  soundfile  {soundfile.__version__} (libsndfile {libsndfile})")
    print(f"  pytest     {_version_of('pytest')}")
    print("")
    print("Optional components (not required by Core):")
    print(f"  librosa    {_optional_module_status('librosa', 'librosa')}")
    print(f"  demucs     {_optional_module_status('demucs')}")
    print(f"  torch      {_optional_module_status('torch', 'torch')}")
    print("Stem separation: not bundled; user-provided stems are supported.")
    print("")

    registry, advanced = discover_advanced_strategies()
    print("Available strategies:")
    for name, strategy in registry.items():
        edition = "Advanced" if name in advanced.strategies else "Core"
        print(f"  {name:14s} [{edition}] {strategy.description}")
    print("")
    print(f"Advanced plugin: {advanced.detail}")
    for error in advanced.errors:
        print(f"  error: {error}")
    print("")
    print("Audio: offline WAV rendering; mono/stereo input; PCM_24 output default.")
    print("Try: dcp demo")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dcp",
        description=(
            "DCP Core - an offline, explainable music-transition engine. "
            "All processing is local."
        ),
        epilog=(
            "Examples:\n"
            "  dcp doctor\n"
            "  dcp demo --out output/demo\n"
            "  dcp analyze song.wav\n"
            "  dcp synthesize --out output/synthetic\n"
            "  dcp crossfade song_a.wav song_b.wav baseline.wav --overlap 8\n"
            "  dcp render song_a.wav song_b.wav dcp.wav "
            "--stems-a stems/a --stems-b stems/b --strategy stem_level\n"
            "  dcp validate dcp.wav --expected-duration 24\n"
            "  dcp compare baseline.wav dcp.wav\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "--debug", action="store_true", help="show tracebacks for unexpected errors"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser(
        "analyze", help="analyze levels, tempo, and optional beat positions"
    )
    analyze.add_argument("path", help="audio file to analyze")
    analyze.add_argument(
        "--analyzer",
        choices=("auto", "basic", "librosa"),
        default="auto",
        help=(
            "analysis backend; auto uses librosa when installed, basic is the "
            "dependency-light fallback"
        ),
    )
    analyze.add_argument("--json", action="store_true", help="print JSON")
    analyze.set_defaults(func=_cmd_analyze)

    synthesize = subparsers.add_parser(
        "synthesize", help="generate two synthetic songs with stems"
    )
    synthesize.add_argument(
        "--out", default="output/synthetic", help="output directory"
    )
    synthesize.add_argument("--duration-a", type=float, default=16.0)
    synthesize.add_argument("--duration-b", type=float, default=16.0)
    synthesize.add_argument("--bpm-a", type=float, default=120.0)
    synthesize.add_argument("--bpm-b", type=float, default=124.0)
    synthesize.add_argument("--sample-rate", type=int, default=44100)
    synthesize.add_argument("--seed", type=int, default=7)
    synthesize.set_defaults(func=_cmd_synthesize)

    crossfade = subparsers.add_parser(
        "crossfade", help="render a conventional equal-power crossfade"
    )
    crossfade.add_argument("song_a", help="Song A audio file")
    crossfade.add_argument("song_b", help="Song B audio file")
    crossfade.add_argument("output", help="output WAV path")
    crossfade.add_argument(
        "--overlap", type=float, default=8.0, help="overlap in seconds"
    )
    crossfade.add_argument(
        "--no-limit",
        action="store_true",
        help="do not apply the automatic master gain that prevents clipping",
    )
    crossfade.set_defaults(func=_cmd_crossfade)

    render = subparsers.add_parser(
        "render", help="plan and render a DCP transition"
    )
    render.add_argument("song_a", help="Song A audio file")
    render.add_argument("song_b", help="Song B audio file")
    render.add_argument("output", help="output WAV path")
    render.add_argument(
        "--stems-a", help="directory containing Song A stem WAV files"
    )
    render.add_argument(
        "--stems-b", help="directory containing Song B stem WAV files"
    )
    render.add_argument(
        "--strategy",
        default="auto",
        help=(
            "auto, crossfade, stem_level, rhythmic, atmospheric, conservative, "
            "optimized, or an installed Advanced strategy"
        ),
    )
    render.add_argument(
        "--transition", type=float, default=8.0, help="transition seconds"
    )
    render.add_argument(
        "--analyzer",
        choices=("auto", "basic", "librosa"),
        default="auto",
        help=(
            "analysis backend; auto uses librosa when installed, basic is the "
            "dependency-light fallback"
        ),
    )
    render.add_argument(
        "--no-limit",
        action="store_true",
        help="do not apply the automatic master gain that prevents clipping",
    )
    render.add_argument("--plan-json", help="write the transition plan as JSON")
    render.add_argument("--report-json", help="write a full render report as JSON")
    render.set_defaults(func=_cmd_render)

    validate = subparsers.add_parser(
        "validate", help="check duration, levels, silence, and discontinuities"
    )
    validate.add_argument("path", help="audio file to validate")
    validate.add_argument(
        "--expected-duration", type=float, default=None, help="expected seconds"
    )
    validate.add_argument(
        "--max-silence",
        type=float,
        default=None,
        help="maximum allowed interior silence in seconds",
    )
    validate.add_argument("--json", action="store_true", help="print JSON")
    validate.set_defaults(func=_cmd_validate)

    compare = subparsers.add_parser(
        "compare", help="compare objective metrics of two renders"
    )
    compare.add_argument("baseline", help="baseline audio file")
    compare.add_argument("candidate", help="DCP audio file")
    compare.add_argument("--json", action="store_true", help="print JSON")
    compare.set_defaults(func=_cmd_compare)

    demo = subparsers.add_parser(
        "demo", help="run the complete synthetic end-to-end demonstration"
    )
    demo.add_argument("--out", default="output/demo", help="output directory")
    demo.add_argument("--duration", type=float, default=16.0, help="track seconds")
    demo.add_argument(
        "--transition", type=float, default=8.0, help="transition seconds"
    )
    demo.add_argument("--sample-rate", type=int, default=44100)
    demo.add_argument("--seed", type=int, default=7)
    demo.set_defaults(func=_cmd_demo)

    doctor = subparsers.add_parser(
        "doctor", help="show environment, dependencies, and plugin status"
    )
    doctor.set_defaults(func=_cmd_doctor)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (DCPError, FileNotFoundError, ValueError) as exc:
        print(f"dcp: error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("dcp: interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        if getattr(args, "debug", False):
            raise
        print(
            f"dcp: unexpected error: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Reproducible end-to-end demonstration of the Core workflow."""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Callable

from . import __version__
from .analysis import TrackAnalysis, analyze_track
from .audio_io import save_audio, save_stem_set
from .config import DEFAULT_CONFIG
from .models import DCPError
from .rendering import render_crossfade, render_plan, rendered_duration_s
from .synthetic import generate_synthetic_track
from .transitions import TransitionPlan, plan_transition
from .validation import (
    ValidationReport,
    compare_audio,
    validate_audio,
)


@dataclass(frozen=True)
class DemoResult:
    output_dir: Path
    baseline_path: Path
    dcp_path: Path
    plan_path: Path
    report_path: Path
    plan: TransitionPlan
    analyses: tuple[TrackAnalysis, TrackAnalysis]
    baseline_report: ValidationReport
    dcp_report: ValidationReport
    comparison: dict[str, object]
    elapsed_s: float


def _relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


def _dependency_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def run_demo(
    *,
    output_dir: str | Path = "output/demo",
    duration_s: float = 16.0,
    transition_s: float = 8.0,
    sample_rate: int = 44100,
    seed: int = 7,
    echo: Callable[[str], None] | None = print,
) -> DemoResult:
    """Generate two synthetic songs, render both outputs, and validate them."""

    started = time.perf_counter()

    def say(message: str) -> None:
        if echo is not None:
            echo(message)

    duration = float(duration_s)
    transition = float(transition_s)
    if duration < 4.0:
        raise DCPError(f"demo duration must be at least 4 seconds, got {duration}")
    if transition < DEFAULT_CONFIG.transition.min_duration_s:
        raise DCPError(
            f"demo transition must be at least "
            f"{DEFAULT_CONFIG.transition.min_duration_s:.1f}s, got {transition}"
        )
    if transition >= duration:
        raise DCPError(
            f"demo transition ({transition:.1f}s) must be shorter than the "
            f"track duration ({duration:.1f}s)"
        )
    if int(seed) < 0:
        raise DCPError("demo seed must be non-negative")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    stem_root = root / "stems"
    stem_a_dir = stem_root / "a"
    stem_b_dir = stem_root / "b"

    say("[1/6] Generating synthetic songs and stems...")
    track_a = generate_synthetic_track(
        bpm=120.0,
        duration_s=duration,
        sample_rate=int(sample_rate),
        seed=int(seed),
        root_midi=57,
        mode="minor",
        vocal_lead_in_s=0.4,
        vocal_lead_out_s=0.8,
    )
    track_b = generate_synthetic_track(
        bpm=124.0,
        duration_s=duration,
        sample_rate=int(sample_rate),
        seed=int(seed) + 1,
        root_midi=60,
        mode="major",
        vocal_lead_in_s=1.0,
        vocal_lead_out_s=0.4,
    )

    song_a_path = save_audio(root / "song_a.wav", track_a.mix)
    song_b_path = save_audio(root / "song_b.wav", track_b.mix)
    save_stem_set(stem_a_dir, track_a.stems)
    save_stem_set(stem_b_dir, track_b.stems)

    say("[2/6] Analyzing tracks...")
    analysis_a = analyze_track(track_a.mix)
    analysis_b = analyze_track(track_b.mix)
    say(
        f"      Song A: {track_a.key_name}, {track_a.bpm:.0f} BPM, "
        f"tempo estimate {analysis_a.tempo_bpm:.2f} BPM "
        f"(confidence {analysis_a.tempo_confidence:.2f})"
        if analysis_a.tempo_bpm is not None
        else f"      Song A: {track_a.key_name}, tempo estimate unavailable"
    )
    say(
        f"      Song B: {track_b.key_name}, {track_b.bpm:.0f} BPM, "
        f"tempo estimate {analysis_b.tempo_bpm:.2f} BPM "
        f"(confidence {analysis_b.tempo_confidence:.2f})"
        if analysis_b.tempo_bpm is not None
        else f"      Song B: {track_b.key_name}, tempo estimate unavailable"
    )

    say(f"[3/6] Rendering conventional crossfade baseline ({transition:.2f}s overlap)...")
    baseline = render_crossfade(
        track_a.mix, track_b.mix, overlap_s=transition
    )
    baseline_path = save_audio(root / "baseline_crossfade.wav", baseline.audio)

    say("[4/6] Planning and rendering stem-level DCP transition...")
    plan = plan_transition(
        track_a.mix,
        track_b.mix,
        stems_a=track_a.stems,
        stems_b=track_b.stems,
        strategy="stem_level",
        duration_s=transition,
        analysis_a=analysis_a,
        analysis_b=analysis_b,
    )
    dcp_render = render_plan(
        track_a.mix,
        track_b.mix,
        plan,
        stems_a=track_a.stems,
        stems_b=track_b.stems,
    )
    dcp_path = save_audio(root / "dcp_stem_transition.wav", dcp_render.audio)
    plan_path = root / "plan.json"
    plan_path.write_text(
        json.dumps(plan.to_dict(), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    say("[5/6] Validating both renders...")
    baseline_expected = (
        track_a.mix.duration_s + track_b.mix.duration_s - transition
    )
    dcp_expected = rendered_duration_s(track_a.mix, track_b.mix, plan)
    baseline_report = validate_audio(
        baseline.audio,
        expected_duration_s=baseline_expected,
        expected_sample_rate=track_a.mix.sample_rate,
        expected_channels=track_a.mix.channels,
    )
    dcp_report = validate_audio(
        dcp_render.audio,
        expected_duration_s=dcp_expected,
        expected_sample_rate=track_a.mix.sample_rate,
        expected_channels=track_a.mix.channels,
    )
    comparison = compare_audio(baseline.audio, dcp_render.audio)

    say("[6/6] Writing report...")
    elapsed = time.perf_counter() - started
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "engine": {
            "name": "dcp-music-engine",
            "edition": "core",
            "version": __version__,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "dependencies": {
                "numpy": _dependency_version("numpy"),
                "scipy": _dependency_version("scipy"),
                "soundfile": _dependency_version("soundfile"),
            },
        },
        "parameters": {
            "duration_s": duration,
            "transition_s": transition,
            "sample_rate": int(sample_rate),
            "seed": int(seed),
            "requested_strategy": "stem_level",
        },
        "files": {
            "song_a": _relative(song_a_path, root),
            "song_b": _relative(song_b_path, root),
            "stems_a": _relative(stem_a_dir, root),
            "stems_b": _relative(stem_b_dir, root),
            "baseline": _relative(baseline_path, root),
            "dcp": _relative(dcp_path, root),
        },
        "songs": {
            "a": {
                **track_a.metadata(),
                "analysis": analysis_a.to_dict(),
            },
            "b": {
                **track_b.metadata(),
                "analysis": analysis_b.to_dict(),
            },
        },
        "plan": plan.to_dict(),
        "renders": {
            "baseline": {
                "file": _relative(baseline_path, root),
                "notes": list(baseline.notes),
                "master_gain_db": round(baseline.master_gain_db, 4),
                "validation": baseline_report.to_dict(),
            },
            "dcp": {
                "file": _relative(dcp_path, root),
                "notes": list(dcp_render.notes),
                "master_gain_db": round(dcp_render.master_gain_db, 4),
                "validation": dcp_report.to_dict(),
            },
        },
        "comparison": comparison,
        "limitations": [
            "Synthetic audio and stylized stems are used; this is not a "
            "copyrighted-song demonstration.",
            "No source separation model is bundled; stems are generated or "
            "supplied by the user.",
            "Objective metrics and automated checks cannot establish musical "
            "quality; listening tests are still required.",
            "Tempo estimation is approximate and can be half/double-time "
            "ambiguous.",
            "This is an offline renderer, not a real-time playback engine.",
            "No key/harmonic compatibility analysis, phrase detection, or "
            "time-stretching is implemented in Core.",
        ],
        "elapsed_s": round(elapsed, 3),
    }
    report_path = root / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    try:
        from .visualizer import save_html_report

        preview_path = root / "preview.html"
        save_html_report(
            preview_path,
            plan.to_dict(),
            report,
            audio_paths={
                "baseline": baseline_path.name,
                "dcp": dcp_path.name,
                "song_a": song_a_path.name,
                "song_b": song_b_path.name,
            },
            title="DCP Demo Transition Preview",
        )
    except Exception:
        preview_path = None

    say("")
    say("Demo complete.")
    say(f"  Song A:    {song_a_path}")
    say(f"  Song B:    {song_b_path}")
    say(f"  Stems A:   {stem_a_dir}")
    say(f"  Stems B:   {stem_b_dir}")
    say(f"  Baseline:  {baseline_path}")
    say(f"  DCP:       {dcp_path}")
    say(f"  Plan:      {plan_path}")
    say(f"  Report:    {report_path}")
    if preview_path and preview_path.exists():
        say(f"  Preview:   {preview_path}")
    say(
        f"  Validation: baseline "
        f"{'PASS' if baseline_report.passed else 'FAIL'}, DCP "
        f"{'PASS' if dcp_report.passed else 'FAIL'}"
    )
    say(f"  Elapsed:   {elapsed:.2f}s")

    return DemoResult(
        output_dir=root,
        baseline_path=baseline_path,
        dcp_path=dcp_path,
        plan_path=plan_path,
        report_path=report_path,
        plan=plan,
        analyses=(analysis_a, analysis_b),
        baseline_report=baseline_report,
        dcp_report=dcp_report,
        comparison=comparison,
        elapsed_s=elapsed,
    )

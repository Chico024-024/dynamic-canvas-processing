"""Command-line behavior, exit codes, and end-to-end demonstration."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from dcp.analysis import librosa_available
from dcp.audio_io import load_audio
from dcp.cli import main
from dcp.validation import validate_audio


def _synthesize(tmp_path: Path, capsys) -> dict[str, Path]:
    out = tmp_path / "synthetic"
    code = main(
        [
            "synthesize",
            "--out",
            str(out),
            "--duration-a",
            "4",
            "--duration-b",
            "4",
            "--sample-rate",
            "11025",
            "--seed",
            "11",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert (out / "song_a.wav").exists()
    return {
        "root": out,
        "a": out / "song_a.wav",
        "b": out / "song_b.wav",
        "stems_a": out / "stems" / "a",
        "stems_b": out / "stems" / "b",
    }


def test_synthesize_writes_manifest_and_stems(tmp_path, capsys):
    paths = _synthesize(tmp_path, capsys)
    assert paths["stems_a"].is_dir()
    assert len(list(paths["stems_a"].glob("*.wav"))) == 4
    manifest = json.loads((paths["root"] / "manifest.json").read_text())
    assert manifest["engine_version"] == "0.1.0"
    assert manifest["a"]["bpm"] == 120.0
    assert manifest["b"]["bpm"] == 124.0
    assert manifest["a"]["synthetic"] is True


def test_analyze_json_reports_file_properties(tmp_path, capsys):
    paths = _synthesize(tmp_path, capsys)
    code = main(["analyze", str(paths["a"]), "--json"])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    payload = json.loads(captured.out)
    assert payload["duration_s"] == pytest.approx(4.0)
    assert payload["sample_rate"] == 11025
    assert payload["channels"] == 2
    assert "peak_dbfs" in payload
    assert "tempo_bpm" in payload


def test_analyze_accepts_explicit_basic_backend(tmp_path, capsys):
    paths = _synthesize(tmp_path, capsys)
    code = main(["analyze", str(paths["a"]), "--analyzer", "basic"])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "Analyzer:       basic" in captured.out
    assert "Detected beats: 0" in captured.out


@pytest.mark.skipif(
    not librosa_available(), reason="optional librosa backend is not installed"
)
def test_render_with_librosa_analyzer_records_selection(tmp_path, capsys):
    paths = _synthesize(tmp_path, capsys)
    output = tmp_path / "dcp.wav"
    report_json = tmp_path / "report.json"
    code = main(
        [
            "render",
            str(paths["a"]),
            str(paths["b"]),
            str(output),
            "--strategy",
            "crossfade",
            "--transition",
            "2",
            "--analyzer",
            "librosa",
            "--report-json",
            str(report_json),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    report = json.loads(report_json.read_text())
    assert report["input"]["analyzer"] == "librosa"
    assert report["analyses"]["a"]["analyzer"] == "librosa"
    assert report["analyses"]["a"]["beat_times_s"]


def test_analyze_missing_file_returns_two(tmp_path, capsys):
    code = main(["analyze", str(tmp_path / "absent.wav")])
    captured = capsys.readouterr()
    assert code == 2
    assert "not found" in captured.err


def test_analyze_corrupt_file_returns_two(tmp_path, capsys):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not audio")
    code = main(["analyze", str(bad)])
    captured = capsys.readouterr()
    assert code == 2
    assert "unsupported or corrupt" in captured.err


def test_crossfade_command_writes_valid_output(tmp_path, capsys):
    paths = _synthesize(tmp_path, capsys)
    output = tmp_path / "baseline.wav"
    code = main(
        [
            "crossfade",
            str(paths["a"]),
            str(paths["b"]),
            str(output),
            "--overlap",
            "2",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert output.exists()
    rendered = load_audio(output)
    assert rendered.duration_s == pytest.approx(6.0, abs=0.02)
    assert validate_audio(
        rendered, expected_duration_s=6.0, expected_sample_rate=11025
    ).passed
    assert "Validation: PASS" in captured.out


def test_render_command_with_stems_writes_plan_and_report(tmp_path, capsys):
    paths = _synthesize(tmp_path, capsys)
    output = tmp_path / "dcp.wav"
    plan_json = tmp_path / "plan.json"
    report_json = tmp_path / "report.json"
    code = main(
        [
            "render",
            str(paths["a"]),
            str(paths["b"]),
            str(output),
            "--stems-a",
            str(paths["stems_a"]),
            "--stems-b",
            str(paths["stems_b"]),
            "--strategy",
            "stem_level",
            "--transition",
            "2",
            "--plan-json",
            str(plan_json),
            "--report-json",
            str(report_json),
        ]
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert output.exists()
    rendered = load_audio(output)
    assert rendered.duration_s == pytest.approx(6.0, abs=0.02)

    plan = json.loads(plan_json.read_text())
    assert plan["strategy"] == "stem_level"
    assert len(plan["automations"]) == 8

    report = json.loads(report_json.read_text())
    assert report["render"]["validation"]["passed"] is True
    assert report["plan"]["strategy"] == "stem_level"
    assert report["input"]["stems_a"] == str(paths["stems_a"])
    assert "Validation: PASS" in captured.out


def test_render_unknown_strategy_returns_two(tmp_path, capsys):
    paths = _synthesize(tmp_path, capsys)
    code = main(
        [
            "render",
            str(paths["a"]),
            str(paths["b"]),
            str(tmp_path / "out.wav"),
            "--strategy",
            "magical",
            "--transition",
            "2",
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert "unknown transition strategy" in captured.err
    assert not (tmp_path / "out.wav").exists()


def test_render_missing_stem_directory_returns_two(tmp_path, capsys):
    paths = _synthesize(tmp_path, capsys)
    code = main(
        [
            "render",
            str(paths["a"]),
            str(paths["b"]),
            str(tmp_path / "out.wav"),
            "--stems-a",
            str(tmp_path / "absent"),
            "--transition",
            "2",
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert "stem directory not found" in captured.err


def test_label_command_records_preferences(tmp_path, capsys, monkeypatch):
    paths = _synthesize(tmp_path, capsys)
    answers = iter(["a", "tie"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    out = tmp_path / "labeling"
    log = tmp_path / "preferences.jsonl"
    code = main(
        [
            "label",
            str(paths["a"]),
            str(paths["b"]),
            "--stems-a",
            str(paths["stems_a"]),
            "--stems-b",
            str(paths["stems_b"]),
            "--transition",
            "2",
            "--max-pairs",
            "2",
            "--seed",
            "9",
            "--out",
            str(out),
            "--log",
            str(log),
            "--note",
            "cli",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert log.exists()
    records = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(records) == 2
    assert records[0]["choice"] == "a"
    assert records[1]["choice"] == "tie"
    assert records[0]["note"] == "cli"
    assert "No model has been trained" in captured.out
    assert len(list(out.rglob("session.json"))) == 1
    assert len(list(out.rglob("*.wav"))) == 5


def test_label_requires_stems(tmp_path, capsys):
    paths = _synthesize(tmp_path, capsys)
    code = main(
        [
            "label",
            str(paths["a"]),
            str(paths["b"]),
            "--transition",
            "2",
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert "requires stems" in captured.err


def test_validate_exit_codes_and_json(tmp_path, capsys):
    paths = _synthesize(tmp_path, capsys)
    assert main(["validate", str(paths["a"]), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True

    assert main(["validate", str(paths["a"]), "--expected-duration", "99"]) == 1
    captured = capsys.readouterr()
    assert "Validation: FAIL" in captured.out

    assert main(["validate", str(tmp_path / "absent.wav")]) == 2
    assert "not found" in capsys.readouterr().err


def test_compare_command_and_json(tmp_path, capsys):
    paths = _synthesize(tmp_path, capsys)
    assert main(["compare", str(paths["a"]), str(paths["b"])]) == 0
    captured = capsys.readouterr()
    assert "Metric" in captured.out
    assert "Objective metrics only" in captured.out

    assert main(["compare", str(paths["a"]), str(paths["b"]), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "delta_rms_dbfs" in payload
    assert "not establish musical quality" in payload["note"]


def test_demo_command_end_to_end(tmp_path, capsys):
    out = tmp_path / "demo"
    code = main(
        [
            "demo",
            "--out",
            str(out),
            "--duration",
            "4",
            "--transition",
            "2",
            "--sample-rate",
            "11025",
            "--seed",
            "3",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    for name in (
        "song_a.wav",
        "song_b.wav",
        "baseline_crossfade.wav",
        "dcp_stem_transition.wav",
        "plan.json",
        "report.json",
    ):
        assert (out / name).exists(), name

    report = json.loads((out / "report.json").read_text())
    assert report["renders"]["baseline"]["validation"]["passed"] is True
    assert report["renders"]["dcp"]["validation"]["passed"] is True
    assert report["plan"]["strategy"] == "stem_level"
    assert "Validation: baseline PASS, DCP PASS" in captured.out


def test_doctor_reports_environment(capsys):
    assert main(["doctor"]) == 0
    captured = capsys.readouterr()
    assert "DCP Core 0.1.0" in captured.out
    assert "Available strategies:" in captured.out
    assert "stem_level" in captured.out
    assert "Stem separation: not bundled" in captured.out
    assert "Advanced plugin:" in captured.out


def test_help_and_missing_command(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    for command in (
        "analyze",
        "synthesize",
        "crossfade",
        "render",
        "label",
        "validate",
        "compare",
        "demo",
        "doctor",
    ):
        assert command in captured.out

    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


def test_module_entry_point_runs_in_subprocess(tmp_path):
    env = dict(os.environ)
    src_dir = str(Path(__file__).resolve().parent.parent / "src")
    env["PYTHONPATH"] = (src_dir + os.pathsep + env["PYTHONPATH"]) if "PYTHONPATH" in env else src_dir
    completed = subprocess.run(
        [sys.executable, "-m", "dcp", "--version"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "dcp 0.1.0"

"""Tests for visual transition previewer and CLI visualize/separate commands."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dcp.cli import main
from dcp.visualizer import generate_html_report, save_html_report


@pytest.fixture
def sample_plan_dict():
    return {
        "schema_version": 1,
        "strategy": "stem_level",
        "requested_strategy": "stem_level",
        "duration_s": 8.0,
        "start_time_s": 8.0,
        "end_time_s": 16.0,
        "reasoning": ["Independent stem automation is used for both songs."],
        "warnings": [],
        "metadata": {
            "stem_aware": True,
            "key_a": "Am",
            "camelot_a": "8A",
            "key_b": "C",
            "camelot_b": "8B",
            "harmonic_compatibility": {
                "compatibility_score": 0.9,
            },
        },
        "automations": [
            {
                "source": "a",
                "stem": "vocals",
                "curve": "smoothstep",
                "points": [
                    {"time_s": 0.0, "gain_db": 0.0},
                    {"time_s": 3.6, "gain_db": -60.0},
                ],
                "effects": [],
            },
            {
                "source": "b",
                "stem": "vocals",
                "curve": "smoothstep",
                "points": [
                    {"time_s": 4.4, "gain_db": -60.0},
                    {"time_s": 8.0, "gain_db": 0.0},
                ],
                "effects": [],
            },
        ],
    }


def test_generate_html_report_contains_key_elements(sample_plan_dict):
    html = generate_html_report(sample_plan_dict, title="Test Transition")
    assert "<!DOCTYPE html>" in html
    assert "Test Transition" in html
    assert "stem_level" in html
    assert "8.00s" in html
    assert "Core: Open Source" in html
    assert "Advanced SDK: Royalty Kit" in html
    assert "curves-canvas" in html
    assert "audio-dcp" in html
    assert "audio-baseline" in html
    assert "Independent stem automation" in html


def test_save_html_report_writes_file(tmp_path, sample_plan_dict):
    out_file = tmp_path / "reports" / "transition_preview.html"
    saved = save_html_report(out_file, sample_plan_dict)
    assert saved.exists()
    content = saved.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "planData" in content


def test_cli_visualize_on_directory(tmp_path, sample_plan_dict, capsys):
    work_dir = tmp_path / "demo_out"
    work_dir.mkdir(parents=True)
    plan_file = work_dir / "plan.json"
    plan_file.write_text(json.dumps(sample_plan_dict))

    code = main(["visualize", str(work_dir)])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "Wrote visual transition preview" in captured.out
    assert (work_dir / "preview.html").exists()


def test_cli_visualize_on_plan_file(tmp_path, sample_plan_dict, capsys):
    plan_file = tmp_path / "custom_plan.json"
    plan_file.write_text(json.dumps(sample_plan_dict))

    code = main(["visualize", str(plan_file)])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert (tmp_path / "custom_plan_preview.html").exists()


def test_cli_visualize_missing_directory_errors(tmp_path, capsys):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    code = main(["visualize", str(empty_dir)])
    captured = capsys.readouterr()
    assert code == 2
    assert "does not contain plan.json" in captured.err


def test_cli_separate_missing_demucs_reports_error(tmp_path, capsys, monkeypatch):
    import dcp.stems.demucs_adapter as da

    monkeypatch.setattr(da, "demucs_available", lambda: False)

    dummy_wav = tmp_path / "track.wav"
    dummy_wav.write_bytes(b"RIFF dummy")

    code = main(["separate", str(dummy_wav), "--out", str(tmp_path / "stems")])
    captured = capsys.readouterr()
    assert code == 2
    assert "Demucs stem separation requires" in captured.err

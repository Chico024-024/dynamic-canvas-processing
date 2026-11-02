"""Local pairwise-preference labeling."""

from __future__ import annotations

import json

import pytest

from dcp.labeling import _ask_choice, run_labeling_session
from dcp.models import DCPError

CANDIDATE_FILES = (
    "01_default.wav",
    "02_wide_vocal_handoff.wav",
    "03_early_drums_exit.wav",
    "04_late_bass_entry.wav",
    "05_atmosphere_blend.wav",
)


def _scripted(answers):
    replies = list(answers)
    prompts: list[str] = []

    def ask(prompt: str) -> str:
        prompts.append(prompt)
        return replies.pop(0)

    return ask, prompts


def _quiet(_message: str) -> None:
    pass


def _run(tmp_path, synthetic_pair, answers, **overrides):
    song_a, song_b = synthetic_pair
    ask, prompts = _scripted(answers)
    result = run_labeling_session(
        song_a.mix,
        song_b.mix,
        stems_a=song_a.stems,
        stems_b=song_b.stems,
        transition_s=2.0,
        output_dir=tmp_path / overrides.pop("out", "labels"),
        seed=overrides.pop("seed", 3),
        ask=ask,
        echo=_quiet,
        **overrides,
    )
    return result, prompts


def test_labeling_session_records_pairwise_choices(tmp_path, synthetic_pair):
    result, prompts = _run(
        tmp_path,
        synthetic_pair,
        ["a", "b", "tie", "skip"],
        max_pairs=4,
        note="unit",
    )

    assert len(prompts) == 4
    assert result.records_written == 4
    assert result.pairs_offered == 4
    assert not result.stopped_early
    assert len(result.candidates) == 5
    for filename in CANDIDATE_FILES:
        assert (result.session_dir / filename).exists(), filename

    session = json.loads((result.session_dir / "session.json").read_text())
    assert session["schema_version"] == 1
    assert session["kind"] == "labeling_session"
    assert session["session"] == result.session_id
    assert session["seed"] == 3
    assert session["note"] == "unit"
    assert len(session["candidates"]) == 5

    raw = result.log_path.read_text(encoding="utf-8")
    records = [json.loads(line) for line in raw.splitlines()]
    assert [record["choice"] for record in records] == ["a", "b", "tie", "skip"]
    assert records[0]["preferred_candidate"] == records[0]["candidate_a"]["name"]
    assert records[1]["preferred_candidate"] == records[1]["candidate_b"]["name"]
    assert records[2]["preferred_candidate"] is None
    assert records[3]["preferred_candidate"] is None

    for record in records:
        assert record["schema_version"] == 1
        assert record["kind"] == "pairwise_preference"
        assert record["session"] == result.session_id
        assert record["note"] == "unit"
        assert record["pair_count"] == 4
        for side in ("candidate_a", "candidate_b"):
            candidate = record[side]
            assert candidate["file"] in CANDIDATE_FILES
            assert candidate["features"]["candidate"] == candidate["name"]
            assert len(candidate["features"]["losses"]) == 7
            assert 0.0 <= candidate["objective"] <= 20.0
        assert len(record["source"]["a"]["sha256"]) == 64
        assert len(record["source"]["b"]["sha256"]) == 64
        assert len(record["source"]["a"]["stems"]["vocals"]) == 64
        assert record["source"]["a"]["stems"]["bass"]
        assert record["transition"]["requested_strategy"] == "optimized"

    leaked = str(tmp_path)
    assert leaked not in raw
    assert leaked.replace("\\", "\\\\") not in raw


def test_labeling_reprompts_and_stops_on_quit(tmp_path, synthetic_pair):
    result, prompts = _run(
        tmp_path, synthetic_pair, ["what", "a", "quit"], max_pairs=5, seed=11
    )
    assert len(prompts) == 3
    assert result.records_written == 1
    assert result.stopped_early
    assert len(result.log_path.read_text().splitlines()) == 1


def test_seed_reproduces_pairs_and_display_order(tmp_path, synthetic_pair):
    first, _ = _run(
        tmp_path,
        synthetic_pair,
        ["tie", "tie", "tie"],
        out="one",
        seed=42,
        max_pairs=3,
    )
    second, _ = _run(
        tmp_path,
        synthetic_pair,
        ["tie", "tie", "tie"],
        out="two",
        seed=42,
        max_pairs=3,
    )

    def order(result):
        return [
            (record["candidate_a"]["name"], record["candidate_b"]["name"])
            for record in map(json.loads, result.log_path.read_text().splitlines())
        ]

    assert order(first) == order(second)


def test_labeling_requires_a_stem_set(synthetic_pair):
    song_a, song_b = synthetic_pair
    with pytest.raises(DCPError, match="requires stems"):
        run_labeling_session(song_a.mix, song_b.mix, transition_s=2.0)


def test_labeling_rejects_invalid_pair_limit(synthetic_pair):
    song_a, song_b = synthetic_pair
    with pytest.raises(DCPError, match="max_pairs"):
        run_labeling_session(
            song_a.mix,
            song_b.mix,
            stems_a=song_a.stems,
            max_pairs=0,
        )


def test_ask_choice_returns_none_on_eof():
    def ask(_prompt: str) -> str:
        raise EOFError

    assert _ask_choice(ask, _quiet) is None

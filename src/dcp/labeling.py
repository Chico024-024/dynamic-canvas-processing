"""Local human-preference labeling for transition candidates.

``dcp label`` renders the optimizer's candidate plans to WAV files, asks the
user which candidate in each pair sounds better, and appends the answer to a
local JSONL file. It is data collection, not learning: no model is trained, no
audio is played by DCP, and nothing is uploaded or sent over the network.

The log is a stream of self-contained pairwise-preference records. Each record
contains the measured features of both displayed candidates, the choice, and
audio content hashes. It intentionally contains no absolute filesystem paths,
so the dataset can be shared independently of where it was recorded.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from . import __version__
from .analysis import TrackAnalysis, analyze_track_with_backend
from .audio_io import save_audio
from .config import DCPConfig, DEFAULT_CONFIG
from .models import AudioBuffer, DCPError, StemSet
from .transitions.context import TransitionContext
from .transitions.optimize import (
    WeightedLossScorer,
    evaluate_candidate,
    generate_candidate_plans,
)
from .transitions.quality import CandidateFeatures, QualityScoringError

_SCHEMA_VERSION = 1
_CHOICE_PROMPT = "Which is better? [a/b/tie/skip/quit]: "
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class LabeledCandidate:
    """One rendered candidate available for comparison."""

    label: str
    filename: str
    features: CandidateFeatures
    objective: float


@dataclass(frozen=True)
class LabelingResult:
    """Summary of one labeling session."""

    session_id: str
    session_dir: Path
    log_path: Path
    candidates: tuple[str, ...]
    pairs_offered: int
    records_written: int
    stopped_early: bool


def run_labeling_session(
    song_a: AudioBuffer,
    song_b: AudioBuffer,
    *,
    stems_a: StemSet | None = None,
    stems_b: StemSet | None = None,
    config: DCPConfig | None = None,
    transition_s: float = 8.0,
    analyzer: str = "auto",
    analysis_a: TrackAnalysis | None = None,
    analysis_b: TrackAnalysis | None = None,
    output_dir: str | Path = "output/labeling",
    log_path: str | Path | None = None,
    max_pairs: int | None = None,
    seed: int | None = None,
    note: str = "",
    ask: Callable[[str], str] | None = None,
    echo: Callable[[str], None] = print,
    now: Callable[[], datetime] | None = None,
) -> LabelingResult:
    """Render candidate pairs locally and append human preferences to JSONL.

    ``ask`` and ``echo`` are injectable for tests and non-interactive front
    ends. Candidate display order is randomized from ``seed``; the seed is
    recorded so a session can be reproduced. At least one stem set is required
    because the candidates are role-timing plans.
    """

    if stems_a is None and stems_b is None:
        raise DCPError(
            "labeling requires stems for at least one song "
            "(provide stems_a and/or stems_b)"
        )
    if not math.isfinite(float(transition_s)) or transition_s <= 0:
        raise DCPError(
            f"transition duration must be a positive finite number, got {transition_s}"
        )
    if max_pairs is not None and (
        isinstance(max_pairs, bool) or not isinstance(max_pairs, int) or max_pairs < 1
    ):
        raise DCPError(f"max_pairs must be a positive integer, got {max_pairs!r}")
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int)
    ):
        raise DCPError(f"seed must be an integer, got {seed!r}")

    cfg = config or DEFAULT_CONFIG
    first_analysis = analysis_a or analyze_track_with_backend(
        song_a, cfg.analysis, backend=analyzer
    )
    second_analysis = analysis_b or analyze_track_with_backend(
        song_b, cfg.analysis, backend=analyzer
    )

    context = TransitionContext(
        song_a=song_a,
        song_b=song_b,
        stems_a=stems_a,
        stems_b=stems_b,
        analysis_a=first_analysis,
        analysis_b=second_analysis,
        requested_duration_s=float(transition_s),
        requested_strategy="optimized",
        config=cfg,
    )

    plans = generate_candidate_plans(context)
    if len(plans) < 2:
        raise DCPError(
            "labeling requires at least two structurally valid candidate "
            f"plans, but generated {len(plans)}"
        )

    created = _utc_now(now)
    session_seed = (
        random.SystemRandom().randrange(2**31) if seed is None else int(seed)
    )
    rng = random.Random(session_seed)
    session_id = f"{created.strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(3)}"
    session_dir = Path(output_dir) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    log_file = (
        Path(log_path) if log_path is not None else Path(output_dir) / "labels.jsonl"
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)

    echo("DCP preference labeling - local only, nothing is uploaded.")
    echo(f"Session:    {session_id}")
    echo(f"Candidates: {session_dir}")
    echo(f"Log:        {log_file}")
    echo("DCP does not play audio; open the files in a local player.")
    echo("Choices: a = first better, b = second better, tie, skip, quit.")
    echo("")

    scorer = WeightedLossScorer(cfg)
    labeled: list[LabeledCandidate] = []
    for index, plan in enumerate(plans, start=1):
        candidate_name = str(plan.metadata.get("candidate", f"candidate_{index}"))
        try:
            evaluation = evaluate_candidate(context, plan, scorer=scorer)
        except QualityScoringError:
            raise
        except DCPError as exc:
            echo(f"Skipped candidate {candidate_name!r}: {exc}")
            continue
        filename = f"{index:02d}_{_safe_filename(candidate_name, index)}.wav"
        save_audio(
            session_dir / filename,
            evaluation.render.audio,
            subtype=cfg.render.wav_subtype,
            allow_clipping=not cfg.render.prevent_clipping,
        )
        labeled.append(
            LabeledCandidate(
                label=evaluation.features.candidate,
                filename=filename,
                features=evaluation.features,
                objective=evaluation.score,
            )
        )

    if len(labeled) < 2:
        raise DCPError(
            "labeling requires at least two renderable candidates, but only "
            f"{len(labeled)} could be rendered"
        )

    sources = {
        "a": _source_record(song_a, stems_a),
        "b": _source_record(song_b, stems_b),
    }
    transition = {
        "requested_strategy": "optimized",
        "requested_duration_s": round(float(transition_s), 6),
        "duration_s": round(plans[0].duration_s, 6),
        "analyzer_a": first_analysis.analyzer,
        "analyzer_b": second_analysis.analyzer,
        "tempo_bpm_a": _optional_round(first_analysis.tempo_bpm),
        "tempo_bpm_b": _optional_round(second_analysis.tempo_bpm),
    }

    session_payload = {
        "schema_version": _SCHEMA_VERSION,
        "kind": "labeling_session",
        "engine_version": __version__,
        "session": session_id,
        "created_at": created.isoformat(timespec="seconds"),
        "seed": session_seed,
        "note": note,
        "sources": sources,
        "transition": transition,
        "candidates": [_candidate_record(candidate) for candidate in labeled],
    }
    (session_dir / "session.json").write_text(
        json.dumps(session_payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    pairs = list(itertools.combinations(range(len(labeled)), 2))
    rng.shuffle(pairs)
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    echo(
        f"{len(labeled)} candidates rendered; {len(pairs)} pair(s) to compare."
    )

    records_written = 0
    stopped_early = False
    for pair_index, (left, right) in enumerate(pairs, start=1):
        if rng.random() < 0.5:
            left, right = right, left
        candidate_a = labeled[left]
        candidate_b = labeled[right]

        echo("")
        echo(f"Pair {pair_index}/{len(pairs)}")
        echo(f"  a: {session_dir / candidate_a.filename}  [{candidate_a.label}]")
        echo(f"  b: {session_dir / candidate_b.filename}  [{candidate_b.label}]")
        choice = _ask_choice(ask, echo)
        if choice is None:
            stopped_early = True
            echo("Input closed; ending the session.")
            break
        if choice == "quit":
            stopped_early = True
            echo("Session ended by user.")
            break

        preferred = (
            candidate_a.label
            if choice == "a"
            else candidate_b.label if choice == "b" else None
        )
        record = {
            "schema_version": _SCHEMA_VERSION,
            "kind": "pairwise_preference",
            "engine_version": __version__,
            "session": session_id,
            "recorded_at": _utc_now(now).isoformat(timespec="seconds"),
            "pair_index": pair_index,
            "pair_count": len(pairs),
            "source": sources,
            "transition": transition,
            "candidate_a": _candidate_record(candidate_a),
            "candidate_b": _candidate_record(candidate_b),
            "choice": choice,
            "preferred_candidate": preferred,
            "note": note,
        }
        _append_jsonl(log_file, record)
        records_written += 1

    return LabelingResult(
        session_id=session_id,
        session_dir=session_dir,
        log_path=log_file,
        candidates=tuple(candidate.label for candidate in labeled),
        pairs_offered=len(pairs),
        records_written=records_written,
        stopped_early=stopped_early,
    )


def _ask_choice(
    ask: Callable[[str], str] | None,
    echo: Callable[[str], None],
) -> str | None:
    """Prompt until a recognized choice is given; None means EOF."""

    reader = input if ask is None else ask
    aliases = {
        "a": "a",
        "1": "a",
        "b": "b",
        "2": "b",
        "t": "tie",
        "tie": "tie",
        "s": "skip",
        "skip": "skip",
        "q": "quit",
        "quit": "quit",
    }
    while True:
        try:
            answer = reader(_CHOICE_PROMPT)
        except EOFError:
            return None
        choice = aliases.get(answer.strip().lower())
        if choice is not None:
            return choice
        echo("Please answer a, b, tie, skip, or quit.")


def _candidate_record(candidate: LabeledCandidate) -> dict[str, object]:
    return {
        "name": candidate.label,
        "file": candidate.filename,
        "objective": round(candidate.objective, 6),
        "features": candidate.features.to_dict(),
    }


def _source_record(
    buffer: AudioBuffer, stems: StemSet | None
) -> dict[str, object]:
    return {
        "sha256": _audio_sha256(buffer),
        "duration_s": round(buffer.duration_s, 6),
        "sample_rate": buffer.sample_rate,
        "channels": buffer.channels,
        "stems": (
            None
            if stems is None
            else {
                name: _audio_sha256(audio)
                for name, audio in sorted(stems.items())
            }
        ),
    }


def _audio_sha256(buffer: AudioBuffer) -> str:
    digest = hashlib.sha256()
    flat = np.ascontiguousarray(buffer.samples).reshape(-1).view(np.uint8)
    digest.update(flat)
    return digest.hexdigest()


def _safe_filename(label: str, index: int) -> str:
    cleaned = _SAFE_NAME.sub("_", label.strip()).strip("._-")
    return cleaned or f"candidate_{index}"


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    line = json.dumps(payload, allow_nan=False)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")


def _optional_round(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(float(value), digits)


def _utc_now(now: Callable[[], datetime] | None) -> datetime:
    moment = datetime.now(timezone.utc) if now is None else now()
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


__all__ = [
    "LabeledCandidate",
    "LabelingResult",
    "run_labeling_session",
]

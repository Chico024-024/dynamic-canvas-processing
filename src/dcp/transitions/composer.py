# src/dcp/transitions/composer.py
"""
Layer 3: Autonomous Composition Planner & Transition Rejection Gate.
Consumes TrackMusicalIR for Song A and Song B, evaluates multi-hypothesis metrical locks,
selects contextual arrangement archetypes, or rejects musically incompatible transitions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

import numpy as np

from dcp.transitions.musical_ir import MetricalHypothesis, PhraseIR, TrackMusicalIR

TransitionMode = Literal[
    "CONTINUOUS_TEMPO_WARP",
    "METRIC_MODULATION_PIVOT",
    "ATEMPORAL_WASH_HANDOFF",
    "REJECTED",
]

ArrangementArchetype = Literal[
    "VACUUM_BREATH_DROP",
    "CONTINUOUS_CROSS_FILTER",
    "POLYRHYTHMIC_4_3_PIVOT",
    "GHOST_WASH_SUSPENSION",
    "REJECTED_FALLBACK",
]


@dataclass(frozen=True)
class ComposedBridgePlan:
    is_viable: bool
    transition_mode: TransitionMode
    archetype: ArrangementArchetype
    launchpad_phrase_a: PhraseIR
    impact_phrase_b: PhraseIR
    hypothesis_a: MetricalHypothesis
    hypothesis_b: MetricalHypothesis
    pitch_shift_semitones_b: int
    breath_beats: int                 # 0 (no breath silence), 1, or 2 beats
    feasibility_score: float          # Heuristic ranking score [0.0, 1.0]
    rejection_reason: str | None
    composer_rationale: str


def compose_optimal_bridge(
    ir_a: TrackMusicalIR,
    ir_b: TrackMusicalIR,
    min_feasibility_threshold: float = 0.58,
    allow_pitch_shift_semitones: int = 2,
) -> ComposedBridgePlan:
    """
    Searches candidate (Launchpad Phrase A, Impact Phrase B, Hypothesis A, Hypothesis B)
    tuples and selects the highest-scoring musical arrangement, or rejects the transition.
    """
    # Filter Launchpad candidates in second half of Song A, Impact candidates in first half of Song B
    n_p_a = len(ir_a.phrases)
    n_p_b = len(ir_b.phrases)
    cand_a = [p for p in ir_a.phrases if p.phrase_index >= max(0, int(0.45 * n_p_a))] or list(ir_a.phrases)
    cand_b = [p for p in ir_b.phrases if 0 < p.phrase_index <= max(1, int(0.60 * n_p_b))] or list(ir_b.phrases)

    best_plan: ComposedBridgePlan | None = None
    best_score = -1e9

    for p_a in cand_a:
        for p_b in cand_b:
            # 1. Harmonic Chroma & Sub-Bass Root Consonance across allowed semitone shifts
            c_a = np.asarray(p_a.chroma_12d, dtype=np.float64)
            c_b = np.asarray(p_b.chroma_12d, dtype=np.float64)
            best_shift = 0
            best_harm = -1.0

            for shift in range(-allow_pitch_shift_semitones, allow_pitch_shift_semitones + 1):
                rot_b = np.roll(c_b, shift)
                chroma_cos = float(np.dot(c_a, rot_b))
                interval = (p_b.bass_root_pc + shift - p_a.bass_root_pc) % 12
                root_bonus = 0.20 if interval in (0, 5, 7) else (-0.18 if interval in (1, 6, 11) else 0.0)
                shift_cost = 0.05 * abs(shift)
                h_val = float(np.clip(chroma_cos + root_bonus - shift_cost, 0.0, 1.0))
                if h_val > best_harm:
                    best_harm = h_val
                    best_shift = shift

            # 2. Vocal Phrase Clearance (penalize cutting mid-lyric or colliding dense vocals)
            vocal_collision = p_a.vocal_activity * p_b.vocal_activity
            vocal_clearance = float(np.clip(
                (1.0 - 0.65 * vocal_collision) * (1.0 if p_a.vocal_resolves_at_end else 0.65),
                0.0, 1.0
            ))

            # 3. Structural Contrast (Launchpad A groove/peak -> Impact B high low-end drop)
            structural_fit = 0.5 * p_a.low_end_weight + 0.5 * p_b.low_end_weight + 0.25 * p_b.boundary_salience
            structural_fit = float(np.clip(structural_fit, 0.0, 1.0))

            # 4. Evaluate Metrical Hypotheses (H_A x H_B)
            for h_a in ir_a.metrical_hypotheses:
                for h_b in ir_b.metrical_hypotheses:
                    # Do not chain two derived metric modulations simultaneously
                    if h_a.relation_to_base == "derived_4_3" and h_b.relation_to_base == "derived_4_3":
                        continue

                    ratio = h_b.bpm / max(h_a.bpm, 1e-3)
                    prior_weight = 0.5 * (
                        h_a.base_detection_confidence * h_a.metrical_plausibility_weight
                        + h_b.base_detection_confidence * h_b.metrical_plausibility_weight
                    )
                    acc_a = np.asarray(h_a.accent_profile_16, dtype=np.float64)
                    acc_b = np.asarray(h_b.accent_profile_16, dtype=np.float64)
                    accent_sim = float(np.dot(acc_a, acc_b))

                    mode: TransitionMode = "REJECTED"
                    arch: ArrangementArchetype = "REJECTED_FALLBACK"
                    breath_beats = 0
                    rhythm_score = 0.0
                    rationale = ""

                    is_metric_pivot = (h_a.relation_to_base == "derived_4_3") or (h_b.relation_to_base == "derived_4_3")

                    if 0.92 <= ratio <= 1.085 and not is_metric_pivot:
                        mode = "CONTINUOUS_TEMPO_WARP"
                        stretch_penalty = abs(1.0 - ratio) * 2.5
                        rhythm_score = float(np.clip(prior_weight * (0.7 + 0.3 * accent_sim) - stretch_penalty, 0.0, 1.0))

                        # Contextually decide whether a Breath silence is appropriate
                        if p_a.vocal_resolves_at_end and p_b.drum_crest_factor > 0.60 and p_b.vocal_activity < 0.55:
                            arch = "VACUUM_BREATH_DROP"
                            breath_beats = 2 if p_b.drum_crest_factor > 0.78 else 1
                            rationale = (
                                f"Direct bijective warp ({h_a.bpm:.1f}->{h_b.bpm:.1f} BPM) with "
                                f"{breath_beats}-beat Vacuum Breath before high-crest drop."
                            )
                        else:
                            arch = "CONTINUOUS_CROSS_FILTER"
                            breath_beats = 0
                            rationale = (
                                f"Direct bijective warp ({h_a.bpm:.1f}->{h_b.bpm:.1f} BPM) using continuous "
                                f"ZDF filter handoff (Breath suppressed to preserve vocal/groove continuity)."
                            )

                    elif 0.93 <= ratio <= 1.07 and is_metric_pivot:
                        mode = "METRIC_MODULATION_PIVOT"
                        arch = "POLYRHYTHMIC_4_3_PIVOT"
                        breath_beats = 1 if p_a.vocal_resolves_at_end else 0
                        rhythm_score = float(np.clip(prior_weight * 0.82 - abs(1.0 - ratio) * 2.0, 0.0, 1.0))
                        rationale = (
                            f"4:3 Metric Modulation pivot ({ir_a.base_bpm:.1f} BPM -> {ir_b.base_bpm:.1f} BPM "
                            f"via {h_a.hypothesis_id}/{h_b.hypothesis_id}, warp={(ratio - 1.0)*100:+.1f}%)."
                        )

                    elif h_a.relation_to_base == "detected_primary" and h_b.relation_to_base == "detected_primary":
                        # Atemporal FDN Freeze Wash when tempos cannot be beat-mixed (e.g., 27% delta)
                        if best_harm >= 0.65 and p_a.vocal_resolves_at_end:
                            mode = "ATEMPORAL_WASH_HANDOFF"
                            arch = "GHOST_WASH_SUSPENSION"
                            breath_beats = 2
                            rhythm_score = 0.58 * prior_weight
                            rationale = (
                                f"Tempo delta ({h_a.bpm:.1f}->{h_b.bpm:.1f} BPM) exceeds beat-mix bounds; "
                                f"suspending rhythm into lossless FDN Freeze wash at phrase resolution."
                            )

                    if mode == "REJECTED":
                        score = 0.25 * best_harm + 0.15 * vocal_clearance
                    else:
                        score = (
                            0.35 * best_harm
                            + 0.30 * rhythm_score
                            + 0.20 * vocal_clearance
                            + 0.15 * structural_fit
                        )

                    if score > best_score:
                        best_score = score
                        is_ok = (mode != "REJECTED") and (score >= min_feasibility_threshold) and (best_harm >= 0.45)
                        rej_reason = None if is_ok else (
                            f"Feasibility score {score:.2f} < {min_feasibility_threshold:.2f} "
                            f"(harmonic={best_harm:.2f}, vocal_clearance={vocal_clearance:.2f})."
                        )
                        best_plan = ComposedBridgePlan(
                            is_viable=is_ok,
                            transition_mode=mode if is_ok else "REJECTED",
                            archetype=arch if is_ok else "REJECTED_FALLBACK",
                            launchpad_phrase_a=p_a,
                            impact_phrase_b=p_b,
                            hypothesis_a=h_a,
                            hypothesis_b=h_b,
                            pitch_shift_semitones_b=best_shift,
                            breath_beats=breath_beats if is_ok else 0,
                            feasibility_score=float(np.clip(score, 0.0, 1.0)),
                            rejection_reason=rej_reason,
                            composer_rationale=rationale if is_ok else f"Transition rejected: {rej_reason}",
                        )

    assert best_plan is not None
    return best_plan


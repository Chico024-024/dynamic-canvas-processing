"""Downbeat-to-downbeat phase alignment for seamless transitions.

This module calculates the cue offset for Song B so that its incoming
downbeat aligns with sub-millisecond precision to Song A's outgoing
downbeat at the transition start.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from ..models import AudioBuffer
from .context import TransitionContext
from .plan import TransitionParams


@dataclass(frozen=True)
class PhaseLockResult:
    """Result of phase lock calculation."""

    success: bool
    """Whether phase lock was successfully achieved."""

    cue_offset_b_s: float
    """Offset in seconds to apply to Song B start for phase alignment."""

    anchor_downbeat_a_s: float
    """The Song A downbeat that serves as the transition anchor."""

    anchor_downbeat_b_s: float
    """The Song B downbeat that will align with anchor_downbeat_a_s."""

    lead_in_duration_s: float
    """Duration from Song B's anchor downbeat to the transition start."""

    bars_overlap: int
    """Number of overlapping bars in the transition."""

    confidence: float
    """Confidence in the phase lock (0.0 to 1.0)."""

    transition_start_a_s: float = 0.0
    """The start time of the transition in Song A (and the output timeline).
    Equal to anchor_downbeat_a_s - lead_in_duration_s.
    Guarantees that transition_start_a_s + (anchor_downbeat_b_s - cue_offset_b_s) == anchor_downbeat_a_s.
    """

    # Failure telemetry (populated when success=False)
    fallback_reason: str | None = None
    """Reason why phase lock failed (e.g., 'insufficient_downbeats', 'low_confidence', 'invalid_cue_offset')."""

    predicted_drift_ms: float | None = None
    """Predicted phase drift in milliseconds over the transition window if no alignment applied."""

    tempo_delta_bpm: float | None = None
    """Absolute tempo difference between Song A and Song B in BPM."""

    def __bool__(self) -> bool:
        """Return True if phase lock was successfully achieved."""
        return self.success


def _failure_result(
    reason: str,
    context: TransitionContext | None = None,
) -> PhaseLockResult:
    tempo_delta = None
    if context is not None:
        t_a = getattr(context.analysis_a, "tempo_bpm", None)
        t_b = getattr(context.analysis_b, "tempo_bpm", None)
        if t_a is not None and t_b is not None:
            tempo_delta = abs(float(t_a) - float(t_b))
    return PhaseLockResult(
        success=False,
        cue_offset_b_s=0.0,
        anchor_downbeat_a_s=0.0,
        anchor_downbeat_b_s=0.0,
        lead_in_duration_s=0.0,
        bars_overlap=0,
        confidence=0.0,
        transition_start_a_s=0.0,
        fallback_reason=reason,
        predicted_drift_ms=None,
        tempo_delta_bpm=tempo_delta,
    )


def calculate_phase_locked_cue_offset(
    context: TransitionContext,
    params: TransitionParams | None = None,
    lead_in_bars: int = 2,
    min_confidence: float = 0.5,
) -> PhaseLockResult:
    """
    Calculate the cue offset for Song B to achieve downbeat-to-downbeat phase lock.

    The goal: Song B's incoming downbeat (at its anchor_bar_b) should land
    exactly on Song A's outgoing downbeat (at its anchor_bar_a) when the
    transition starts.

    This function can work in two modes:
    1. If params with anchor_bar_a and anchor_bar_b are provided, use those explicitly
    2. If not, auto-select the best outgoing downbeat in Song A near the
       transition region and find a corresponding incoming downbeat in Song B.

    Args:
        context: TransitionContext with analyses for both songs.
        params: Optional TransitionParams with anchor bar specifications.
        lead_in_bars: Number of bars of lead-in for Song B before the transition.
        min_confidence: Minimum confidence threshold for downbeat detection.

    Returns:
        PhaseLockResult with cue offset and metadata. The `success` field indicates
        whether phase lock was achieved. On failure, telemetry fields describe why.
    """
    # Require downbeat data for both tracks
    downbeats_a = list(context.analysis_a.downbeat_times_s)
    downbeats_b = list(context.analysis_b.downbeat_times_s)

    if not downbeats_a or not downbeats_b:
        return PhaseLockResult(
            success=False,
            cue_offset_b_s=0.0,
            anchor_downbeat_a_s=0.0,
            anchor_downbeat_b_s=0.0,
            lead_in_duration_s=0.0,
            bars_overlap=0,
            confidence=0.0,
            fallback_reason="insufficient_downbeats",
            predicted_drift_ms=None,
            tempo_delta_bpm=None,
        )

    # Check confidence
    if context.analysis_a.tempo_confidence < min_confidence:
        return PhaseLockResult(
            success=False,
            cue_offset_b_s=0.0,
            anchor_downbeat_a_s=0.0,
            anchor_downbeat_b_s=0.0,
            lead_in_duration_s=0.0,
            bars_overlap=0,
            confidence=0.0,
            fallback_reason="low_confidence",
            predicted_drift_ms=None,
            tempo_delta_bpm=None,
        )
    if context.analysis_b.tempo_confidence < min_confidence:
        return PhaseLockResult(
            success=False,
            cue_offset_b_s=0.0,
            anchor_downbeat_a_s=0.0,
            anchor_downbeat_b_s=0.0,
            lead_in_duration_s=0.0,
            bars_overlap=0,
            confidence=0.0,
            fallback_reason="low_confidence",
            predicted_drift_ms=None,
            tempo_delta_bpm=None,
        )

    # Check if params specifies explicit anchor bars
    use_explicit_anchors = params is not None and params.anchor_bar_a is not None and params.anchor_bar_b is not None

    if use_explicit_anchors:
        # Use explicitly specified anchor bars (for testing/backward compatibility)
        anchor_bar_a = params.anchor_bar_a
        anchor_bar_b = params.anchor_bar_b

        downbeats_a = list(context.analysis_a.downbeat_times_s)
        downbeats_b = list(context.analysis_b.downbeat_times_s)

        if anchor_bar_a - 1 >= len(downbeats_a) or anchor_bar_a < 1:
            return PhaseLockResult(
                success=False,
                cue_offset_b_s=0.0,
                anchor_downbeat_a_s=0.0,
                anchor_downbeat_b_s=0.0,
                lead_in_duration_s=0.0,
                bars_overlap=0,
                confidence=0.0,
                fallback_reason="invalid_anchor_bar_a",
                predicted_drift_ms=None,
                tempo_delta_bpm=None,
            )
        if anchor_bar_b - 1 >= len(downbeats_b) or anchor_bar_b < 1:
            return PhaseLockResult(
                success=False,
                cue_offset_b_s=0.0,
                anchor_downbeat_a_s=0.0,
                anchor_downbeat_b_s=0.0,
                lead_in_duration_s=0.0,
                bars_overlap=0,
                confidence=0.0,
                fallback_reason="invalid_anchor_bar_b",
                predicted_drift_ms=None,
                tempo_delta_bpm=None,
            )

        anchor_downbeat_a = downbeats_a[anchor_bar_a - 1]
        anchor_downbeat_b = downbeats_b[anchor_bar_b - 1]

        # Calculate lead-in duration using the lead_in_bars parameter
        tempo_b = context.analysis_b.tempo_bpm
        if tempo_b is None or tempo_b <= 0:
            return _failure_result("invalid_tempo_b", context)

        beat_duration_b = 60.0 / tempo_b
        bar_duration_b = beat_duration_b * 4  # Assuming 4/4 meter
        lead_in_duration_s = lead_in_bars * bar_duration_b

        # Phase-lock timeline geometry:
        # Both tracks must have identical elapsed duration from the transition start
        # to their respective anchor downbeats:
        # transition_start_a = anchor_downbeat_a - lead_in_duration_s
        # cue_offset_b = anchor_downbeat_b - lead_in_duration_s
        # Output downbeat B time = transition_start_a + (anchor_downbeat_b - cue_offset_b)
        #                        = (anchor_downbeat_a - lead_in) + lead_in == anchor_downbeat_a
        transition_start_in_a = anchor_downbeat_a - lead_in_duration_s
        cue_offset_b = anchor_downbeat_b - lead_in_duration_s

        # Validate cue offset - must be non-negative and within Song B
        if cue_offset_b < 0:
            return _failure_result("negative_cue_offset", context)
        if cue_offset_b >= context.song_b.duration_s:
            return _failure_result("cue_offset_exceeds_song_b", context)

        # Validate transition start - must be non-negative and within Song A
        if transition_start_in_a < 0 or transition_start_in_a >= context.song_a.duration_s:
            return _failure_result("transition_start_out_of_bounds", context)

        # Calculate confidence based on downbeat detection quality
        confidence = min(
            context.analysis_a.tempo_confidence,
            context.analysis_b.tempo_confidence,
        )

        # Calculate tempo delta for telemetry
        tempo_a = context.analysis_a.tempo_bpm
        tempo_delta = abs(tempo_a - tempo_b) if (tempo_a is not None and tempo_b is not None) else None

        # Overlap duration is from transition start to the end of Song A
        overlap_duration_a = context.song_a.duration_s - transition_start_in_a
        overlap_bars = max(1, int(round(overlap_duration_a / bar_duration_b)))

        return PhaseLockResult(
            success=True,
            cue_offset_b_s=cue_offset_b,
            anchor_downbeat_a_s=anchor_downbeat_a,
            anchor_downbeat_b_s=anchor_downbeat_b,
            lead_in_duration_s=lead_in_duration_s,
            bars_overlap=overlap_bars,
            confidence=confidence,
            transition_start_a_s=transition_start_in_a,
            fallback_reason=None,
            predicted_drift_ms=None,
            tempo_delta_bpm=tempo_delta,
        )
    else:
        # Auto-select mode: find best downbeats near transition region
        return _calculate_auto_phase_lock(context, lead_in_bars=lead_in_bars, min_confidence=min_confidence)


def _calculate_auto_phase_lock(
    context: TransitionContext,
    lead_in_bars: int = 2,
    min_confidence: float = 0.5,
) -> PhaseLockResult:
    """
    Auto-select the best downbeats for phase locking.

    Finds the Song A downbeat nearest to the transition start region
    and a corresponding Song B downbeat that can be cued with the
    specified lead-in to achieve phase alignment.
    """
    downbeats_a = list(context.analysis_a.downbeat_times_s)
    downbeats_b = list(context.analysis_b.downbeat_times_s)

    if not downbeats_a or not downbeats_b:
        return _failure_result("insufficient_downbeats", context)

    # Check confidence
    if context.analysis_a.tempo_confidence < min_confidence or context.analysis_b.tempo_confidence < min_confidence:
        return _failure_result("low_confidence", context)

    # Determine the target transition start time in Song A
    target_transition_start = context.song_a.duration_s - context.requested_duration_s

    tempo_a = context.analysis_a.tempo_bpm
    if tempo_a is None or tempo_a <= 0:
        return _failure_result("invalid_tempo_a", context)

    tempo_b = context.analysis_b.tempo_bpm
    if tempo_b is None or tempo_b <= 0:
        return _failure_result("invalid_tempo_b", context)

    beat_duration_b = 60.0 / tempo_b
    bar_duration_b = beat_duration_b * 4  # Assuming 4/4 meter
    lead_in_duration_s = lead_in_bars * bar_duration_b

    # Target anchor in Song A:
    # Since transition_start_in_a = anchor_downbeat_a - lead_in_duration_s,
    # the ideal anchor_downbeat_a is target_transition_start + lead_in_duration_s
    target_anchor_a = target_transition_start + lead_in_duration_s

    # Find downbeats in Song A that leave room for lead-in and occur before track ends
    candidate_downbeats_a = [
        d for d in downbeats_a
        if d >= lead_in_duration_s and d < context.song_a.duration_s
    ]

    if not candidate_downbeats_a:
        # Fallback to any valid downbeats before end of Song A
        fallback_candidates = [d for d in downbeats_a if 0 < d < context.song_a.duration_s]
        if not fallback_candidates:
            return _failure_result("insufficient_downbeats", context)
        anchor_downbeat_a = min(fallback_candidates, key=lambda d: abs(d - target_anchor_a))
        # Adapt lead-in duration if necessary
        lead_in_duration_s = min(lead_in_duration_s, anchor_downbeat_a)
    else:
        anchor_downbeat_a = min(candidate_downbeats_a, key=lambda d: abs(d - target_anchor_a))

    # The transition starts lead_in_duration_s before anchor_downbeat_a
    transition_start_in_a = anchor_downbeat_a - lead_in_duration_s

    # Validate transition start is within track bounds
    if transition_start_in_a < 0 or transition_start_in_a >= context.song_a.duration_s:
        return _failure_result("transition_start_out_of_bounds", context)

    # Find candidate downbeats in Song B
    candidate_downbeats_b = [d for d in downbeats_b if d >= lead_in_duration_s]

    if not candidate_downbeats_b:
        if downbeats_b:
            last_downbeat_b = downbeats_b[-1]
            bars_forward = max(1, int(math.ceil((lead_in_duration_s - last_downbeat_b) / bar_duration_b)))
            extrapolated_downbeat_b = last_downbeat_b + bars_forward * bar_duration_b
            if 0 <= (extrapolated_downbeat_b - lead_in_duration_s) < context.song_b.duration_s:
                candidate_downbeats_b = [extrapolated_downbeat_b]
            else:
                return _failure_result("no_valid_anchor_b", context)
        else:
            return _failure_result("insufficient_downbeats", context)

    valid_candidates_b = [
        d for d in candidate_downbeats_b
        if 0 <= d - lead_in_duration_s < context.song_b.duration_s
    ]

    if not valid_candidates_b:
        return _failure_result("no_valid_candidates_b", context)

    # Pick the candidate that gives the most natural lead-in
    anchor_downbeat_b = min(
        valid_candidates_b,
        key=lambda d: abs((d - lead_in_duration_s) - lead_in_duration_s)
    )

    cue_offset_b = anchor_downbeat_b - lead_in_duration_s
    if cue_offset_b < 0:
        return _failure_result("negative_cue_offset", context)
    if cue_offset_b >= context.song_b.duration_s:
        return _failure_result("cue_offset_exceeds_song_b", context)

    confidence = min(
        context.analysis_a.tempo_confidence,
        context.analysis_b.tempo_confidence,
    )

    tempo_delta = abs(tempo_a - tempo_b)
    overlap_duration_a = context.song_a.duration_s - transition_start_in_a
    overlap_bars = max(1, int(round(overlap_duration_a / bar_duration_b)))

    return PhaseLockResult(
        success=True,
        cue_offset_b_s=cue_offset_b,
        anchor_downbeat_a_s=anchor_downbeat_a,
        anchor_downbeat_b_s=anchor_downbeat_b,
        lead_in_duration_s=lead_in_duration_s,
        bars_overlap=overlap_bars,
        confidence=confidence,
        transition_start_a_s=transition_start_in_a,
        fallback_reason=None,
        predicted_drift_ms=None,
        tempo_delta_bpm=tempo_delta,
    )


def apply_phase_lock_to_context(
    context: TransitionContext,
    lock_result: PhaseLockResult,
    params: TransitionParams | None = None,
) -> tuple[TransitionContext, TransitionParams]:
    """
    Apply phase lock result to a TransitionContext and update params.

    Returns a new context with adjusted Song B timing and updated params.
    """
    # The cue offset tells us when to start Song B relative to its own timeline
    # The transition start in the output timeline is lock_result.transition_start_a_s
    # so that Song B's anchor downbeat coincides with Song A's anchor downbeat.
    new_start_time = lock_result.transition_start_a_s

    # Create updated params with the calculated cue offset
    if params is None:
        params = TransitionParams()

    updated_params = replace(params, cue_offset_b_s=lock_result.cue_offset_b_s)

    # Create a new context with the adjusted timing
    # Note: We don't modify the original context; the caller uses the
    # cue_offset_b_s and new_start_time to render correctly
    return context, updated_params


__all__ = [
    "PhaseLockResult",
    "calculate_phase_locked_cue_offset",
    "apply_phase_lock_to_context",
]
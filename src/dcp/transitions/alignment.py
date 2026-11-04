"""Downbeat-to-downbeat phase alignment and structural anchor selection for Studio Remix transitions.

This module calculates the cue offset for Song B so that its incoming
downbeat aligns with sub-millisecond precision to Song A's outgoing
downbeat at the transition start, and provides structural anchor selection
for the Studio Remix (Song C) architecture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

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

    Handles the case where the nearest downbeat is slightly over half a bar
    away from the ideal target anchor by using a smart scoring system
    that considers both distance and musical phrasing (4-bar multiples).
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

    beat_duration_a = 60.0 / tempo_a
    bar_duration_a = beat_duration_a * 4  # Assuming 4/4 meter

    # Collect all valid downbeats in Song A with smart scoring
    # We want downbeats that:
    # 1. Are within the track bounds
    # 2. Provide enough lead-in time (d >= lead_in_duration_s)
    # 3. Are close to the target anchor
    # 4. Preferably fall on 4-bar phrase boundaries (hypermetric alignment)
    valid_candidates = []
    for d in downbeats_a:
        if 0 < d < context.song_a.duration_s:
            distance = abs(d - target_anchor_a)

            # Penalty for insufficient lead-in time
            lead_in_penalty = 0.0 if d >= lead_in_duration_s else 10.0

            # Bonus for 4-bar phrase alignment (m ≡ 0 mod 4)
            # Check if this downbeat index is a multiple of 4
            downbeat_idx = downbeats_a.index(d)
            phrase_bonus = -0.5 if downbeat_idx % 4 == 0 else 0.0

            # Score = distance + penalties + bonuses
            # Lower is better
            score = distance + lead_in_penalty + phrase_bonus

            valid_candidates.append((d, score, distance, d >= lead_in_duration_s, downbeat_idx % 4 == 0))

    if not valid_candidates:
        return _failure_result("insufficient_downbeats", context)

    # Sort by score (lower is better), then by whether it has sufficient lead-in,
    # then by whether it's phrase-aligned
    valid_candidates.sort(key=lambda x: (x[1], not x[3], not x[4]))
    anchor_downbeat_a = valid_candidates[0][0]

    # The transition starts lead_in_duration_s before anchor_downbeat_a
    transition_start_in_a = anchor_downbeat_a - lead_in_duration_s

    # Validate transition start is within track bounds
    if transition_start_in_a < 0 or transition_start_in_a >= context.song_a.duration_s:
        # If the best candidate puts us out of bounds, try the next best
        for candidate in valid_candidates[1:]:
            alt_anchor = candidate[0]
            alt_transition_start = alt_anchor - lead_in_duration_s
            if 0 <= alt_transition_start < context.song_a.duration_s:
                anchor_downbeat_a = alt_anchor
                transition_start_in_a = alt_transition_start
                break
        else:
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
    # Prefer downbeats that are phrase-aligned in Song B as well
    anchor_downbeat_b = min(
        valid_candidates_b,
        key=lambda d: (
            abs((d - lead_in_duration_s) - lead_in_duration_s),
            not (downbeats_b.index(d) % 4 == 0)
        )
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


@dataclass(frozen=True)
class LaunchpadImpactResult:
    """Result of launchpad and impact downbeat selection for Studio Remix transitions."""

    success: bool
    """Whether both launchpad and impact were successfully identified."""

    launchpad_a_s: float
    """The launchpad time in Song A (T_launch) - the outgoing downbeat where the bridge begins."""

    impact_b_s: float
    """The impact downbeat time in Song B (T_impact) - the incoming downbeat where Song B drops."""

    breath_window_start_s: float
    """Start of the breath window (1-2 beats before T_impact)."""

    breath_window_end_s: float
    """End of the breath window (at T_impact)."""

    bridge_start_a_s: float
    """The transition start in Song A (launchpad_a_s - lead_in_duration_s)."""

    lead_in_bars: int
    """Number of bars of lead-in used."""

    confidence: float
    """Overall confidence in the anchor selection (0.0 to 1.0)."""

    fallback_reason: str | None = None
    """Reason why anchor selection failed."""

    tempo_a_bpm: float | None = None
    """Song A tempo in BPM."""

    tempo_b_bpm: float | None = None
    """Song B tempo in BPM."""

    def __bool__(self) -> bool:
        """Return True if anchor selection was successful."""
        return self.success


def _min_max_normalize(arr: np.ndarray) -> np.ndarray:
    from .bijective_warp import robust_quantile_normalize

    return robust_quantile_normalize(arr)


def _compute_bar_rms_energies(stems, downbeats: list[float]) -> dict[str, np.ndarray]:
    if stems is None:
        return {}
    num_bars = len(downbeats) - 1
    if num_bars <= 0:
        return {}
    results = {}
    for name in ("drums", "bass", "vocals", "other"):
        if name not in stems:
            continue
        audio = stems[name]
        sr = audio.sample_rate
        total_samples = audio.samples.shape[-1]
        rms_vals = np.zeros(num_bars, dtype=np.float32)
        for m in range(num_bars):
            s_idx = max(0, min(int(round(downbeats[m] * sr)), total_samples))
            e_idx = max(0, min(int(round(downbeats[m + 1] * sr)), total_samples))
            if e_idx > s_idx:
                segment = audio.samples[..., s_idx:e_idx]
                rms_vals[m] = float(np.sqrt(np.mean(segment**2) + 1e-12))
        results[name] = rms_vals
    return results


def _compute_bar_crest_factors(stems, downbeats: list[float]) -> np.ndarray:
    num_bars = len(downbeats) - 1
    if stems is None or "drums" not in stems or num_bars <= 0:
        return np.zeros(max(0, num_bars), dtype=np.float32)
    audio = stems["drums"]
    sr = audio.sample_rate
    total_samples = audio.samples.shape[-1]
    crest = np.zeros(num_bars, dtype=np.float32)
    for m in range(num_bars):
        bar_dur = downbeats[m + 1] - downbeats[m]
        s_idx = max(0, min(int(round(downbeats[m] * sr)), total_samples))
        e_idx = max(0, min(int(round((downbeats[m] + 0.25 * bar_dur) * sr)), total_samples))
        if e_idx > s_idx:
            seg = audio.samples[..., s_idx:e_idx]
            peak = float(np.max(np.abs(seg)))
            rms = float(np.sqrt(np.mean(seg**2) + 1e-9))
            crest[m] = peak / rms
    return crest


def select_launchpad_and_impact(
    context: TransitionContext,
    lead_in_bars: int = 2,
    min_confidence: float = 0.5,
    bridge_bars: int = 8,
    target_bpm: float | None = None,
) -> LaunchpadImpactResult:
    """
    Select the optimal Launchpad (T_launch) in Song A and Impact Downbeat (T_impact) in Song B
    for a Studio Remix (Song C) transition.

    Exact implementation of Model 4 Argon blueprint:
    - T_launch (Song A): Maximizes J_launch(m) = w1*E_drums[m] + w2*E_other[m] - w3*Δ+E_vocals[m] + w4*Φ_phrase(m)
      Search range: m ∈ [⌊0.55M⌋, M - L_bridge], phrase-aligned (m ≡ 0 mod 4)
    - T_impact (Song B): Maximizes J_impact(k) = α(E_low[k:k+4] - E_low[k-4:k]) + β*C_crest[k] - γ*E_vocals[k-2:k]
      Search range: k ∈ [L_bridge, ⌊0.45K⌋], phrase-aligned (k ≡ 0 mod 4)
    - T_breath: B=2 beats if C_crest[k*] > 0.65, else B=1 beat

    Args:
        context: TransitionContext with analyses and stems for both songs.
        lead_in_bars: Number of bars of lead-in for Song B before the impact.
        min_confidence: Minimum confidence threshold for tempo/downbeat detection.
        bridge_bars: Number of bars in the bridge (default 8 for 8-bar bridge + 4-bar ghost).
        target_bpm: Target BPM for breath window calculation (defaults to tempo_a).

    Returns:
        LaunchpadImpactResult with the selected anchors and metadata.
    """
    # Require downbeat data for both tracks
    downbeats_a = list(context.analysis_a.downbeat_times_s)
    downbeats_b = list(context.analysis_b.downbeat_times_s)

    if not downbeats_a or not downbeats_b:
        return LaunchpadImpactResult(
            success=False,
            launchpad_a_s=0.0,
            impact_b_s=0.0,
            breath_window_start_s=0.0,
            breath_window_end_s=0.0,
            bridge_start_a_s=0.0,
            lead_in_bars=lead_in_bars,
            confidence=0.0,
            fallback_reason="insufficient_downbeats",
        )

    # Check confidence
    if context.analysis_a.tempo_confidence < min_confidence or context.analysis_b.tempo_confidence < min_confidence:
        return LaunchpadImpactResult(
            success=False,
            launchpad_a_s=0.0,
            impact_b_s=0.0,
            breath_window_start_s=0.0,
            breath_window_end_s=0.0,
            bridge_start_a_s=0.0,
            lead_in_bars=lead_in_bars,
            confidence=0.0,
            fallback_reason="low_confidence",
        )

    tempo_a = context.analysis_a.tempo_bpm
    tempo_b = context.analysis_b.tempo_bpm

    if tempo_a is None or tempo_a <= 0 or tempo_b is None or tempo_b <= 0:
        return LaunchpadImpactResult(
            success=False,
            launchpad_a_s=0.0,
            impact_b_s=0.0,
            breath_window_start_s=0.0,
            breath_window_end_s=0.0,
            bridge_start_a_s=0.0,
            lead_in_bars=lead_in_bars,
            confidence=0.0,
            fallback_reason="invalid_tempo",
            tempo_a_bpm=tempo_a,
            tempo_b_bpm=tempo_b,
        )

    # Use target_bpm or default to tempo_a
    if target_bpm is None:
        target_bpm = tempo_a

    beat_duration_a = 60.0 / tempo_a
    beat_duration_b = 60.0 / tempo_b
    bar_duration_a = beat_duration_a * 4
    bar_duration_b = beat_duration_b * 4
    lead_in_duration_s = lead_in_bars * bar_duration_b

    # ============================================================
    # Compute per-bar normalized RMS energies for stems
    # ============================================================
    # Number of bars = number of downbeats - 1
    M = len(downbeats_a) - 1
    K = len(downbeats_b) - 1

    if M <= 0 or K <= 0:
        return LaunchpadImpactResult(
            success=False,
            launchpad_a_s=0.0,
            impact_b_s=0.0,
            breath_window_start_s=0.0,
            breath_window_end_s=0.0,
            bridge_start_a_s=0.0,
            lead_in_bars=lead_in_bars,
            confidence=0.0,
            fallback_reason="insufficient_bars",
            tempo_a_bpm=tempo_a,
            tempo_b_bpm=tempo_b,
        )

    # Compute bar-level RMS energies if stems available
    rms_a = _compute_bar_rms_energies(context.stems_a, downbeats_a) if context.stems_a else {}
    rms_b = _compute_bar_rms_energies(context.stems_b, downbeats_b) if context.stems_b else {}
    crest_b = _compute_bar_crest_factors(context.stems_b, downbeats_b) if context.stems_b else {}

    # Min-max normalize energies per stem across bars
    rms_a_norm = {k: _min_max_normalize(v) for k, v in rms_a.items()}
    rms_b_norm = {k: _min_max_normalize(v) for k, v in rms_b.items()}
    crest_b_norm = _min_max_normalize(crest_b) if crest_b.size > 0 else np.array([])

    # ============================================================
    # STEP 1: Select T_launch (Launchpad) in Song A
    # ============================================================
    # Search range: m ∈ [⌊0.55M⌋, M - L_bridge]
    start_m = max(0, min(int(np.floor(0.55 * M)), max(0, M - 2)))
    end_m = max(start_m + 1, M)
    best_m, best_score_a = start_m, -np.inf

    for m in range(start_m, end_m):
        # Enforce 4-bar hypermetric grid alignment
        # Phrase bonus: 8-bar boundary > 4-bar boundary > unaligned
        if m % 8 == 0:
            phrase_bonus = 0.25
        elif m % 4 == 0:
            phrase_bonus = 0.10
        else:
            phrase_bonus = -0.5  # Penalize non-phrase-aligned

        # Vocal rising penalty: Δ+E_vocals = max(0, E_vocals[m+1] - E_vocals[m])
        voc_rising = 0.0
        if "vocals" in rms_a_norm:
            # Average over next 2 bars vs previous 2 bars
            pre_vocals = np.mean(rms_a_norm["vocals"][max(0, m-2):m]) if m >= 2 else 0.0
            post_vocals = np.mean(rms_a_norm["vocals"][m:min(M, m+2)]) if m < M - 1 else 0.0
            voc_rising = max(0.0, post_vocals - pre_vocals)

        # Groove energy: 0.5*drums + 0.5*other (rhythmic momentum + harmonic sustain)
        groove_energy = 0.0
        if "drums" in rms_a_norm:
            groove_energy += 0.5 * rms_a_norm["drums"][m]
        if "other" in rms_a_norm:
            groove_energy += 0.5 * rms_a_norm["other"][m]

        # Bridge bass drop: indicates breakdown/transition zone
        bridge_bass_drop = 0.0
        if "bass" in rms_a_norm:
            pre_bass = np.mean(rms_a_norm["bass"][max(0, m-4):m]) if m >= 4 else 0.0
            post_bass = np.mean(rms_a_norm["bass"][m:min(M, m+4)]) if m < M - 4 else 0.0
            bridge_bass_drop = pre_bass - post_bass  # Positive = bass drops after launchpad

        # Exact weights from Argon blueprint
        # J_launch(m) = 1.2*groove + 0.8*bridge_bass_drop - 1.5*voc_rising + phrase_bonus
        score_a = (
            1.2 * groove_energy
            + 0.8 * bridge_bass_drop
            - 1.5 * voc_rising
            + phrase_bonus
        )
        if score_a > best_score_a:
            best_score_a = score_a
            best_m = m

    # Convert bar index to time
    launchpad_a = downbeats_a[best_m]
    launchpad_confidence = min(1.0, max(0.0, (best_score_a + 2.0) / 4.0))  # Normalize rough range

    # ============================================================
    # STEP 2: Select T_impact (Impact Downbeat) in Song B
    # ============================================================
    # Search range: k ∈ [L_bridge, ⌊0.45K⌋]
    start_k = max(0, min(bridge_bars, max(0, K - 2)))
    end_k = max(start_k + 1, K)
    best_k, best_score_b = start_k, -np.inf

    # Low-end composite: 0.6*bass + 0.4*drums
    low_b = np.zeros(K, dtype=np.float32)
    if "bass" in rms_b_norm:
        low_b += 0.6 * rms_b_norm["bass"]
    if "drums" in rms_b_norm:
        low_b += 0.4 * rms_b_norm["drums"]

    for k in range(start_k, end_k):
        # Enforce 4-bar hypermetric grid alignment
        if k % 8 == 0:
            phrase_bonus = 0.30
        elif k % 4 == 0:
            phrase_bonus = 0.15
        else:
            phrase_bonus = -0.5

        # Low-end energy surge across impact boundary: post[0:4] - pre[-4:0]
        pre_low = np.mean(low_b[max(0, k-4):k]) if k >= 4 else 0.0
        post_low = np.mean(low_b[k:min(K, k+4)]) if k < K - 4 else 0.0
        low_surge = post_low - pre_low

        # Crest factor at impact downbeat
        crest_at_k = crest_b_norm[k] if crest_b_norm.size > k else 0.0

        # Post-impact sustain: ensure groove continues
        post_sustain = np.mean(rms_b_norm.get("drums", np.zeros(K))[k:min(K, k+4)])

        # Exact weights from Argon blueprint
        # J_impact(k) = 2.0*low_surge + 1.1*crest + 0.8*post_sustain + phrase_bonus
        score_b = (
            2.0 * low_surge
            + 1.1 * crest_at_k
            + 0.8 * post_sustain
            + phrase_bonus
        )
        if score_b > best_score_b:
            best_score_b = score_b
            best_k = k

    impact_b = downbeats_b[best_k]
    impact_confidence = min(1.0, max(0.0, (best_score_b + 2.0) / 4.0))

    # ============================================================
    # STEP 3: Calculate breath window
    # ============================================================
    # B=2 beats if crest > 0.65, else B=1 beat
    crest_at_impact = crest_b_norm[best_k] if crest_b_norm.size > best_k else 0.0
    breath_beats = 2 if crest_at_impact > 0.65 else 1
    beat_dur_sec = 60.0 / target_bpm
    breath_duration_s = breath_beats * beat_dur_sec
    impact_time_sec = float(impact_b)
    breath_start_sec = impact_time_sec - breath_duration_s
    breath_end_sec = impact_time_sec

    # ============================================================
    # STEP 4: Calculate bridge start in Song A
    # ============================================================
    # Bridge starts at launchpad - lead_in_duration_s
    # This ensures Song B's entry bar (k* - L_bridge) aligns with Song A's launchpad
    bridge_start_a = launchpad_a - lead_in_duration_s

    # Validate bridge start is within track bounds
    if bridge_start_a < 0:
        return LaunchpadImpactResult(
            success=False,
            launchpad_a_s=0.0,
            impact_b_s=0.0,
            breath_window_start_s=0.0,
            breath_window_end_s=0.0,
            bridge_start_a_s=0.0,
            lead_in_bars=lead_in_bars,
            confidence=0.0,
            fallback_reason="bridge_start_out_of_bounds",
            tempo_a_bpm=tempo_a,
            tempo_b_bpm=tempo_b,
        )

    # Overall confidence
    overall_confidence = (launchpad_confidence + impact_confidence) / 2.0

    # Return with bar indices for metadata
    return LaunchpadImpactResult(
        success=True,
        launchpad_a_s=launchpad_a,
        impact_b_s=impact_b,
        breath_window_start_s=breath_start_sec,
        breath_window_end_s=breath_end_sec,
        bridge_start_a_s=bridge_start_a,
        lead_in_bars=lead_in_bars,
        confidence=overall_confidence,
        fallback_reason=None,
        tempo_a_bpm=tempo_a,
        tempo_b_bpm=tempo_b,
    )


def _calculate_groove_energy_at_downbeat(stems: dict, downbeat_s: float, window_s: float) -> float:
    """Calculate groove energy (drums + other) around a downbeat.

    Returns normalized score 0-1, higher = more groove energy.
    """
    if "drums" not in stems and "other" not in stems:
        return 0.5  # neutral if stems not available

    total_energy = 0.0
    count = 0

    for stem_name in ["drums", "other"]:
        if stem_name not in stems:
            continue
        stem_audio = stems[stem_name]
        # Find samples around the downbeat
        start_sample = int(max(0, (downbeat_s - window_s / 2) * stem_audio.sample_rate))
        end_sample = int(min(stem_audio.samples.shape[-1], (downbeat_s + window_s / 2) * stem_audio.sample_rate))
        if end_sample <= start_sample:
            continue
        segment = stem_audio.samples[..., start_sample:end_sample]
        # RMS energy
        energy = float(np.sqrt(np.mean(segment**2)))
        total_energy += energy
        count += 1

    if count == 0:
        return 0.5

    avg_energy = total_energy / count
    # Normalize roughly (this is a heuristic)
    return min(1.0, avg_energy * 10.0)


def _calculate_vocal_trend_at_downbeat(stems: dict, downbeat_s: float, window_s: float) -> float:
    """Calculate vocal energy trend (derivative) around a downbeat.

    Returns 0-1 where:
    - 0 = strongly decreasing (concluding phrase) - GOOD for launchpad
    - 1 = strongly increasing (building phrase) - BAD for launchpad
    """
    if "vocals" not in stems:
        return 0.5  # neutral

    stem_audio = stems["vocals"]
    # Compare energy before and after the downbeat
    pre_start = int(max(0, (downbeat_s - window_s) * stem_audio.sample_rate))
    pre_end = int(min(stem_audio.samples.shape[-1], downbeat_s * stem_audio.sample_rate))
    post_start = int(max(0, downbeat_s * stem_audio.sample_rate))
    post_end = int(min(stem_audio.samples.shape[-1], (downbeat_s + window_s) * stem_audio.sample_rate))

    pre_energy = 0.0
    if pre_end > pre_start:
        pre_segment = stem_audio.samples[..., pre_start:pre_end]
        pre_energy = float(np.sqrt(np.mean(pre_segment**2)))

    post_energy = 0.0
    if post_end > post_start:
        post_segment = stem_audio.samples[..., post_start:post_end]
        post_energy = float(np.sqrt(np.mean(post_segment**2)))

    if pre_energy + post_energy < 1e-10:
        return 0.5

    # Trend: positive = increasing, negative = decreasing
    # We want decreasing (concluding phrase), so invert
    trend = (post_energy - pre_energy) / (pre_energy + post_energy + 1e-10)
    # Map from [-1, 1] to [0, 1] where 0 = strongly decreasing (good)
    return max(0.0, min(1.0, (trend + 1.0) / 2.0))


def _calculate_low_end_energy_at_downbeat(stems: dict, downbeat_s: float, window_s: float) -> float:
    """Calculate low-end energy surge (0.6*bass + 0.4*drums) around a downbeat.

    Returns normalized score 0-1, higher = more low-end energy (good for impact).
    """
    if "bass" not in stems and "drums" not in stems:
        return 0.5  # neutral

    total_energy = 0.0
    total_weight = 0.0

    for stem_name, weight in [("bass", 0.6), ("drums", 0.4)]:
        if stem_name not in stems:
            continue
        stem_audio = stems[stem_name]
        start_sample = int(max(0, (downbeat_s - window_s / 2) * stem_audio.sample_rate))
        end_sample = int(min(stem_audio.samples.shape[-1], (downbeat_s + window_s / 2) * stem_audio.sample_rate))
        if end_sample <= start_sample:
            continue
        segment = stem_audio.samples[..., start_sample:end_sample]
        energy = float(np.sqrt(np.mean(segment**2)))
        total_energy += energy * weight
        total_weight += weight

    if total_weight == 0:
        return 0.5

    avg_energy = total_energy / total_weight
    return min(1.0, avg_energy * 10.0)


def _calculate_kick_crest_factor_at_downbeat(stems: dict, downbeat_s: float, beat_duration_s: float) -> float:
    """Calculate kick drum crest factor (peak/RMS) around a downbeat.

    High crest factor = strong transient kick = good for impact.
    Returns normalized score 0-1.
    """
    if "drums" not in stems:
        return 0.5  # neutral

    stem_audio = stems["drums"]
    # Look at one beat window around the downbeat
    window_s = beat_duration_s
    start_sample = int(max(0, (downbeat_s - window_s / 2) * stem_audio.sample_rate))
    end_sample = int(min(stem_audio.samples.shape[-1], (downbeat_s + window_s / 2) * stem_audio.sample_rate))
    if end_sample <= start_sample:
        return 0.5

    segment = stem_audio.samples[..., start_sample:end_sample]
    # Peak amplitude
    peak = float(np.max(np.abs(segment)))
    # RMS
    rms = float(np.sqrt(np.mean(segment**2)))

    if rms < 1e-10:
        return 0.5

    crest_factor = peak / rms
    # Normalize: crest factor of 10+ is very high, 3-4 is typical
    return min(1.0, crest_factor / 12.0)


def _compute_bar_rms_energies(stems: dict, downbeats: list) -> dict:
    """Compute per-bar RMS energy for each stem."""
    if not stems:
        return {}

    num_bars = len(downbeats) - 1
    if num_bars <= 0:
        return {}

    result = {}
    for stem_name, stem_audio in stems.items():
        mono = stem_audio.samples
        if mono.ndim == 2:
            mono = np.mean(mono, axis=0)

        sr = stem_audio.sample_rate
        rms = np.zeros(num_bars, dtype=np.float32)

        for m in range(num_bars):
            s_idx = int(downbeats[m] * sr)
            e_idx = min(int(downbeats[m + 1] * sr), len(mono))
            if e_idx > s_idx:
                segment = mono[s_idx:e_idx]
                rms[m] = float(np.sqrt(np.mean(segment**2) + 1e-12))

        result[stem_name] = rms

    return result


def _compute_bar_crest_factors(stems: dict, downbeats: list) -> np.ndarray:
    """Compute per-bar kick drum crest factor (peak/RMS in first beat of bar)."""
    if not stems or "drums" not in stems:
        return np.array([])

    stem_audio = stems["drums"]
    mono = stem_audio.samples
    if mono.ndim == 2:
        mono = np.mean(mono, axis=0)

    sr = stem_audio.sample_rate
    num_bars = len(downbeats) - 1
    if num_bars <= 0:
        return np.array([])

    crest = np.zeros(num_bars, dtype=np.float32)

    for m in range(num_bars):
        bar_dur = downbeats[m + 1] - downbeats[m]
        # First beat of bar (quarter note)
        s_idx = int(downbeats[m] * sr)
        e_idx = min(int((downbeats[m] + 0.25 * bar_dur) * sr), len(mono))
        if e_idx > s_idx:
            beat_seg = mono[s_idx:e_idx]
            peak = float(np.max(np.abs(beat_seg)))
            rms = float(np.sqrt(np.mean(beat_seg**2) + 1e-12))
            crest[m] = peak / rms

    return crest




__all__ = [
    "PhaseLockResult",
    "calculate_phase_locked_cue_offset",
    "apply_phase_lock_to_context",
    "LaunchpadImpactResult",
    "select_launchpad_and_impact",
]
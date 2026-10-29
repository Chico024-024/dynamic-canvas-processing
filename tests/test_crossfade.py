"""Conventional crossfade behavior and safety tests."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from conftest import make_constant, make_tone
from dcp.config import DEFAULT_CONFIG
from dcp.models import RenderError
from dcp.rendering import render_crossfade


def test_crossfade_duration_and_samples():
    song_a = make_constant(0.25, duration_s=2.0)
    song_b = make_constant(-0.25, duration_s=2.0)
    result = render_crossfade(song_a, song_b, overlap_s=1.0)

    assert result.audio.sample_rate == 22050
    assert result.audio.channels == 2
    assert result.audio.duration_s == pytest.approx(3.0)
    assert np.all(np.isfinite(result.audio.samples))
    assert result.master_gain_db == 0.0

    overlap_frames = 22050
    start = song_a.frames - overlap_frames
    assert np.allclose(result.audio.samples[:, :start], 0.25)
    assert result.audio.samples[0, start] == pytest.approx(0.25)
    assert result.audio.samples[0, start + overlap_frames - 1] == pytest.approx(
        -0.25, abs=1e-4
    )
    assert np.allclose(result.audio.samples[:, start + overlap_frames :], -0.25)
    assert "Equal-power crossfade" in result.notes[0]


def test_crossfade_prevents_clipping_with_master_gain():
    song_a = make_constant(0.8, duration_s=2.0)
    song_b = make_constant(0.8, duration_s=2.0)
    result = render_crossfade(song_a, song_b, overlap_s=1.0)

    assert result.audio.peak <= 0.999 + 1e-9
    assert result.master_gain_db < 0
    assert any("prevent clipping" in note for note in result.notes)


def test_crossfade_can_allow_clipping_for_diagnostics():
    config = replace(
        DEFAULT_CONFIG,
        render=replace(DEFAULT_CONFIG.render, prevent_clipping=False),
    )
    result = render_crossfade(
        make_constant(0.8, duration_s=2.0),
        make_constant(0.8, duration_s=2.0),
        overlap_s=1.0,
        config=config,
    )
    assert result.audio.peak > 1.0
    assert result.master_gain_db == 0.0
    assert any("allowed" in note for note in result.notes)


def test_crossfade_adapts_sample_rate_and_channels():
    song_a = make_tone(duration_s=2.0, sample_rate=22050, channels=2)
    song_b = make_tone(
        duration_s=2.0, frequency=550.0, sample_rate=11025, channels=1
    )
    result = render_crossfade(song_a, song_b, overlap_s=1.0)

    assert result.audio.sample_rate == 22050
    assert result.audio.channels == 2
    assert result.audio.duration_s == pytest.approx(3.0, abs=0.02)
    assert any("resampled" in note for note in result.notes)
    assert any("channel" in note for note in result.notes)


def test_crossfade_mono_output():
    song_a = make_tone(duration_s=2.0, channels=1)
    song_b = make_tone(duration_s=2.0, channels=1)
    result = render_crossfade(song_a, song_b, overlap_s=0.5)
    assert result.audio.channels == 1


@pytest.mark.parametrize("overlap", [0.0, -1.0, math.nan, math.inf])
def test_crossfade_rejects_invalid_overlap(overlap):
    with pytest.raises(RenderError, match="positive finite"):
        render_crossfade(
            make_tone(duration_s=1.0),
            make_tone(duration_s=1.0),
            overlap_s=overlap,
        )


def test_crossfade_rejects_overlap_longer_than_tracks():
    with pytest.raises(RenderError, match="exceeds the shorter"):
        render_crossfade(
            make_tone(duration_s=1.0),
            make_tone(duration_s=0.5),
            overlap_s=0.75,
        )


def test_crossfade_overlap_equal_to_track_length_is_allowed():
    song = make_tone(duration_s=1.0)
    result = render_crossfade(song, song, overlap_s=1.0)
    assert result.audio.duration_s == pytest.approx(1.0)
    assert np.all(np.isfinite(result.audio.samples))

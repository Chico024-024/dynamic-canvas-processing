"""Synthetic material must be deterministic, valid, and exactly stem-summed."""

from __future__ import annotations

import numpy as np
import pytest

from dcp.synthetic import generate_synthetic_track, midi_to_hz


def test_generation_is_deterministic_for_same_seed():
    first = generate_synthetic_track(
        bpm=120, duration_s=3.0, sample_rate=11025, seed=42
    )
    second = generate_synthetic_track(
        bpm=120, duration_s=3.0, sample_rate=11025, seed=42
    )
    assert np.array_equal(first.mix.samples, second.mix.samples)
    for name in first.stems.names:
        assert np.array_equal(
            first.stems[name].samples, second.stems[name].samples
        )


def test_different_seeds_change_material():
    first = generate_synthetic_track(
        bpm=120, duration_s=3.0, sample_rate=11025, seed=1
    )
    second = generate_synthetic_track(
        bpm=120, duration_s=3.0, sample_rate=11025, seed=2
    )
    assert not np.array_equal(first.mix.samples, second.mix.samples)


def test_mix_equals_sum_of_stems():
    track = generate_synthetic_track(
        bpm=120, duration_s=4.0, sample_rate=22050, seed=3
    )
    stem_sum = np.sum(
        np.stack([track.stems[name].samples for name in track.stems.names]),
        axis=0,
    )
    assert np.allclose(track.mix.samples, stem_sum, atol=1e-12)


def test_format_and_peak_are_as_documented():
    track = generate_synthetic_track(
        bpm=120, duration_s=3.0, sample_rate=22050, seed=4
    )
    assert track.mix.sample_rate == 22050
    assert track.mix.channels == 2
    assert track.mix.frames == 3 * 22050
    assert abs(track.mix.peak - 0.65) < 1e-9
    assert track.stems.names == ("drums", "bass", "other", "vocals")
    metadata = track.metadata()
    assert metadata["synthetic"] is True
    assert metadata["stems"] == ["drums", "bass", "other", "vocals"]
    assert track.key_name == "A minor"


def test_vocal_stem_respects_lead_in_and_lead_out():
    lead_in = 0.5
    lead_out = 0.5
    track = generate_synthetic_track(
        bpm=120,
        duration_s=4.0,
        sample_rate=22050,
        seed=5,
        vocal_lead_in_s=lead_in,
        vocal_lead_out_s=lead_out,
    )
    vocals = track.stems["vocals"].samples
    assert np.max(np.abs(vocals[:, : int(0.4 * 22050)])) < 1e-6
    assert np.max(np.abs(vocals[:, -int(0.4 * 22050) :])) < 1e-6


def test_mono_generation_reduces_stems_to_one_channel():
    track = generate_synthetic_track(
        bpm=100, duration_s=3.0, sample_rate=11025, channels=1, seed=6
    )
    assert track.mix.channels == 1
    for name in track.stems.names:
        assert track.stems[name].channels == 1
    stem_sum = np.sum(
        np.stack([track.stems[name].samples for name in track.stems.names]),
        axis=0,
    )
    assert np.allclose(track.mix.samples, stem_sum, atol=1e-12)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"bpm": 30.0}, "bpm"),
        ({"duration_s": 1.0}, "duration_s"),
        ({"sample_rate": 4000}, "sample_rate"),
        ({"channels": 3}, "channels"),
        ({"root_midi": 200}, "root_midi"),
        ({"mode": "dorian"}, "mode"),
        ({"vocal_lead_in_s": -0.1}, "non-negative"),
        ({"vocal_lead_in_s": 1.5, "vocal_lead_out_s": 1.5}, "shorter"),
    ],
)
def test_invalid_parameters_raise_value_error(kwargs, message):
    arguments = {"duration_s": 3.0, **kwargs}
    with pytest.raises(ValueError, match=message):
        generate_synthetic_track(**arguments)


def test_midi_to_hz_reference_pitch():
    assert midi_to_hz(69) == pytest.approx(440.0)
    assert midi_to_hz(60) == pytest.approx(261.625565, rel=1e-6)

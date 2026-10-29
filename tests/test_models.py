"""Data-model invariants for buffers and stem sets."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_tone
from dcp.models import AudioBuffer, StemSet


def test_audio_buffer_properties():
    tone = make_tone(duration_s=1.0, amplitude=0.5)
    assert tone.channels == 2
    assert tone.frames == 22050
    assert tone.duration_s == pytest.approx(1.0)
    assert tone.peak == pytest.approx(0.5, abs=1e-4)
    assert tone.rms == pytest.approx(0.35355, abs=1e-4)
    assert tone.mono().shape == (22050,)


def test_audio_buffer_rejects_invalid_shapes_and_rate():
    with pytest.raises(ValueError, match="at least one channel"):
        AudioBuffer(np.zeros((0, 10)), 22050)
    with pytest.raises(ValueError, match="at least one frame"):
        AudioBuffer(np.zeros((2, 0)), 22050)
    with pytest.raises(ValueError, match="positive"):
        AudioBuffer(np.zeros((2, 10)), 0)
    with pytest.raises(ValueError, match="2-D"):
        AudioBuffer(np.zeros(10), 22050)
    with pytest.raises(TypeError, match="numpy.ndarray"):
        AudioBuffer([[0.0, 0.0]], 22050)


def test_stem_set_canonical_order_and_extras():
    audio = make_tone(duration_s=0.2)
    stems = StemSet(
        {
            "zither": audio,
            "vocals": audio,
            "drums": audio,
            "bass": audio,
            "other": audio,
        }
    )
    assert stems.names == ("drums", "bass", "other", "vocals", "zither")
    assert stems.sample_rate == audio.sample_rate
    assert stems.channels == audio.channels
    assert stems["vocals"] is audio
    assert "bass" in stems


def test_stem_set_mapping_is_read_only():
    audio = make_tone(duration_s=0.2)
    stems = StemSet({"vocals": audio})
    with pytest.raises(TypeError):
        stems.stems["bass"] = audio


def test_stem_set_requires_matching_formats():
    stereo = make_tone(duration_s=0.2, channels=2)
    mono = make_tone(duration_s=0.2, channels=1)
    with pytest.raises(ValueError, match="channels"):
        StemSet({"vocals": stereo, "drums": mono})

    low_rate = make_tone(duration_s=0.2, sample_rate=11025)
    with pytest.raises(ValueError, match="sample rate"):
        StemSet({"vocals": stereo, "drums": low_rate})


def test_stem_set_requires_valid_names_and_audio():
    audio = make_tone(duration_s=0.2)
    with pytest.raises(ValueError, match="non-empty"):
        StemSet({"": audio})
    with pytest.raises(TypeError, match="not an AudioBuffer"):
        StemSet({"vocals": np.zeros((2, 10))})
    with pytest.raises(ValueError, match="at least one stem"):
        StemSet({})

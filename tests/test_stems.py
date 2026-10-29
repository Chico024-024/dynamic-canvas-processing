"""The replaceable stem-separation contract and its explicit absence."""

from __future__ import annotations

import pytest

from conftest import make_tone
from dcp.stems import (
    StemSeparationUnavailable,
    StemSeparator,
    UnconfiguredSeparator,
)


def test_unconfigured_separator_is_unavailable_and_explains_alternatives():
    separator = UnconfiguredSeparator()
    assert separator.is_available() is False
    with pytest.raises(StemSeparationUnavailable) as excinfo:
        separator.separate(make_tone(duration_s=0.5))
    message = str(excinfo.value)
    assert "No stem-separation backend" in message
    assert "Provide stem WAV files" in message
    assert "licensed separator" in message


def test_custom_separator_satisfies_protocol():
    class DummySeparator:
        name = "dummy"

        def is_available(self) -> bool:
            return True

        def separate(self, audio, *, timeout_s=None):
            from dcp.models import StemSet

            return StemSet({"vocals": audio})

    separator = DummySeparator()
    assert isinstance(separator, StemSeparator)
    audio = make_tone(duration_s=0.5)
    stems = separator.separate(audio, timeout_s=10.0)
    assert stems["vocals"] is audio
    assert separator.is_available() is True

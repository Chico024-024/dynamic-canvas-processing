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


def test_demucs_adapter_satisfies_stem_separator_protocol():
    from dcp.stems import DemucsAdapter, StemSeparator

    adapter = DemucsAdapter()
    assert isinstance(adapter, StemSeparator)
    assert adapter.name == "demucs"


def test_demucs_adapter_raises_when_unavailable(monkeypatch):
    import dcp.stems.demucs_adapter as da
    from dcp.stems import DemucsAdapter, StemSeparationUnavailable

    monkeypatch.setattr(da, "demucs_available", lambda: False)
    adapter = DemucsAdapter(cache_dir=None)
    assert adapter.is_available() is False
    with pytest.raises(StemSeparationUnavailable) as exc:
        adapter.separate(make_tone(duration_s=0.5))
    assert "pip install demucs torch" in str(exc.value)


def test_demucs_adapter_caching_roundtrip(tmp_path):
    from dcp.models import StemSet
    from dcp.stems import DemucsAdapter

    audio = make_tone(duration_s=0.5, sample_rate=11025)
    stem_set = StemSet(
        {
            "drums": make_tone(frequency=100.0, duration_s=0.5, sample_rate=11025),
            "bass": make_tone(frequency=200.0, duration_s=0.5, sample_rate=11025),
            "other": make_tone(frequency=400.0, duration_s=0.5, sample_rate=11025),
            "vocals": make_tone(frequency=800.0, duration_s=0.5, sample_rate=11025),
        }
    )

    adapter = DemucsAdapter(cache_dir=tmp_path / "cache")
    assert adapter._load_from_cache(audio) is None

    adapter._save_to_cache(audio, stem_set)
    cached = adapter._load_from_cache(audio)
    assert set(cached.names) == {"drums", "bass", "other", "vocals"}
    assert cached.source_label == "demucs_cached"


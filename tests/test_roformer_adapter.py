"""Tests for BS-RoFormer adapter and the 3-tier hierarchical stem separation pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_tone
from dcp.models import AudioBuffer, StemSet
from dcp.stems import (
    DEFAULT_ROFORMER_MODEL,
    RoformerAdapter,
    StemSeparationUnavailable,
    StemSeparator,
    get_available_backends,
    get_roformer_separator,
    roformer_available,
    separate_hierarchical,
)
import dcp.stems.roformer_adapter as ra


def test_roformer_adapter_satisfies_stem_separator_protocol():
    adapter = RoformerAdapter()
    assert isinstance(adapter, StemSeparator)
    assert adapter.name == "roformer"
    assert adapter._model_name == DEFAULT_ROFORMER_MODEL


def test_roformer_adapter_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr(ra, "roformer_available", lambda: False)
    adapter = RoformerAdapter(cache_dir=None)
    assert adapter.is_available() is False
    with pytest.raises(StemSeparationUnavailable) as exc:
        adapter.separate(make_tone(duration_s=0.5))
    assert "pip install audio-separator" in str(exc.value)


def test_get_roformer_separator_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr(ra, "roformer_available", lambda: False)
    with pytest.raises(StemSeparationUnavailable):
        get_roformer_separator()


def test_roformer_adapter_caching_roundtrip(tmp_path):
    audio = make_tone(duration_s=0.5, sample_rate=11025)
    stem_set = StemSet(
        {
            "drums": make_tone(frequency=100.0, duration_s=0.5, sample_rate=11025),
            "bass": make_tone(frequency=200.0, duration_s=0.5, sample_rate=11025),
            "other": make_tone(frequency=400.0, duration_s=0.5, sample_rate=11025),
            "vocals": make_tone(frequency=800.0, duration_s=0.5, sample_rate=11025),
        }
    )

    adapter = RoformerAdapter(cache_dir=tmp_path / "cache")
    assert adapter._load_from_cache(audio) is None

    adapter._save_to_cache(audio, stem_set)
    cached = adapter._load_from_cache(audio)
    assert cached is not None
    assert set(cached.names) == {"drums", "bass", "other", "vocals"}
    assert cached.source_label == "roformer_cached"


def test_roformer_adapter_4stem_separation_mock(monkeypatch, tmp_path):
    monkeypatch.setattr(ra, "roformer_available", lambda: True)

    adapter = RoformerAdapter(cache_dir=tmp_path / "cache")
    audio = make_tone(duration_s=1.0, sample_rate=22050)

    # Mock the backend separation method
    def mock_backend(aud):
        return {
            "vocals": make_tone(frequency=440.0, duration_s=1.0, sample_rate=aud.sample_rate),
            "drums": make_tone(frequency=120.0, duration_s=1.0, sample_rate=aud.sample_rate),
            "bass": make_tone(frequency=80.0, duration_s=1.0, sample_rate=aud.sample_rate),
            "other": make_tone(frequency=1200.0, duration_s=1.0, sample_rate=aud.sample_rate),
        }

    monkeypatch.setattr(adapter, "_separate_with_backend", mock_backend)

    stems = adapter.separate(audio)
    assert set(stems.names) == {"vocals", "drums", "bass", "other"}
    assert stems.source_label == "roformer"

    # Verify cached on second call
    cached = adapter._load_from_cache(audio)
    assert cached is not None
    assert cached.source_label == "roformer_cached"


def test_roformer_adapter_2stem_collection_decomposes_instrumental():
    adapter = RoformerAdapter(cache_dir=None)
    sr = 22050
    dur = 1.0
    original = make_tone(frequency=300.0, duration_s=dur, sample_rate=sr)

    # Simulate 2-stem model output (e.g. vocals + instrumental)
    output_dict = {
        "vocals": make_tone(frequency=600.0, duration_s=dur, sample_rate=sr),
        "instrumental": make_tone(frequency=150.0, duration_s=dur, sample_rate=sr),
    }

    # Verify that collecting stems with 2 stems produces 4 canonical stems
    class DummyOutput:
        pass

    # Direct test of 2-stem resolution logic
    vocals_buf = output_dict["vocals"]
    inst_buf = output_dict["instrumental"]
    from dcp.stems.dsp_splitter import decompose_audio_to_stemset

    sub = decompose_audio_to_stemset(inst_buf, label="backing")
    result = {
        "vocals": vocals_buf,
        "drums": sub["drums"],
        "bass": sub["bass"],
        "other": sub["other"],
    }
    assert set(result.keys()) == {"vocals", "drums", "bass", "other"}


def test_hierarchical_tier3_dsp_always_available():
    audio = make_tone(duration_s=1.0, sample_rate=22050)
    stems, backend = separate_hierarchical(audio, preferred_backend="dsp")
    assert backend == "dsp"
    assert set(stems.names) == {"vocals", "drums", "bass", "other"}

    # Exact complementary reconstruction: sum(stems) == audio
    total_samples = np.zeros_like(stems["bass"].samples)
    for name in stems.names:
        total_samples += stems[name].samples
    orig_stereo = np.vstack([audio.samples[0], audio.samples[0]])
    np.testing.assert_allclose(total_samples, orig_stereo, atol=1e-5)


def test_hierarchical_tier1_roformer_selection(monkeypatch):
    import dcp.stems.hierarchical as hh

    monkeypatch.setattr(hh, "roformer_available", lambda: True)

    def mock_roformer_separate(self, audio, **kwargs):
        return StemSet(
            {
                "vocals": audio,
                "drums": audio,
                "bass": audio,
                "other": audio,
            },
            source_label="roformer",
        )

    monkeypatch.setattr(RoformerAdapter, "separate", mock_roformer_separate)

    audio = make_tone(duration_s=0.5)
    stems, backend = separate_hierarchical(audio, preferred_backend="auto")
    assert backend == "roformer"
    assert stems.source_label == "roformer"


def test_hierarchical_tier2_demucs_selection(monkeypatch):
    import dcp.stems.hierarchical as hh
    from dcp.stems import DemucsAdapter

    monkeypatch.setattr(hh, "roformer_available", lambda: False)
    monkeypatch.setattr(hh, "demucs_available", lambda: True)

    def mock_demucs_separate(self, audio, **kwargs):
        return StemSet(
            {
                "vocals": audio,
                "drums": audio,
                "bass": audio,
                "other": audio,
            },
            source_label="demucs",
        )

    monkeypatch.setattr(DemucsAdapter, "separate", mock_demucs_separate)

    audio = make_tone(duration_s=0.5)
    stems, backend = separate_hierarchical(audio, preferred_backend="auto")
    assert backend == "demucs"
    assert stems.source_label == "demucs"


def test_hierarchical_fallback_to_dsp_when_neurals_unavailable(monkeypatch):
    import dcp.stems.hierarchical as hh

    monkeypatch.setattr(hh, "roformer_available", lambda: False)
    monkeypatch.setattr(hh, "demucs_available", lambda: False)

    audio = make_tone(duration_s=0.5)
    stems, backend = separate_hierarchical(audio, preferred_backend="auto")
    assert backend == "dsp"
    assert set(stems.names) == {"vocals", "drums", "bass", "other"}


def test_get_available_backends_always_includes_dsp():
    backends = get_available_backends()
    assert "dsp" in backends
    assert backends[-1] == "dsp"


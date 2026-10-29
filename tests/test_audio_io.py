"""Audio loading, saving, resampling, and stem-directory tests."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from conftest import make_constant, make_silence, make_stems, make_tone
from dcp.audio_io import (
    convert_channels,
    load_audio,
    load_stem_set,
    match_format,
    resample_audio,
    save_audio,
    save_stem_set,
)
from dcp.models import AudioBuffer, AudioFormatError, StemSet


def test_round_trip_preserves_signal_and_format(tmp_path):
    original = make_tone(duration_s=1.5, frequency=330.0, amplitude=0.4)
    path = save_audio(tmp_path / "nested" / "tone.wav", original)

    assert path.exists()
    info = sf.info(str(path))
    assert info.subtype == "PCM_24"
    assert info.samplerate == original.sample_rate
    assert info.channels == 2

    loaded = load_audio(path)
    assert loaded.sample_rate == original.sample_rate
    assert loaded.channels == original.channels
    assert loaded.frames == original.frames
    assert np.allclose(loaded.samples, original.samples, atol=1e-6)


def test_load_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_audio(tmp_path / "absent.wav")


def test_load_corrupt_file_raises_audio_format_error(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"this is not a wav file")
    with pytest.raises(AudioFormatError, match="unsupported or corrupt"):
        load_audio(bad)


def test_load_directory_raises(tmp_path):
    with pytest.raises(AudioFormatError, match="not a file"):
        load_audio(tmp_path)


def test_load_rejects_more_than_two_channels(tmp_path):
    path = tmp_path / "surround.wav"
    sf.write(str(path), np.zeros((1000, 3)), 22050, subtype="PCM_24")
    with pytest.raises(AudioFormatError, match="mono or stereo"):
        load_audio(path)


def test_save_rejects_more_than_two_channels(tmp_path):
    buffer = AudioBuffer(np.zeros((3, 1000)), 22050)
    with pytest.raises(AudioFormatError, match="mono or stereo output"):
        save_audio(tmp_path / "surround.wav", buffer)


def test_save_refuses_non_finite_samples(tmp_path):
    samples = np.zeros((2, 100), dtype=np.float64)
    samples[0, 10] = np.nan
    buffer = AudioBuffer(samples, 22050)
    with pytest.raises(AudioFormatError, match="non-finite"):
        save_audio(tmp_path / "nan.wav", buffer)


def test_save_refuses_clipping_unless_allowed(tmp_path):
    loud = make_constant(1.2, duration_s=0.1)
    with pytest.raises(AudioFormatError, match="above 1.0"):
        save_audio(tmp_path / "loud.wav", loud)
    path = save_audio(tmp_path / "loud.wav", loud, allow_clipping=True)
    assert path.exists()


def test_resample_preserves_duration_and_channels():
    tone = make_tone(duration_s=1.0, sample_rate=22050)
    resampled = resample_audio(tone, 44100)
    assert resampled.sample_rate == 44100
    assert resampled.channels == 2
    assert abs(resampled.duration_s - tone.duration_s) < 1.0 / 44100
    assert resampled.peak <= 0.5 + 1e-3


def test_resample_same_rate_returns_same_buffer():
    tone = make_tone(duration_s=0.5)
    assert resample_audio(tone, 22050) is tone


def test_resample_rejects_invalid_rate():
    with pytest.raises(AudioFormatError, match="positive"):
        resample_audio(make_tone(duration_s=0.5), 0)


def test_mono_to_stereo_duplicates_channel():
    mono = make_tone(duration_s=0.5, channels=1)
    stereo = convert_channels(mono, 2)
    assert stereo.channels == 2
    assert np.array_equal(stereo.samples[0], stereo.samples[1])
    assert np.array_equal(stereo.samples[0], mono.samples[0])


def test_stereo_to_mono_averages_channels():
    tone = make_tone(duration_s=0.5, channels=2)
    mono = convert_channels(tone, 1)
    assert mono.channels == 1
    assert np.allclose(mono.samples[0], tone.samples.mean(axis=0))


def test_convert_channels_rejects_unsupported_layouts():
    three_channel = AudioBuffer(np.zeros((3, 100)), 22050)
    with pytest.raises(AudioFormatError, match="mono-to-stereo"):
        convert_channels(three_channel, 2)
    with pytest.raises(AudioFormatError, match="mono or stereo"):
        convert_channels(make_tone(duration_s=0.5), 3)


def test_match_format_adapts_and_reports_notes():
    reference = make_tone(duration_s=1.0, sample_rate=44100, channels=2)
    other = make_tone(duration_s=1.0, sample_rate=22050, channels=1)
    adapted, notes = match_format(other, reference)
    assert adapted.sample_rate == 44100
    assert adapted.channels == 2
    assert len(notes) == 2
    assert any("resampled" in note for note in notes)
    assert any("channel" in note for note in notes)


def test_load_audio_applies_targets(tmp_path):
    source = make_tone(duration_s=1.0, sample_rate=44100, channels=2)
    path = save_audio(tmp_path / "source.wav", source)
    loaded = load_audio(path, target_sample_rate=22050, target_channels=1)
    assert loaded.sample_rate == 22050
    assert loaded.channels == 1
    assert abs(loaded.duration_s - 1.0) < 0.01


def test_stem_set_round_trip(tmp_path):
    stems = make_stems(duration_s=1.0)
    written = save_stem_set(tmp_path / "stems", stems)
    assert set(written) == set(stems.names)

    loaded = load_stem_set(tmp_path / "stems")
    assert loaded.names == stems.names
    for name in stems.names:
        assert np.allclose(
            loaded[name].samples, stems[name].samples, atol=1e-6
        )


def test_load_stem_set_rejects_empty_directory(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(AudioFormatError, match="no WAV stems"):
        load_stem_set(empty)


def test_load_stem_set_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_stem_set(tmp_path / "absent")


def test_load_stem_set_rejects_mismatched_formats(tmp_path):
    save_audio(
        tmp_path / "one.wav", make_tone(duration_s=1.0, sample_rate=22050)
    )
    save_audio(
        tmp_path / "two.wav", make_tone(duration_s=1.0, sample_rate=44100)
    )
    with pytest.raises(AudioFormatError, match="invalid stem set"):
        load_stem_set(tmp_path)


def test_save_stem_set_rejects_unsafe_names(tmp_path):
    stems = StemSet({"bad/name": make_tone(duration_s=0.2)})
    with pytest.raises(AudioFormatError, match="safe filename"):
        save_stem_set(tmp_path, stems)


def test_stem_set_rejects_reserved_mix_name():
    with pytest.raises(ValueError, match="reserved"):
        StemSet({"__mix__": make_tone(duration_s=0.2)})


def test_silence_round_trip(tmp_path):
    silence = make_silence(duration_s=0.5)
    path = save_audio(tmp_path / "silence.wav", silence)
    loaded = load_audio(path)
    assert loaded.frames == silence.frames
    assert loaded.peak == 0.0

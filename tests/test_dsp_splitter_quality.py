"""Quality and mathematical verification tests for the multi-stage DSP stem separator."""

from __future__ import annotations

import numpy as np

from dcp.models import AudioBuffer
from dcp.stems.dsp_splitter import decompose_audio_to_stemset


def _synthesize_four_stem_mix(sr: int = 44100, duration_s: float = 4.0) -> tuple[AudioBuffer, dict[str, np.ndarray]]:
    n = int(round(duration_s * sr))
    t = np.arange(n, dtype=np.float64) / sr

    # 1. Sustained 65 Hz sub-bass (mono/center)
    bass_mono = 0.30 * np.sin(2.0 * np.pi * 65.0 * t)
    bass_true = np.vstack([bass_mono, bass_mono]).astype(np.float32)

    # 2. Sharp broadband percussive impulses every 0.5s
    drums_mono = np.zeros(n, dtype=np.float64)
    rng = np.random.default_rng(123)
    for beat_t in np.arange(0.25, duration_s - 0.1, 0.5):
        idx = int(round(beat_t * sr))
        burst_len = int(0.015 * sr)
        env = np.exp(-np.arange(burst_len) / (0.003 * sr))
        drums_mono[idx : idx + burst_len] += 0.45 * rng.standard_normal(burst_len) * env
    drums_true = np.vstack([drums_mono, drums_mono]).astype(np.float32)

    # 3. Center lead vocal proxy: vibrato-modulated harmonic stack (440 Hz + 880 Hz + 1320 Hz)
    vibrato = 5.5 * np.sin(2.0 * np.pi * 5.5 * t)
    phase_voc = 2.0 * np.pi * np.cumsum(440.0 + vibrato) / sr
    am_env = 0.5 + 0.5 * np.sin(2.0 * np.pi * 1.7 * t)
    voc_mono = 0.25 * am_env * (
        np.sin(phase_voc) + 0.6 * np.sin(2.0 * phase_voc) + 0.3 * np.sin(3.0 * phase_voc)
    )
    vocals_true = np.vstack([voc_mono, voc_mono]).astype(np.float32)

    # 4. Wide stereo repeating synth pad (decorrelated L/R @ 2200 Hz & 3300 Hz)
    pad_L = 0.20 * (np.sin(2.0 * np.pi * 2200.0 * t) + 0.5 * np.sin(2.0 * np.pi * 3300.0 * t))
    pad_R = 0.20 * (np.cos(2.0 * np.pi * 2200.0 * t) - 0.5 * np.cos(2.0 * np.pi * 3300.0 * t))
    other_true = np.vstack([pad_L, pad_R]).astype(np.float32)

    mix = (bass_true + drums_true + vocals_true + other_true).astype(np.float32)
    return AudioBuffer(samples=mix, sample_rate=sr), {
        "bass": bass_true,
        "drums": drums_true,
        "vocals": vocals_true,
        "other": other_true,
    }


def test_dsp_splitter_bit_exact_reconstruction_and_no_ghost_energy() -> None:
    mix_buf, _ = _synthesize_four_stem_mix()
    stemset = decompose_audio_to_stemset(mix_buf, label="test")

    reconstructed = (
        stemset.stems["bass"].samples
        + stemset.stems["drums"].samples
        + stemset.stems["vocals"].samples
        + stemset.stems["other"].samples
    )
    max_err = float(np.max(np.abs(reconstructed - mix_buf.samples)))
    assert max_err < 1e-5, f"Reconstruction error {max_err} exceeds float32 tolerance"

    # Verify Partition of Unity eliminates the old phase-inverted ghost-bleed bug:
    # sum of individual stem RMS powers should not wildly exceed total mix power
    mix_power = float(np.mean(mix_buf.samples ** 2))
    sum_stem_powers = sum(float(np.mean(s.samples ** 2)) for s in stemset.stems.values())
    assert sum_stem_powers <= 1.15 * mix_power, "Ghost anti-phase energy inflation detected!"


def test_dsp_splitter_routes_dominant_sources_to_correct_stems() -> None:
    mix_buf, ground_truth = _synthesize_four_stem_mix()
    stemset = decompose_audio_to_stemset(mix_buf, label="synth")

    # Correlation between each true component and its target separated stem
    for name in ("bass", "drums", "vocals", "other"):
        true_sig = ground_truth[name].ravel()
        pred_sig = stemset.stems[name].samples.ravel()
        corr = float(np.dot(true_sig, pred_sig) / (np.linalg.norm(true_sig) * np.linalg.norm(pred_sig) + 1e-12))
        assert corr > 0.55, f"Stem '{name}' correlation {corr:.3f} too low"


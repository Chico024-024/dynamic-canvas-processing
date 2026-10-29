# Testing

## How to run

```bash
# On the tested Windows environment
.venv/Scripts/python.exe -m pytest -q

# Any activated environment
python -m pytest -q
```

Optional focused runs:

```bash
python -m pytest tests/test_audio_io.py -q
python -m pytest tests/test_planner.py tests/test_render_plan.py -q
python -m pytest -q -k "advanced or fallback"
```

Verified result for the 0.1 milestone: **183 passed** in 20.77 seconds on
Windows 11 with Python 3.12.15 and the optional librosa 0.11.0 backend
installed. Without librosa, three backend/CLI tests are skipped and the
remaining suite still passes. Runtime varies with machine load.

The test suite does not require a GPU, network access, a source-separation
model, or copyrighted audio. librosa is optional; its tests skip cleanly when
the `analysis` extra is absent.

## Test design

- All audio fixtures are generated with NumPy or the deterministic
  `generate_synthetic_track` function.
- The synthetic tracks are 22.05 kHz in most tests to keep runtime low; the
  full demo runs at 44.1 kHz.
- Strategy tests pass explicit `TrackAnalysis` objects where a tempo decision
  must be deterministic. This avoids making plan assertions depend on the
  tempo estimator's behavior. Detected-beat tests supply explicit beat times,
  so boundary assertions are exact rather than dependent on librosa output.
- Tempo analysis itself is tested separately on a 16-second 120 BPM synthetic
  track, where the estimate is known to be reliable.
- The optional librosa backend is tested on the same synthetic track, with
  tempo tolerance and structural checks on beat ordering, duration, confidence,
  and JSON output. It does not claim real-music beat accuracy.
- Tests use temporary directories and never write into the repository except
  the demo, which writes under the ignored `output/` directory.
- Assertions check audible/structural behavior (duration, boundaries, exact
  stem equality, spectral attenuation) as well as status and error messages.

## Coverage by area

| File | What it verifies |
| --- | --- |
| `test_audio_io.py` | 24-bit WAV round trip, missing/corrupt files, non-finite and clipping refusal, resampling, mono/stereo conversion, format adaptation, stem directory import/export, unsafe names, channel limits. |
| `test_synthetic.py` | Seed determinism, exact mix = sum of stems, peak and format, mono generation, vocal lead-in/out, invalid parameters. |
| `test_analysis.py` | Peak/RMS accuracy, silence ratio, short-track behavior, 120 BPM tempo estimate, non-finite rejection, JSON serialization, confidence-gated tempo usability. |
| `test_analysis_backend.py` | Basic backend metadata, explicit backend selection, unknown backend rejection, short-track fallback, optional librosa beat detection, confidence threshold, beat ordering, and `auto` selection. |
| `test_curves.py` | dB conversion, floor-to-zero behavior, equal-power identity, interpolation, held endpoints, curve differences, invalid inputs. |
| `test_effects.py` | High-frequency attenuation, low-frequency preservation, input immutability, effect start boundaries, Nyquist and cutoff validation, out-of-window copy behavior. |
| `test_crossfade.py` | Output length, start/end boundaries, equal-power behavior, clipping prevention and diagnostics mode, resampling/channel adaptation, invalid overlaps. |
| `test_render_plan.py` | Independent stem gains, stem coverage errors, unknown stems, full-mix behavior with supplied stems, output duration, clipping prevention, late transitions, format adaptation. |
| `test_planner.py` | All six Core strategies, registry behavior, stem windows, detected-beat and whole-bar quantization, tempo fallback, atmospheric sweeps, auto selection, duration clamping, JSON export. |
| `test_optimize.py` | Objective weighting, normalized loss bounds, inert musical placeholder, optimized-plan validity and metadata, no-stem fallback. |
| `test_plan_validation.py` | Structural invariants: gain ranges, point ordering, duplicate sources/stems, curves, effects, Nyquist, windows. |
| `test_validation.py` | Finite/NaN detection, clipping, duration/SR/channel checks, interior silence, edge silence, discontinuity, metrics, comparison disclaimer. |
| `test_advanced.py` | Core operation when Advanced is absent, broken plugins, discovery failure, valid plugin registration, Core-name protection, planner construction. |
| `test_models.py` | AudioBuffer properties and invalid shapes, StemSet ordering, read-only mapping, format/name validation. |
| `test_stems.py` | Unconfigured separator error, custom separator satisfying the protocol. |
| `test_cli.py` | Every CLI command, analyzer selection, JSON outputs, plan/report writing, exit codes 0/1/2, help, doctor, `python -m dcp`, and the complete demo. |

## What the assertions can and cannot prove

Verified by tests:

- renders are finite, correctly sized, correctly sampled, and below full scale;
- no unexpected interior silence or extreme adjacent-sample jumps appear in the
  demo material;
- stems reconstruct their synthetic mix exactly;
- plans are structurally consistent and respect gain/effect limits;
- fallback strategies behave as documented;
- Core keeps working with Advanced absent or broken.

Not verified by tests:

- perceptual quality, musicality, or listener preference;
- behavior on arbitrary commercial recordings;
- separation quality or artifacts;
- beat-tracking accuracy or absolute beat alignment on real music;
- row-level equality of librosa output across versions or platforms;
- real-time performance, latency, or gap-free playback;
- licensing compliance of third-party models or binaries.

Objective checks are necessary but not sufficient. The roadmap starts real-music
listening evaluation before expanding claims.

## Determinism and environment sensitivity

Synthetic generation is seeded. DSP uses NumPy/SciPy operations that are
deterministic for the same input and versions, but floating-point results can
differ slightly across platforms or library builds. Tests use tolerances for
audio comparisons where resampling or filtering is involved, and exact equality
only for direct array operations and the synthetic stem sum.

Tempo tests target 120 BPM because the estimator is known to be reliable there.
It is explicitly documented as approximate and octave-ambiguous elsewhere; no
test pretends otherwise.

Detected-beat plan tests use explicit beat times, not librosa inference, so plan
boundaries are deterministic. The librosa backend test asserts tempo tolerance,
beat ordering, and confidence gating rather than exact beat timestamps; beat
tracking can change with library versions or native builds.

## Adding tests

- Put reusable fixtures in `tests/conftest.py`.
- Prefer synthetic signals with known expected properties.
- For plans, assert behavior (gains, times, coverage, warnings), not just that a
  field exists.
- Keep new tests independent of generated files in `output/`.
- If adding a real-music fixture, confirm rights and keep it out of source
  control unless the license explicitly allows redistribution.

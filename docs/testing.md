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

Verified result: **265 passed** in 36.94 seconds on Windows 11 with Python
3.12.15, librosa 0.11.0, and PyTorch 2.15.0 (CPU) installed. torchaudio and
Demucs are not installed, so their optional runtime paths are not exercised.
Coverage includes all core features, harmonic key detection, Camelot
compatibility, downbeat phase alignment, content-hash caching, the
model-download consent guard, the Demucs adapter contract, and the interactive
HTML visualizer. Runtime varies with machine load.

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
- Labeling tests inject scripted `ask`/`echo` functions instead of using real
  input, so sessions are deterministic and non-interactive. They render 2-second
  transitions at 22.05 kHz to keep runtime low.
- Assertions check audible/structural behavior (duration, boundaries, exact
  stem equality, spectral attenuation) as well as status and error messages.

## Coverage by area

| File | What it verifies |
| --- | --- |
| `test_audio_io.py` | 24-bit WAV round trip, missing/corrupt files, non-finite and clipping refusal, resampling, mono/stereo conversion, format adaptation, stem directory import/export, unsafe names, channel limits. |
| `test_synthetic.py` | Seed determinism, exact mix = sum of stems, peak and format, mono generation, vocal lead-in/out, invalid parameters. |
| `test_analysis.py` | Peak/RMS accuracy, silence ratio, short-track behavior, 120 BPM tempo estimate, non-finite rejection, JSON serialization, confidence-gated tempo usability. |
| `test_analysis_backend.py` | Basic backend metadata, explicit backend selection, unknown backend rejection, short-track fallback, optional librosa beat detection, confidence threshold, beat ordering, and `auto` selection. |
| `test_key_detection.py` | Camelot distance and compatibility, key detection fallback on short/silent audio and a synthetic tone, downbeat phase estimation, and downbeat-aligned rhythmic planning. |
| `test_curves.py` | dB conversion, floor-to-zero behavior, equal-power identity, interpolation, held endpoints, curve differences, invalid inputs. |
| `test_effects.py` | High-frequency attenuation, low-frequency preservation, input immutability, effect start boundaries, Nyquist and cutoff validation, out-of-window copy behavior. |
| `test_crossfade.py` | Output length, start/end boundaries, equal-power behavior, clipping prevention and diagnostics mode, resampling/channel adaptation, invalid overlaps. |
| `test_render_plan.py` | Independent stem gains, stem coverage errors, unknown stems, full-mix behavior with supplied stems, output duration, clipping prevention, late transitions, format adaptation. |
| `test_cache.py` | Content/config cache keying, full-mix and stem round trips, miss/mismatch behavior, delete/clear/list/stats, and the cached-transition helper. |
| `test_planner.py` | All six Core strategies, registry behavior, stem windows, detected-beat and whole-bar quantization, tempo fallback, atmospheric sweeps, auto selection, duration clamping, JSON export. |
| `test_alignment.py` | Phase-lock timing and failure telemetry, source/output timeline mapping, and sample-accurate rendered downbeat coincidence. |
| `test_optimize.py` | Objective weighting, normalized loss bounds, inert musical placeholder, optimized-plan validity and metadata, no-stem fallback, candidate generation, the default scorer matching the objective, custom scorer selection, and rejection of unnamed or non-finite scorers. |
| `test_policy_network.py` | Feature extraction and tensor conversion, network forward/parameter shapes, and `NeuralPlanStrategy` plan/fallback behavior using untrained random weights. |
| `test_labeling.py` | Candidate rendering, pairwise JSONL records, A/B/tie/skip mapping, re-prompting, quit and EOF handling, seed-reproducible pair order, stem requirement, invalid pair limits, and absence of absolute paths in logs. |
| `test_plan_validation.py` | Structural invariants: gain ranges, point ordering, duplicate sources/stems, curves, effects, Nyquist, windows. |
| `test_validation.py` | Finite/NaN detection, clipping, duration/SR/channel checks, interior silence, edge silence, discontinuity, metrics, comparison disclaimer. |
| `test_advanced.py` | Core operation when Advanced is absent, broken plugins, discovery failure, valid plugin registration, Core-name protection, planner construction. |
| `test_models.py` | AudioBuffer properties and invalid shapes, StemSet ordering, read-only mapping, format/name validation. |
| `test_stems.py` | Unconfigured separator error, custom separator satisfying the protocol, and the Demucs adapter's protocol, unavailable error, and cache round trip. |
| `test_model_registry.py` | Download consent guard, explicit flag/environment opt-in, cached-model return, corrupted-cache eviction, hash verification, and cache listing/clearing. |
| `test_visualizer.py` | HTML report contents and file writing, CLI visualization on directories/plans, and the missing-Demucs error path. |
| `test_cli.py` | Every CLI command including `label`, analyzer selection, JSON outputs, plan/report writing, exit codes 0/1/2, help, doctor, `python -m dcp`, and the complete demo. |

## What the assertions can and cannot prove

Verified by tests:

- renders are finite, correctly sized, correctly sampled, and below full scale;
- no unexpected interior silence or extreme adjacent-sample jumps appear in the
  demo material;
- stems reconstruct their synthetic mix exactly;
- plans are structurally consistent and respect gain/effect limits;
- fallback strategies behave as documented;
- preference logs contain both candidates' features, a consistent choice mapping,
  and no absolute paths;
- Core keeps working with Advanced absent or broken.

Not verified by tests:

- perceptual quality, musicality, or listener preference;
- whether collected preferences are consistent, sufficient, or predictive;
- any trained ranking model, since none exists;
- accuracy or musical value of the untrained neural policy; tests verify
  architecture, shapes, and fallback behavior only;
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

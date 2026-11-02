# DCP - Dynamic Canvas Processing

DCP is an offline music-transition engine. Its long-term goal is to make the
boundary between two songs sound composed rather than merely crossfaded: vocals
finish naturally, instrumental layers overlap deliberately, low end does not
pile up, and timing decisions are explained and testable.

This repository is the **DCP Core 0.1 prototype**: a runnable, CPU-only,
explainable foundation. It is not a streaming service, not a real-time engine,
and it does not bundle a source-separation model.

**Status:** All 265 automated tests passing. The full synthetic demo renders and validates both a conventional crossfade and a stem-level DCP transition, and generates an interactive HTML preview. Core includes lightweight harmonic key detection, downbeat snapping, a Demucs stem separation adapter, and a visual previewer, all while remaining independent of proprietary Advanced features.

**Two-Tier Architecture & Licensing:**
- **DCP Core (Open Source Basic Engine):** Permissively licensed public foundation. Offline deterministic rendering, 5 rule-based strategies + RATO candidate optimization, chroma-based key detection, downbeat snap, pluggable contracts for librosa beat tracking and Demucs source separation, CLI (`dcp`), and visual HTML previewer.
- **DCP Advanced SDK (Royalty & Licensing Kit):** Commercial SDK designed to be embedded in streaming services (Spotify, Apple Music, Tidal, etc.) and player apps on a royalty/licensing basis. Adds real-time low-latency playback precomputing, deep neural phrase segmentation, phase-aligned time-stretching, neural harmonic modulation, and a learned quality ranking model.

See [`LICENSE.md`](LICENSE.md) and [`docs/licensing.md`](docs/licensing.md).

## What is implemented

All items below are implemented and covered by automated tests:

- Audio loading and saving, with mono/stereo and sample-rate conversion.
- Deterministic synthetic songs with four stems and a mix that is exactly the
  sum of those stems.
- Basic analysis: duration, sample rate, channels, peak, RMS, near-silence
  ratio, onset count, and a confidence-gated tempo estimate.
- **Harmonic Key Detection & Camelot Matching:** Lightweight CPU-only chroma feature
  extraction and Krumhansl-Schmuckler key correlation. Computes Camelot wheel codes
  (e.g. 8A, 8B) and harmonic compatibility distance/scores between tracks.
- **Downbeat / Measure Grid Snap:** Energy-based downbeat estimation in 4/4 meter.
  The `rhythmic` strategy snaps transition start points to bar boundaries
  (downbeats) when detected.
- Optional librosa analysis backend for tempo and beat positions, with a
  conservative cross-check against the basic estimator.
- A conventional equal-power crossfade baseline.
- User-supplied stem import from WAV directories.
- **Demucs Stem Separation Adapter:** Optional `StemSeparator` wrapping Meta's
  Demucs model with content-hash caching under `.dcp_cache/stems/`. Core works
  fully without it; users can separate via `dcp separate` or `render --separate`.
- A transition-plan model with per-source, per-stem gain automation, smooth
  curves, optional low-pass sweep effects, reasoning, warnings, and JSON export.
- Six strategies: five rule-based (`crossfade`, `stem_level`, `rhythmic`,
  `atmospheric`, `conservative`) plus the objective-based `optimized`
  candidate search (RATO), which scores and selects among timing plans.
- A pluggable `QualityScorer` contract for candidate ranking. Core's default
  `WeightedLossScorer` is the hand-tuned weighted loss sum.
- A local pairwise-labeling workflow (`dcp label`) that renders candidate pairs
  to WAV, asks which one is better, and appends measured features plus the
  choice to a JSONL file.
- **Visual Transition Previewer (`dcp visualize`):** Generates a standalone,
  interactive HTML dashboard with an A/B audio player (baseline crossfade vs
  DCP transition), stem automation gain curves canvas, key match badges, and
  engine diagnostic metrics.
- Offline rendering of plans to WAV, with automatic clipping prevention.
- Objective validation: finite samples, duration, sample rate, channels, peak,
  clipping, interior silence, and a discontinuity heuristic.
- A command-line interface and a reproducible end-to-end demo.
- A stable, optional DCP Advanced plugin boundary with graceful failure when no
  plugin is installed.

## What Core does not do (Advanced SDK Scope)

These deliberate boundaries keep Core lightweight and define the Advanced SDK tier:

- **No real-time playback engine.** Core renders offline WAV files. Real-time
  playback buffer precomputing belongs to the Advanced SDK.
- **No deep neural phrase segmentation.** Core uses energy-based downbeats and
  fractional windows; neural chorus/verse boundary tracking is an Advanced SDK feature.
- **No time-stretching or tempo warping.** Songs are never warped in Core; only
  1:1 or 2:1 ratios are aligned rhythmically. High-fidelity phase vocoding belongs
  to the Advanced SDK.
- **No learned ranking model.** Core provides `dcp label` to collect preference
  data, but training and deploying pairwise neural ranking models is an Advanced capability.
- **No cloud upload or forced network calls.** Core operates 100% locally on CPU.

## Requirements

- Python 3.11 or newer.
- NumPy, SciPy, and SoundFile (installed automatically).
- pytest for the development/test workflow.
- Optional: librosa 0.10-0.11 for beat tracking and a second tempo estimator,
  installed through the `analysis` extra. It pulls in numba, llvmlite, and
  scikit-learn, so it is deliberately not a required dependency.
- No GPU, model download, or network access is required. librosa computes
  locally and does not upload audio.

The verified environment for this milestone was Windows 11 with Python 3.12.15,
NumPy 2.5.4, SciPy 1.18.1, SoundFile 0.14.0 (libsndfile 1.2.2), pytest
9.2.0, and optionally librosa 0.11.0. The code has no intentional Windows-only
dependency, but macOS and Linux have not been exercised in this milestone.

## Install

### Windows, using uv (the tested path)

From the project directory:

```bash
uv venv .venv --python 3.12
# Note: Use --link-mode copy if your project is inside OneDrive or a cloud-synced folder
uv pip install --link-mode copy --python .venv/Scripts/python.exe -e ".[dev]"

# Optional beat tracking and secondary tempo estimate
uv pip install --link-mode copy --python .venv/Scripts/python.exe -e ".[dev,analysis]"
```

### Standard virtual environment

The equivalent with pip is:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev]"

# Optional
python -m pip install -e ".[dev,analysis]"
```

After installation, `dcp` is available inside the environment. Running
`python -m dcp` works as well.

## Quick start

```bash
# Show the environment, available strategies, and plugin status
.venv/Scripts/dcp.exe doctor

# Generate two songs, stems, a crossfade baseline, and a DCP stem transition
.venv/Scripts/dcp.exe demo --out output/demo

# Run the automated test suite
.venv/Scripts/python.exe -m pytest -q
```

The demo writes:

```text
output/demo/
  song_a.wav
  song_b.wav
  stems/a/{drums,bass,other,vocals}.wav
  stems/b/{drums,bass,other,vocals}.wav
  baseline_crossfade.wav
  dcp_stem_transition.wav
  plan.json
  report.json
```

On the verified run (`--duration 16 --transition 8 --sample-rate 44100
--seed 7`), Song A analyzes as 119.99 BPM (confidence 0.69) and Song B as
124.03 BPM (confidence 0.71). Both 24.000-second outputs pass validation with
no clipping and no interior silence. The demo completed in under seven seconds
in the verified environment. The DCP output uses eight independent stem
automations. The baseline and DCP output are not claimed to be perceptually
equivalent or superior; no listening test has been conducted.

## Command-line interface

All processing is local. Commands return exit code `0` on success or a passing
validation, `1` on a failed validation or unexpected internal error, and `2` on
an expected user error such as a missing file or invalid parameter.

| Command | Exact syntax | Purpose |
| --- | --- | --- |
| `doctor` | `dcp doctor` | Show Python, dependency, strategy, and Advanced-plugin status. |
| `analyze` | `dcp analyze PATH [--json] [--analyzer auto\|basic\|librosa]` | Measure duration, levels, silence, onsets, tempo, and (with librosa) beat positions. |
| `synthesize` | `dcp synthesize [--out DIR] [--duration-a S] [--duration-b S] [--bpm-a BPM] [--bpm-b BPM] [--sample-rate HZ] [--seed N]` | Generate two synthetic songs and their stems. |
| `crossfade` | `dcp crossfade A B OUT [--overlap S] [--no-limit]` | Render the conventional equal-power baseline and validate it. |
| `render` | `dcp render A B OUT [--stems-a DIR] [--stems-b DIR] [--strategy NAME] [--transition S] [--analyzer auto\|basic\|librosa] [--no-limit] [--plan-json PATH] [--report-json PATH]` | Plan, render, and validate a DCP transition. |
| `label` | `dcp label A B [--stems-a DIR] [--stems-b DIR] [--transition S] [--analyzer auto\|basic\|librosa] [--out DIR] [--log PATH] [--max-pairs N] [--seed N] [--note TEXT] [--no-limit]` | Render optimizer candidates locally and record human A/B/tie/skip preferences as JSONL. Requires stems for at least one song. |
| `validate` | `dcp validate PATH [--expected-duration S] [--max-silence S] [--json]` | Run structural and numeric checks on an audio file. |
| `compare` | `dcp compare BASELINE CANDIDATE [--json]` | Compare objective metrics of two renders. |
| `demo` | `dcp demo [--out DIR] [--duration S] [--transition S] [--sample-rate HZ] [--seed N]` | Run the complete synthetic workflow. |

Global options are `--version` and `--debug` (show tracebacks for unexpected
errors). `--no-limit` disables the automatic master gain that prevents clipping;
the resulting file may clip and fail validation.

`--analyzer auto` is the CLI default: it uses librosa when installed and basic
analysis otherwise. If an installed librosa backend fails, `auto` records the
failure and keeps going with basic analysis. `basic` is always available.
Requesting `librosa` explicitly raises a clear error if the optional extra is
missing or the backend fails.

Example with user-provided stems:

```bash
dcp render song_a.wav song_b.wav dcp.wav \
  --stems-a stems/a --stems-b stems/b \
  --strategy stem_level --transition 8 \
  --plan-json plan.json --report-json report.json
```

If either stem directory is omitted, that song is treated as a single full mix.
The strategy then may use stem automation only for the supplied side, and the
plan records warnings rather than pretending missing stems exist.

## Human preference labeling

`dcp label` collects the human signal that no amount of feature engineering can
supply on its own:

```bash
dcp label song_a.wav song_b.wav \
  --stems-a stems/a --stems-b stems/b \
  --transition 8 --max-pairs 6 \
  --out output/labeling --note "first session"
```

What happens:

1. The optimizer's candidate timing plans are generated and rendered once each
   to `output/labeling/<session>/NN_<candidate>.wav`.
2. DCP prints two candidate paths and asks which sounds better. It does not
   play audio; open the files in any local player.
3. Valid answers are `a`, `b`, `tie`, `skip`, and `quit`. Pair selection and
   left/right order are randomized from `--seed` and recorded.
4. Each answer is appended immediately to `output/labeling/labels.jsonl`. A
   `session.json` records the seed, candidate list, transition settings, and
   input hashes.

Each JSONL record contains the measured features of both candidates, the
weighted objective as a baseline, the choice, and SHA-256 hashes of the input
audio and stems. It contains **no absolute paths**; audio stays local and no
network request is made.

This is the data-collection half of a human-feedback loop, not a learning
system. It does not train a model, and the optimizer does not consume the log.
A learned scorer is future work; until then Core keeps the deterministic
`WeightedLossScorer`.

Programmatic entry points are `generate_candidate_plans`,
`evaluate_candidate`, `WeightedLossScorer`, and the `QualityScorer` protocol in
`dcp.transitions`, plus `run_labeling_session` in `dcp.labeling`.

## Programmatic use

```python
from dcp import generate_synthetic_track, save_audio
from dcp.analysis import analyze_track_with_backend
from dcp.rendering import render_plan, rendered_duration_s
from dcp.transitions import plan_transition
from dcp.validation import validate_audio

song_a = generate_synthetic_track(bpm=120, duration_s=16, seed=7)
song_b = generate_synthetic_track(
    bpm=124, duration_s=16, seed=8, root_midi=60, mode="major"
)

# "auto" uses librosa when installed and basic analysis otherwise.
analysis_a = analyze_track_with_backend(song_a.mix)
analysis_b = analyze_track_with_backend(song_b.mix)

plan = plan_transition(
    song_a.mix,
    song_b.mix,
    stems_a=song_a.stems,
    stems_b=song_b.stems,
    strategy="stem_level",
    duration_s=8,
    analysis_a=analysis_a,
    analysis_b=analysis_b,
)
result = render_plan(
    song_a.mix, song_b.mix, plan, stems_a=song_a.stems, stems_b=song_b.stems
)
save_audio("dcp.wav", result.audio)
report = validate_audio(
    result.audio,
    expected_duration_s=rendered_duration_s(song_a.mix, song_b.mix, plan),
    expected_sample_rate=song_a.mix.sample_rate,
    expected_channels=song_a.mix.channels,
)
print(report.format_text())
```

`render_plan` performs validation of the plan and its stem coverage. It refuses
to render a plan that references stems not supplied, or that fails to cover a
supplied stem.

## Transition strategies

| Strategy | Actual behavior |
| --- | --- |
| `crossfade` | Full-mix fade, linear in decibels, over the requested duration. Stems are ignored. The separate `render_crossfade` API provides the equal-power baseline. |
| `stem_level` | Independent decibel-smooth fades for every supplied stem, with staggered timing: vocals from the two songs do not overlap, bass is reduced early, drums overlap, and Song B's `other` layer enters early. Falls back to `conservative` if no stems exist. |
| `rhythmic` | Only used when both tempo estimates pass the confidence threshold and their ratio is 1:1 or 2:1 within tolerance. With detected beats available, the transition starts on a detected Song A beat near the requested boundary and fade boundaries follow actual beat times; otherwise boundaries fall back to whole-beat fractions of Song A's tempo grid. Assumes 4/4, does not detect downbeats, does not phase-align Song B's internal beats, and does not stretch audio. Falls back to `conservative` if the estimates are unreliable. |
| `atmospheric` | Requires stems. Uses a longer overlap (1.4x requested, at least 10 seconds, clamped to the shorter track), keeps vocals apart, and applies two cascaded one-pole low-pass sweeps to the `other` stems. Falls back to `conservative` without stems. |
| `conservative` | Predictable full-mix smooth fade with no tempo, key, stem, or phrase assumptions. |
| `optimized` | Generates several candidate stem-timing plans, renders and scores each against a normalized objective (`loudness`, `continuity`, `vocal`, `rhythm`, `distortion`, `complexity`; `musical` is a placeholder), and returns the lowest-scoring plan. Requires stems for at least one song, otherwise falls back to `conservative`. |
| `auto` | Chooses `rhythmic` if tempo is reliable, otherwise `stem_level` if any stems are supplied, otherwise `conservative`. |

Every strategy returns a `TransitionPlan` with reasoning and warnings. A plan is
validated before rendering. Strategy names from installed Advanced plugins
cannot replace Core names.

## Audio and rendering policy

- Internal processing is float64, de-interleaved as `(channels, frames)`.
- File input and output are mono or stereo. Other channel counts are rejected.
- Song B is adapted to Song A's sample rate and channel layout using polyphase
  resampling and mono/stereo conversion; the plan/report records this.
- WAV output defaults to 24-bit PCM. Original sample rate and channel count are
  preserved unless adaptation is required.
- Output duration for a plan is
  `max(duration(A), plan.start_time_s + duration(B))`.
- If the rendered peak exceeds 0.999, a master gain is applied and noted, unless
  clipping prevention is disabled.
- Automatic gain decisions are deterministic.

## Validation

`validate_audio` checks:

- non-empty audio and finite samples;
- duration against an expected value (tolerance `max(0.05 s, 2 samples)`);
- expected sample rate and channel count;
- peak level at or below full scale;
- clipping as the share of samples at or above -0.0009 dBFS;
- longest interior silence (default limit 3.0 s, threshold -70 dBFS);
- an adjacent-sample step heuristic (default limit 1.5).

These are **objective** checks. They can reveal broken renders, truncation,
clipping, and obvious discontinuities. They cannot prove that a transition is
musically pleasing. Listening tests and perceptual evaluation remain future
work.

## Tests

```bash
.venv/Scripts/python.exe -m pytest -q
```

Verified result: **265 passed** in 36.94 seconds with the optional librosa
0.11.0 and PyTorch 2.15.0 (CPU) backends installed (runtime varies with machine
load). Optional backends are not required for Core rendering. The suite covers
audio I/O,
synthetic determinism and exact stem reconstruction, analysis, gain curves,
filter effects, crossfade behavior, plan validation, stem coverage, every Core
strategy, detected-beat planning, the objective-based optimizer, the pluggable
scorer contract, local preference labeling and its JSONL schema, fallback
behavior, Advanced-plugin absence/failure, validation metrics, CLI exit codes,
and the end-to-end demo.

## Project layout

```text
src/dcp/
  audio_io.py       audio loading, saving, resampling, stem directories
  models.py         AudioBuffer, StemSet, error types
  config.py         frozen analysis/transition/render configuration
  synthetic.py      deterministic demo music and stems
  analysis/         basic measurements, optional librosa beat tracking
  stems/            replaceable stem-separation interface
  transitions/      plan model, curves, strategies, quality scoring, optimizer
  rendering/        crossfade baseline, plan renderer, filters
  validation/       checks, metrics, objective comparison
  advanced.py       optional Advanced entry-point discovery
  labeling.py       local pairwise preference sessions and JSONL logging
  demo.py           reproducible end-to-end demo
  cli.py            command-line interface
tests/              265 automated tests
examples/demo.py    demo wrapper
docs/               architecture, feature matrix, roadmap, licensing
```

## DCP Core and DCP Advanced

**DCP Core** is the public foundation in this repository. It is usable on its
own and has no dependency on proprietary code.

**DCP Advanced** is a planned proprietary edition. It is not implemented here.
Core exposes a local, failure-tolerant entry-point boundary
(`dcp.advanced.strategies`); any separately distributed package may register
additional strategies that satisfy the same `TransitionStrategy` contract.
Core remains fully operational if the package is missing, broken, or rejects a
plugin. No licensing check, DRM, payment integration, or artificial limitation
is present in this prototype.

The intended commercial model and metering concepts are documented, but no
metering, billing, or network code exists. See
[`docs/licensing.md`](docs/licensing.md).

## Privacy and security

- Audio stays local. DCP Core makes no network requests and uploads nothing.
- Preference logs produced by `dcp label` stay in the chosen local directory and
  contain no absolute filesystem paths.
- No API keys or credentials are included.
- The Advanced boundary treats plugin failures as local errors and never
  disables Core rendering.
- Future separators or hosted services must require explicit user authorization
  before any audio leaves the machine.

## Documentation

- [Architecture](docs/architecture.md)
- [Feature matrix: Core versus Advanced](docs/feature_matrix.md)
- [Roadmap and next priorities](docs/roadmap.md)
- [Licensing and commercial considerations](docs/licensing.md)
- [Testing](docs/testing.md)

## Known limitations

Tempo estimation is approximate and can confuse half-time and double-time
(especially with sparse or synthetic patterns). The basic analyzer has no beat
tracker; when the optional librosa backend is installed, detected beat phase is
also approximate, downbeats are not identified, and the cross-checked confidence
can reject contradictory estimates. Rhythmic planning assumes 4/4, uses Song A's
beat grid only, does not phase-align Song B, and does not stretch either song.
Stem-level planning relies entirely on the quality of supplied stems; separation
artifacts are the user's responsibility. No perceptual quality evaluation has
been done, and no preference-trained model exists. `dcp label` records pairwise
human judgments but does not use them to change any score. The renderer is
offline and not suitable for live playback without a precompute and buffering
design.

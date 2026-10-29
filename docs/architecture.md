# DCP Architecture

This document describes the DCP Core 0.1 implementation. It separates what is
built and tested from the extension points intended for a future proprietary
DCP Advanced edition.

## Scope and principles

DCP is currently an **offline renderer**. It takes two complete songs, optional
stem sets, and a transition request; it produces a plan, a rendered WAV, and an
objective validation report. The design favors:

- deterministic DSP rather than unpredictable model behavior;
- explainable plans with reasoning and warnings;
- strict numerical validation at boundaries;
- local-only processing;
- a small, replaceable extension boundary rather than premature services;
- honest fallbacks when analysis or stems are missing.

No source-separation model, network service, GPU, or real-time audio path is
part of this milestone.

## Component map

```text
                 +----------------------+
                 |  CLI / demo / Python |
                 +----------+-----------+
                            |
              +-------------v-------------+
              | analysis  -> tracks        |
              | transitions -> plans      |
              +-------------+-------------+
                            |
        +-------------------+--------------------+
        |                                        |
+-------v--------+                     +---------v---------+
| rendering      |                     | validation        |
| crossfade      |                     | checks + metrics  |
| plan renderer  |                     | comparison        |
| DSP effects    |                     +-------------------+
+-------+--------+
        |
+-------v--------+
| audio_io       |
| soundfile      |
+----------------+
```

Supporting modules hold data models, configuration, synthetic material, optional
stem-separation contracts, and the Advanced plugin boundary.

## Data representation

`AudioBuffer` stores de-interleaved float64 samples with shape
`(channels, frames)` and an integer sample rate. It validates the array shape
and positive rate, but allows non-finite values so that validation can report
them; file writing refuses non-finite data.

`StemSet` is a frozen mapping of stem name to `AudioBuffer`. All stems must share
a sample rate and channel count. The name `__mix__` is reserved for the full mix
inside plans. Canonical names are `drums`, `bass`, `other`, and `vocals`; extra
user names are permitted and sorted after the canonical names.

File input and output are limited to mono and stereo. `soundfile` handles
decoding and encoding; libsndfile determines the supported input formats. WAV
output defaults to 24-bit PCM. Polyphase resampling and explicit mono/stereo
conversion adapt Song B or supplied stems to Song A's format, and the adaptation
is recorded in render notes and reports.

## End-to-end data flow

1. **Generate or load** two songs and optional stem directories.
2. **Analyze** each song (duration, levels, silence, onsets, tempo estimate, and
   optionally detected beat positions through librosa).
3. **Plan** a transition with a selected strategy. The planner supplies a
   `TransitionContext` containing songs, stems, analyses, the requested
   duration, and the configuration.
4. **Validate** the resulting `TransitionPlan` structurally and numerically.
5. **Render** the full mixes or supplied stems with per-stem gain automation
   and optional effects.
6. **Finalize** the output, checking for non-finite values and applying a master
   gain if the peak exceeds the ceiling.
7. **Validate** the rendered audio and write a plan/report JSON where requested.

The crossfade baseline follows a shorter path: it renders the two full mixes
with equal-power fades and validates its own output.

## Analysis

`analysis/basic.py` provides objective measurements:

- duration, sample rate, channel count;
- peak and RMS in dBFS;
- near-silence ratio using windowed RMS below a threshold;
- onset count and a tempo estimate for tracks at least one second long.

The tempo estimator is deliberately lightweight:

1. Compute a short-time Fourier transform of the mono mix (default window 1024,
   hop 256).
2. Build a log-magnitude spectral-flux onset envelope and smooth it.
3. Autocorrelate the envelope and search lags corresponding to 70-180 BPM.
4. Take the normalized autocorrelation peak as the confidence value, refine the
   lag parabolically, and count detected onset peaks.

The estimate is **not** a beat tracker. It does not identify downbeats, bar
phase, or absolute beat positions, and it can confuse half-time and double-time
patterns. The analyzer marks likely octave ambiguity and emits a warning note;
rhythmic planning additionally requires both estimates to exceed the confidence
threshold.

Configuration defaults:

| Setting | Default |
| --- | --- |
| STFT window / hop | 1024 / 256 frames |
| Silence threshold | -60 dBFS |
| Silence window | 0.05 s |
| Tempo range | 70-180 BPM |
| Tempo confidence threshold | 0.30 |
| Envelope smoothing | 5 frames |

### Optional librosa backend

`analysis/librosa_backend.py` adds beat tracking without making librosa a Core
dependency. The module is imported normally, but librosa itself is imported only
inside `analyze_track_librosa`; `librosa_available()` uses
`importlib.util.find_spec` without importing the package.

`analyze_track_librosa` first computes the full basic analysis, then:

1. returns the basic result unchanged for tracks shorter than one second or
   effectively silent;
2. runs `librosa.beat.beat_track` on the mono float32 signal with
   `units="time"` and `trim=False`;
3. keeps finite, in-duration beat times, deduplicates and sorts them, and
   requires at least four beats and a positive mean interval;
4. uses librosa's tempo but computes a conservative, cross-checked confidence.

The confidence is an engineering heuristic, not a calibrated probability:

| Relationship to the basic tempo estimate | Behaviour |
| --- | --- |
| Within 3% | `max(basic confidence, 0.80)` |
| Within 3% of 2x or 0.5x | `max(basic confidence, 0.65)`, marked ambiguous |
| No basic tempo but at least 8 regular beats | 0.55 |
| No basic tempo and weak evidence | 0.29, below the default threshold |
| Contradictory estimate | capped at 0.29 |

A confidence below the default 0.30 planning threshold prevents
`rhythmic` from using the estimate. A successful result sets
`analyzer="librosa"` and carries `beat_times_s`; every fallback retains
`analyzer="basic"` and an explanatory note. Downbeats and musical phrases are
never detected.

`analyze_track_with_backend(buffer, backend="auto" | "basic" | "librosa")`
selects the implementation. `auto` chooses librosa only when it can be found
and falls back to basic analysis with a note if librosa fails; explicitly
requesting librosa without the package, or when it fails, raises
`AnalysisError` with a clear message. The CLI defaults to `auto`, so the
optional backend is used when installed without making it a render dependency.
The library planner does not import librosa itself; callers pass analyses
produced by `analyze_track_with_backend` when they want beat data.

## Transition plans

`TransitionPlan` is a frozen data object and the central contract between
planning and rendering:

- selected strategy and originally requested strategy;
- transition start, duration, and end on the output timeline;
- a tuple of `StemAutomation` objects;
- human-readable reasoning and warnings;
- metadata and a versioned JSON representation (`schema_version: 1`).

Each `StemAutomation` identifies a source (`a` or `b`), a stem name or
`__mix__`, a curve (`smoothstep` or `linear`), an ordered tuple of `GainPoint`
values in dB, and optional effects. Gain-point times and effect times are
relative to the transition start. The transition start is also where Song B
begins in the output.

Validation rejects, among other things:

- missing or duplicate automations and missing coverage of either source;
- non-positive duration or inconsistent end time;
- fewer than two gain points, unordered points, or points outside the window;
- gains above the configured maximum or below the fade floor;
- unsupported curves or effects;
- effect cutoffs at or above the Nyquist limit when a sample rate is supplied;
- effect windows outside the transition.

The plan is validated again before rendering. Rendering additionally checks stem
coverage and that the transition ends no later than Song A.

## Strategies

The Core registry contains six implementations with distinct behavior:

### `crossfade`

Full-mix automation only, with linear-in-decibels fades across the transition.
Stems are ignored by design. The separate `render_crossfade` function is the
equal-power conventional baseline and is not the same code path.

### `stem_level`

Every supplied stem receives its own smooth fade. Timing is staggered by
fractional windows: Song A's vocals finish before Song B's vocals enter, bass is
de-emphasized earlier, drums overlap longer, and Song B's `other` layer enters
early. If both songs supply vocals, the plan reports the intentional vocal gap;
if a stem is missing, the plan does not claim vocal, drum, bass, or pad timing
for that side. Missing stems use the full mix as one stream. With no stems at
all, the strategy falls back to conservative behavior.

These windows are heuristic and deterministic. They are **not** derived from
detected vocal phrases, sections, or musical salience.

### `rhythmic`

Requires usable tempo estimates for both songs and a ratio of 1:1 or 2:1 within
a 2% tolerance.

When Song A's analysis contains detected beat times, the strategy searches
detected beats for the one nearest the requested transition start that still
leaves at least one whole bar within the duration bounds. The plan then starts
exactly on that beat, and every fade boundary uses an actual detected beat time
relative to it. The transition duration becomes the distance from that beat to
the end of Song A, so it can be a few milliseconds off an exact bar count; whole
bars are used only for the fade boundaries. If the detected grid is too short or
cannot fit a bar, the strategy falls back to whole-beat fractions of Song A's
tempo-derived grid.

Both paths assume 4/4. Neither detects downbeats or aligns Song B's internal
beat phase to Song A's: Song B's automation is placed on Song A's timing grid
while Song B starts at the transition boundary unwarped. No audio is
time-stretched. If the tempo estimates are unreliable or mismatched, the
strategy returns a conservative plan with the reason recorded. Plan metadata
records `alignment` (`detected_beats` or `tempo_fractions`), the analyzer names,
detected-beat counts, and the anchor beat time when used.

### `atmospheric`

Requires stems. It extends the overlap to 1.4 times the requested duration
(minimum 10 seconds), clamped to the shorter track. Vocals are kept apart; pads
overlap broadly. A two-stage one-pole low-pass sweep darkens Song A's `other`
stem while Song B's `other` stem opens up. If only one side has an `other` stem,
only that side is filtered, and a warning is recorded. Missing stems fall back
to conservative behavior.

### `conservative`

A predictable full-mix smooth fade. It makes no tempo, key, stem, or phrase
assumptions and is the final fallback for unavailable capabilities.

### `optimized` (RATO)

Generates a small, fixed set of role-timing candidates, renders each candidate,
and selects the lowest weighted objective. The objective combines normalized
engineering proxies: short-window loudness variation, gain-curve continuity,
vocal overlap, beat-grid distance, master-gain reduction, and automation
complexity. The `musical` term is currently a hard-coded zero placeholder; it
does not measure musical quality.

The candidate set is hand-designed rather than continuously optimized. It
includes the `stem_level` timing as its default candidate, so it ties that
strategy when no other candidate scores lower. On the synthetic demo, the
default candidate won and the rendered audio was effectively identical to
`stem_level`. RATO also renders every candidate, so it costs several times a
single plan render. The objective is not a perceptual model; listening tests
remain required.

`auto` selects:

1. `rhythmic` if tempo analysis is reliable for both songs;
2. otherwise `stem_level` if either song has stems;
3. otherwise `conservative`.

Every strategy returns constraints and warnings; none silently escalates to an
unimplemented capability.

## Rendering

### Baseline

`render_crossfade` adapts Song B to Song A's sample rate and channels, places
Song B so it starts `overlap_s` before Song A ends, and applies cosine/sine
equal-power fades. The output duration is `A + B - overlap`.

### Plan renderer

`render_plan` creates a zero-initialized output buffer whose length is:

```text
max(duration(A), plan.start_time_s + duration(B))
```

Song A is placed at frame 0 and Song B at `plan.start_time_s`. For each
automation it resolves either the full mix or a supplied stem, adapts the format,
applies effects, evaluates the gain envelope at each sample, and accumulates the
result.

Coverage rules are strict:

- A plan referencing stems without supplied stems is rejected.
- Every supplied stem must be covered by an automation, otherwise the renderer
  rejects the plan rather than silently dropping audio.
- A full-mix plan with supplied stems is allowed, but records that stems were
  not used.

The renderer does not mix stems and the full mix for the same source. Stems are
trusted as supplied; separation artifacts are not corrected.

### Effects

The only Core effect is `lowpass_sweep`. It is applied offline in short chunks
(default 256 samples) using two cascaded one-pole low-pass filters. Cutoff is
interpolated logarithmically between the start and end values over the effect
window. The final cutoff is held after the sweep. The filter is stable because
it operates on the input signal (not feedback across chunks) and carries the
filter state between chunks; it is not a linear-phase or high-order design.

### Finalization and clipping

`finalize_output` checks that every sample is finite. If the peak is above the
ceiling (default 0.999), it applies a single master gain reduction and records
the dB value. With clipping prevention disabled, the peak may exceed full scale
and the render is unsuitable for normal playback. A master gain is not applied
when it is not needed.

## Validation

`validation/checks.py` computes:

- non-empty frame/channel counts and finite-sample status;
- duration against an expected value (tolerance at least 0.05 seconds);
- expected sample rate and channels;
- peak level;
- near-full-scale ratio and a clipping check;
- longest interior silence using windowed RMS;
- maximum adjacent-sample step as a click/discontinuity heuristic;
- objective metrics including crest factor and DC offset.

Interior silence excludes runs touching either edge of the file. The default
gap limit is 3 seconds and the silence threshold is -70 dBFS. The discontinuity
limit is 1.5 in normalized sample units. These thresholds are heuristics, not
perceptual judgments.

`compare_audio` reports metric deltas and always includes a note that objective
metrics do not establish musical quality.

## Advanced boundary

`advanced.py` defines the optional extension point:

```text
importlib.metadata entry-point group: dcp.advanced.strategies
```

A separately installed package advertises an object (class or instance) with
`name`, `description`, and `plan(context)`. It may use the public
`TransitionContext`, `TransitionPlan`, `StemSet`, `TrackAnalysis`, and
`TransitionStrategy` types. Discovery:

- imports entry points locally;
- registers valid strategies into the same registry as Core;
- rejects duplicate Core names;
- records plugin errors;
- never prevents Core from rendering.

There is no Advanced implementation, license check, DRM, metering, billing, or
network access in Core. The boundary exists so a future proprietary package can
be installed optionally without modifying Core.

A future Advanced package might declare:

```toml
[project.entry-points."dcp.advanced.strategies"]
phrase_planner = "dcp_advanced.phrase:PhrasePlanner"
```

The plugin runs in the same process and is trusted code once installed. The
failure handling is robustness, not a sandbox.

### Commercial metering boundary (design only)

If a commercial edition later needs usage measurement, it should live outside
Core and be opt-in. A plausible local interface would count events such as
`transitions_rendered`, processed audio seconds, engine/version identifiers, and
strategy identifiers, with no audio content. No counters, storage, network
requests, or billing code exist in this prototype. Any metering design requires
legal and privacy review and explicit user authorization before activation.

## Offline rendering versus real-time playback

DCP Core renders complete files. It has no audio callback, ring buffer, device
I/O, or latency guarantee. A streaming player would need a separate integration
layer that:

- analyzes and plans ahead of time, preferably on a worker thread or offline;
- pre-renders or caches transition buffers before they are needed;
- keeps file access, model inference, and network calls out of the audio
  callback;
- falls back to a short precomputed crossfade if a transition is not ready.

Those pieces are not implemented. The current demo's runtime is not a real-time
performance claim.

## Testing strategy

The suite uses synthetic signals so it is reproducible and free of third-party
copyrighted audio. Tests cover structural validation, numeric safety, all Core
strategies, detected-beat and tempo-fraction planning, the optional librosa
backend, fallback paths, stem coverage, CLI exit codes, Advanced absence and
plugin failure, and the full demo. Librosa-specific tests skip when the optional
package is not installed, so Core remains testable with its required
dependencies alone. See [`testing.md`](testing.md).

## Security and privacy

- No network calls are made by Core.
- Audio is read from and written to local paths only.
- No credentials or secrets are required.
- Optional separator implementations must respect the documented local-only
  default unless the user explicitly authorizes an external service.
- Plugin discovery imports third-party code; installation is a trust decision.

## Known architectural limitations

- Tempo analysis is approximate; the basic analyzer has no beat tracker.
- Optional librosa beat phase is approximate as well, and downbeats, sections,
  and vocal phrases are not detected.
- Stem timing is rule-based, not phrase-aware.
- No time-stretching, key analysis, harmonic mixing, or adaptive EQ.
- Format support is whatever libsndfile decodes, limited by DCP to mono/stereo;
  only WAV output is documented and tested.
- Rendering is single-process and offline; no caching or incremental render yet.
- No listening/perceptual quality evaluation has been performed.

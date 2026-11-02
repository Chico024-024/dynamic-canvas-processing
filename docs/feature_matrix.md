# Feature Matrix

Statuses describe the **0.1 Core prototype** and the **planned** proprietary
edition. No Advanced feature is implemented in this repository. A "planned"
entry is a direction, not a delivery claim.

| Capability | DCP Core 0.1 | DCP Advanced (planned) | Evidence / notes |
| --- | --- | --- | --- |
| Local audio loading | Implemented, tested | Shared | Mono/stereo files through libsndfile; corrupt/missing inputs rejected. |
| WAV export | Implemented, tested | Shared | 24-bit PCM default; clipping guard. |
| Sample-rate conversion | Implemented, tested | Shared | Polyphase resampling; Song B/stems adapted to Song A. |
| Channel conversion | Implemented, tested | Shared | Mono-to-stereo and stereo-to-mono; other layouts rejected. |
| Duration, SR, channel, peak, RMS analysis | Implemented, tested | Shared | Deterministic measurements in dBFS. |
| Near-silence analysis | Implemented, tested | Shared | Windowed RMS below -60 dBFS by default. |
| Onset counting | Implemented, tested | Shared | Spectral-flux envelope. |
| Tempo estimation | Implemented, tested; approximate | Refined tracking planned | Basic autocorrelation over 70-180 BPM with confidence and octave warning; optional librosa estimate is cross-checked against it. |
| Beat position tracking | Optional librosa backend, tested | Refined tracking planned | Detected beat times when the `analysis` extra is installed; confidence is cross-checked and can be rejected. Not used when librosa is absent. |
| Downbeat / section tracking | Implemented, tested | Neural phrase tracking planned | Core implements energy-based downbeat estimation in 4/4 meter (`estimate_downbeats`) and bar-boundary snapping in `rhythmic`. Neural verse/chorus/drop tracking planned for Advanced. |
| Tempo-grid transition placement | Implemented, tested: detected beats, downbeat snap, whole-bar fractions | Phrase-aware planning planned | `rhythmic` starts on a detected downbeat/bar boundary when available and uses real beat times. Song B is not warped. |
| Conventional equal-power crossfade | Implemented, tested | Shared | Baseline renderer and CLI command. |
| Stem import and management | Implemented, tested | Shared | WAV directory per song; canonical and custom names. |
| Per-stem gain automation | Implemented, tested | Enhanced automation planned | Independent dB curves per source/stem. |
| Smooth fade curves | Implemented, tested | Additional curve models planned | Smoothstep and linear-in-dB. |
| Low-pass filter sweep | Implemented, tested | More advanced chains planned | Two cascaded one-pole stages, offline. |
| Parametric EQ / frequency-aware mixing | Not implemented | Planned | Core has no EQ bands, masking analysis, or spectral ducking. |
| Five rule-based strategies | Implemented, tested | More strategies planned | `crossfade`, `stem_level`, `rhythmic`, `atmospheric`, `conservative`. |
| Objective-based candidate search (RATO) | Implemented, tested | Richer optimization planned | Renders and scores five timing candidates; objective uses engineering proxies, not perceptual quality; `musical` term is inert. |
| Pluggable quality scorer | Implemented, tested | Learned scorer planned | `QualityScorer` contract; default `WeightedLossScorer` is the hand-tuned weighted sum. No learned model is shipped or loaded. |
| Local human A/B preference labeling | Implemented, tested | Training pipeline planned | `dcp label` renders candidate pairs, asks which is better, and appends features plus the choice to JSONL. Local only; no upload and no training. |
| Learned quality/ranking model | Not implemented | Planned | No training command, model artifact, scikit-learn/LightGBM dependency, or learned scorer exists in this milestone. |
| Automatic strategy selection | Implemented, tested | Intelligent selection/scoring planned | Rule-based `auto`: rhythmic, stem-level, conservative. |
| Plan reasoning and warnings | Implemented, tested | Richer diagnostics planned | Plans expose decisions, fallbacks, and uncertain analysis. |
| Plan JSON export | Implemented, tested | Versioned advanced plans planned | `schema_version: 1`. |
| Vocal phrase awareness | Not implemented | Planned | Core uses fixed fractional windows, not detected phrases. |
| Key detection / harmonic compatibility | Implemented, tested | Polyphonic key modulation planned | Chroma extraction + Krumhansl-Schmuckler key correlation, Camelot codes, harmonic distance & compatibility scoring in Core. |
| Tempo matching / time-stretching | Not implemented | Planned; quality/artifact sensitive | Songs are never warped in Core; only 1:1 or 2:1 tempo ratios accepted for rhythmic planning. Phase vocoding belongs to Advanced. |
| Source separation | Optional Demucs adapter, tested | Real-time cloud/GPU model planned | Concrete Demucs adapter with content-hash caching under `.dcp_cache/stems/`. Core works without it; users can supply stems or use `dcp separate`. |
| User-provided stems required as fallback | Implemented, tested | Still supported | Core works fully with supplied stems. |
| Perceptual transition quality scoring | Not implemented | Planned | Objective proxies and human preference collection exist; no perceptual score or trained model. |
| Preview comparison | Implemented, tested | In-app player integration planned | Interactive visual HTML previewer (`dcp visualize`) with A/B player and stem curves canvas, plus `compare` and `dcp label`. |
| Advanced plugin discovery | Implemented, tested | Proprietary package planned separately | Local entry points; broken/missing plugins never break Core. |
| Licensing/DRM/enforcement | Not implemented by design | Commercial terms under review | No checks, payments, metering, or network code. |
| Metering/billing | Not implemented | Design-only concepts | See [`licensing.md`](licensing.md); no data collection. |
| Real-time playback | Not implemented | Future integration research | Offline file rendering only. |
| Precompute/caching for playback | Not implemented | Planned | Architecture must keep heavy work out of audio callbacks. |
| Hosted/cloud processing | Not implemented | Optional future service | Core never uploads audio. |
| GPU acceleration | Not required, not used | Possible for future models | Milestone is CPU-only. |
| Multichannel (>2) audio | Not supported | Out of current scope | Mono/stereo enforced at file I/O. |
| Optional analysis dependencies | librosa extra; Core unaffected when absent | Shared | ISC-licensed librosa; pulls numba/llvmlite and is never required for Core rendering. |
| Automated test suite | 265 tests passing | Shared plus Advanced tests | See [`testing.md`](testing.md). |

## Core independence

DCP Core is a complete offline workflow on its own:

```bash
dcp synthesize --out output/synthetic
dcp crossfade output/synthetic/song_a.wav output/synthetic/song_b.wav \
  output/synthetic/baseline.wav --overlap 8
dcp render output/synthetic/song_a.wav output/synthetic/song_b.wav \
  output/synthetic/dcp.wav --stems-a output/synthetic/stems/a \
  --stems-b output/synthetic/stems/b --strategy stem_level --transition 8
```

The Advanced boundary is optional. In the verified environment no Advanced
package was installed, `dcp doctor` reported that condition, and the full demo
and all tests passed.

## What "Advanced" does not mean here

The repository does not contain a stub or fake Advanced implementation, does not
pretend proprietary strategies exist, and does not limit Core artificially. The
planned commercial capabilities above require separate design, implementation,
quality evaluation, licensing review, and commercial authorization.

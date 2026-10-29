# Roadmap

This is a direction document, not a promise of dates. The 0.1 Core prototype is
the only completed milestone described here.

## Milestone 0.1 - completed

Verified on 2026-10-29:

- synthetic two-song workflow with four stems per song;
- analysis, plans, equal-power crossfade, stem-level DCP rendering;
- optional librosa analysis backend for detected beats and a cross-checked
  tempo estimate, with detected-beat rhythmic planning;
- objective-based candidate search (`optimized`/RATO), with objective-only scoring;
- WAV output, objective validation, CLI, and demo;
- optional Advanced plugin boundary with no Advanced implementation;
- 183 automated tests passing;
- 24-second baseline and DCP demo outputs, both passing validation.

Detailed evidence is in the demo `report.json` and
[`testing.md`](testing.md).

## Three highest-priority next steps

### 1. Evaluate on real music with a listening-test protocol

The current demo uses synthetic, stylized material. Before adding more
algorithms, DCP needs evidence that its plan choices survive real recordings and
user-supplied stems.

Deliverables:

- a small, locally stored evaluation set with documented rights and no uploads;
- a repeatable A/B listening sheet comparing crossfade and DCP renders;
- objective measurements alongside listening outcomes, not instead of them;
- failure cases for vocal cut-offs, bass collisions, tempo drift, and stems with
  bleed;
- a regression report that distinguishes code failures from subjective
  preferences.

Exit criteria: the team can state which strategies help on which material, and
which claims are unsupported. No perceptual superiority is claimed until this
exists.

### 2. Spike an optional local source-separation backend

Keep `StemSeparator` replaceable and Core fully operational without it.

Deliverables:

- evaluate Demucs or a licensed alternative on CPU for runtime, memory, quality,
  and supported formats;
- review code, pretrained-weight, and training-data terms separately before
  integration;
- implement a thin adapter with explicit timeouts, cancellation, and a clear
  "separation unavailable" error;
- never download weights or send audio anywhere without explicit authorization;
- cache separation results locally by content hash where appropriate;
- verify DCP still passes all tests with the backend absent and broken.

Exit criteria: a user can opt into local separation with informed licensing and
resource expectations, or continue supplying stems manually with no regression.

### 3. Harden the production integration and precompute boundary

The offline renderer is not a real-time engine. The next integration step is to
make transitions precomputable and cacheable for a player.

Deliverables:

- a deterministic render manifest including inputs, plan hash, engine version,
  configuration, and output hashes;
- a cache keyed by song/stem content and transition parameters;
- clear separation between analysis/planning/render workers and any future audio
  callback;
- a fallback path that can use a pre-rendered conservative transition if a
  planned render is late;
- performance measurements on representative lengths and hardware;
- a stable API for an external player or Advanced package to request plans and
  buffers.

Exit criteria: DCP can supply a pre-rendered transition to a playback layer
without doing file access, model work, or network I/O in the callback. Real-time
gap-free playback remains unproven until actually built and tested.

## Later milestones

### 0.2 - Analysis and planning depth

- Better beat robustness on real music, including downbeat and bar-phase
  research beyond the optional librosa beat tracker.
- Key estimation and harmonic-compatibility experiments, clearly labeled as
  heuristic.
- Phrase and vocal-boundary detection with confidence and fallbacks.
- More expressive but bounded automation curves and EQ primitives.
- A documented evaluation corpus and regression metrics.

### 0.3 - Advanced edition prototype (separate proprietary repository)

Keep Core interfaces stable. Explore, behind the existing entry-point boundary:

- phrase-aware planning and intelligent strategy selection;
- frequency-aware mixing and improved stem automation;
- tempo-difference handling with quality safeguards;
- transition scoring and preview workflows;
- optimized rendering and production integrations;
- optional hosted or premium models only with explicit authorization and
  privacy review.

### 0.4 - Commercial readiness research

- Licensing, third-party rights, and royalty definitions reviewed by qualified
  counsel.
- Optional opt-in metering design with no audio content and no hidden network
  calls.
- Packaging, support, and enterprise integration requirements.
- Security review of plugin loading and any service boundary.

No billing, DRM, royalty enforcement, or metering is implemented now.

## Research questions

- Which measured features correlate with listeners judging a transition as
  intentional?
- How reliable are lightweight tempo estimates on real, syncopated music, and
  when should DCP refuse to use them?
- How much does stem bleed from separation degrade a stem-level transition?
- When is a conservative fallback preferable to an aggressive plan?
- What latency and memory budget would a streaming integration impose?
- Can quality scoring be calibrated without a large proprietary listening set?

## Explicit non-goals for the near term

- Training a source-separation model from scratch.
- Real-time, gap-free playback.
- DJ-deck control, timecode, or live performance features.
- Payment, licensing enforcement, or royalty collection.
- Uploading user audio to a service by default.
- Claiming musical intelligence or perceptual quality without evidence.

## Risks and open issues

- **Copyright and licenses:** user audio, model weights, datasets, and generated
  outputs have distinct rights. See [`licensing.md`](licensing.md).
- **Model licensing:** a permissively licensed separator codebase does not by
  itself settle pretrained-weight or training-data terms.
- **Quality measurement:** objective checks can pass while a transition sounds
  wrong, and vice versa.
- **Tempo ambiguity:** half/double-time confusion remains a known limitation.
- **Scope pressure:** adding models can compromise the CPU-only, offline
  baseline. Optional components must fail gracefully.

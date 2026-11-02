# DCP Advanced SDK: World-Class Automix Engine Specification

> **Status:** Production Architecture Blueprint  
> **Target Tier:** DCP Advanced SDK (Commercial / Royalty-Based Licensing Kit)  
> **Foundation:** Builds upon DCP Core 0.1 (265 tests passing, deterministic offline engine)

---

## 1. Executive Summary & Core vs. Advanced Boundary

DCP Core provides the open-source, deterministic, CPU-only audio foundation. **DCP Advanced SDK** is the proprietary, licensable SDK designed for integration into major music streaming platforms (**Spotify, Apple Music, Tidal, Amazon Music**), DJ software, and mobile audio apps.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DCP Two-Tier Architecture                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  DCP Core (Open Source Foundation - MIT / Apache-2.0)                       │
│  ├── Deterministic offline rendering, polyphase resampling, 24-bit WAV      │
│  ├── 6 Rule-Based Strategies + RATO Optimization (crossfade, stem, etc.)   │
│  ├── Chroma STFT Key Estimator & Energy Downbeat Snap (Krumhansl-Schmuckler)│
│  ├── Replaceable Contracts: StemSeparator, QualityScorer, TransitionStrategy │
│  └── Interactive Visualizer (dcp visualize / preview.html)                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  DCP Advanced SDK (Commercial Royalty Kit - Proprietary C++ / ONNX)         │
│  ├── ONNX-First Neural Perception (BeatNet, Silero-VAD, KeyNet, Structure)  │
│  ├── Structured Parameter Policy Network (TransitionParams generation)      │
│  ├── Sample-Accurate Downbeat-to-Downbeat Phase Locking                     │
│  ├── Bounded Transient-Preserving Time-Stretch (±8% limit + tempo ramp)     │
│  ├── Asynchronous Lookahead Precomputation Worker (30–45s budget)          │
│  ├── Cryptographic Telemetry Hook for Per-Stream Royalty Auditing          │
│  └── Multi-Factor Deterministic Cache Invalidation                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Critical Architectural Principles (The 4 Traps Resolved)

### 2.1 Trap 1: Structured Parameter Policy (No Autoregressive Curve Generation)
* **Problem:** Generating raw continuous `(time_s, gain_db)` tokens via an autoregressive transformer leads to non-monotonic wobbles, micro-discontinuities, clicks, and high latency.
* **Resolution:** The neural policy network outputs high-level structured discrete targets (`TransitionParams`):
  ```python
  class TransitionParams:
      strategy_type: str                  # e.g., "stem_level", "rhythmic_drop"
      duration_bars: int                  # Whole bars (e.g. 4, 8, 16)
      anchor_bar_a: int                   # Measure boundary in Song A
      cue_offset_b_s: float               # Song B start offset for phase lock
      vocal_handoff: tuple[float, float]  # (A_vocal_fade_out_s, B_vocal_fade_in_s)
      bass_crossover_bar: int             # Exact bar where low-end switches
      drum_blend_bars: int                # Drum overlap duration
      filter_target_hz: dict[str, float]  # Stem-specific cutoff frequencies
  ```
  The existing deterministic DCP curve engine translates these parameters into mathematically pure, smoothstep decibel curves. **Guarantees 100% click-free, monotonically stable transitions.**

### 2.2 Trap 2: Asynchronous Lookahead Precompute (Never in the Audio Thread)
* **Problem:** Source separation (Demucs) on CPU takes $0.8\times$ to $1.5\times$ realtime. Realtime separation in playback threads causes instant audio dropouts.
* **Resolution:** Streaming playback uses an **Asynchronous Lookahead Worker**:
  - When Song A reaches $70\%$ progress, a background thread launches precomputation.
  - Generates the transition buffer **30–45 seconds before Song A ends**.
  - If background worker misses deadline (e.g. heavy thermal throttling), the engine falls back to a beat-aligned full-mix crossfade seamlessly.

### 2.3 Trap 3: Downbeat-to-Downbeat Phase Locking
* **Problem:** Aligning Song A to its beat grid while starting Song B at $t=0$ creates phase clashes and polyrhythmic misalignment.
* **Resolution:** Measure both songs' downbeat positions and calculate the cue offset $t_{\text{cue\_b}}$:
  $$t_{\text{cue\_b}} = \text{downbeat\_times\_b}[k] - \text{lead\_in\_duration}$$
  Song B is cued so that **Beat 1 of Song B's incoming phrase lands with sub-millisecond phase accuracy on Beat 1 of Song A's outgoing phrase**.

### 2.4 Trap 4: Transient-Preserving Time-Stretch Bounds ($\pm 8\%$)
* **Problem:** Phase vocoding across arbitrary tempo ratios (e.g. 100 BPM to 140 BPM) destroys transients and smears percussion.
* **Resolution:**
  - If $|\Delta \text{BPM}| \le 8\%$: Apply phase-locked time-stretching (e.g., Rubber Band / Elastique) during the overlap window.
  - If $|\Delta \text{BPM}| > 8\%$:
    1. **Tempo Ramp:** Gradually ramp tempo across 4–8 bars prior to transition.
    2. **Energy Drop / Breakdown Cut:** Cut on Beat 1 of the measure with a reverb freeze tail rather than an overlapping blend.

---

## 3. Cache Invalidation & Telemetry Specification

### 3.1 Multi-Factor Cache Invalidation
To prevent stale renders when stems, parameters, or engines update:
$$\text{Cache Key} = \text{SHA256}\Big(\text{hash}(A) + \text{hash}(B) + \text{params} + \text{engine\_ver} + \text{config\_hash} + \text{stem\_ver}\Big)$$

Stored locally at `.dcp_cache/transitions/<cache_key>/`:
- `transition.wav` (rendered audio buffer)
- `plan.json` (exact execution plan)
- `validation.json` (objective metrics: peak, rms, silence, discontinuity checks)

### 3.2 Cryptographic Royalty Telemetry Hook
For commercial licensing agreements (e.g. streaming royalty per transition):
```json
{
  "event": "dcp_automix_rendered",
  "client_id": "client_app_spotify_prod",
  "transition_id": "dcp_9f82a1b4c3e7...",
  "track_a_isrc": "USUM71703841",
  "track_b_isrc": "GBAYE0601477",
  "duration_s": 8.0,
  "strategy": "neural_phrase_lock",
  "sample_rate": 44100,
  "timestamp_utc": 1798549200,
  "signature": "3045022100e4d1..."
}
```
Provides verifiable proof of automix generation for rights holders and streaming platform audits.

---

## 4. Phase Roadmap

### Phase 1: Neural Perception Foundation (ONNX-First)
- **1.1 Neural Beat & Downbeat Tracker:** BeatNet / TCN exported to ONNX Runtime.
- **1.2 Silero-VAD Vocal Activity Detection:** 1MB ONNX model running in $<2$ ms on CPU for sample-accurate vocal entry/exit boundaries.
- **1.3 Neural Key Detection & Calibration:** KeyNet/CREMA ONNX model with temperature scaling.
- **1.4 Structure & Phrase Segmentation:** CNN/RNN boundary detector (verse, chorus, breakdown, drop).

### Phase 2: Stem-Aware Transition Engine
- **2.1 Integrated Source Separation:** Content-hashed Demucs adapter with multi-stage fallback.
- **2.2 Structured Policy Network (`TransitionParams`):** Predicts discrete targets; deterministic smoothstep curve generation.
- **2.3 Professional Effect Rack:** 4-band Parametric EQ, soft-knee compressor, FDN reverb, tempo-synced delay.

### Phase 3: Learned Quality Ranking & Adaptive Selection
- **3.1 Pairwise Siamese Ranker:** Bradley-Terry model trained on `dcp label` JSONL human preference dataset.
- **3.2 Pareto-Optimal Candidate Search:** Evaluates multi-objective tradeoff without rigid scalar weights.
- **3.3 Meta-Learner Strategy Selection:** Context-aware automatic strategy selector.

### Phase 4: Harmonic Mixing & Phase Locking
- **4.1 Downbeat-to-Downbeat Phase Lock:** Cue offset calculation for measure alignment.
- **4.2 Bounded Time-Stretch ($\pm 8\%$):** Transient-preserving phase vocoding + tempo-ramp fallbacks.
- **4.3 Harmonic Energy Wheel:** Camelot relative/energy modulation plans.

### Phase 5: Real-Time Streaming & SDK Packaging
- **5.1 Streaming Pull Graph:** Fixed block size (256/512 frames), lock-free queues.
- **5.2 Lookahead Precomputation Worker:** 30–45s lookahead buffer at $70\%$ track progress.
- **5.3 C++ / ONNX Core & Language Bindings:** Python (`pybind11`), Swift (iOS/macOS), Kotlin (Android).
- **5.4 Royalty Telemetry & Licensing Engine:** Cryptographic audit trail.

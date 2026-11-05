/**
 * DCP Core Types - Apple Music Grade
 * Zero-dependency, real-time safe, deterministic
 */

#ifndef DCP_TYPES_H
#define DCP_TYPES_H

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Fundamental Types
// =============================================================================

typedef int32_t dcp_bool;
typedef int32_t dcp_int;
typedef int64_t dcp_long;
typedef float dcp_float;
typedef double dcp_double;
typedef const char* dcp_string;

// Real-time safe: no heap, fixed sizes
#define DCP_TRUE  1
#define DCP_FALSE 0
#define DCP_MAX_CHANNELS 8
#define DCP_MAX_STEMS 8
#define DCP_MAX_BREAKPOINTS 256
#define DCP_MAX_EFFECTS 16
#define DCP_MAX_PHRASES 64
#define DCP_MAX_BEATS 512
#define DCP_MAX_DOWNBEATS 128

// =============================================================================
// Error Codes
// =============================================================================

typedef enum {
    DCP_OK = 0,
    DCP_ERROR_INVALID_ARGUMENT = -1,
    DCP_ERROR_NULL_POINTER = -2,
    DCP_ERROR_BUFFER_TOO_SMALL = -3,
    DCP_ERROR_INVALID_SAMPLE_RATE = -4,
    DCP_ERROR_INVALID_CHANNEL_COUNT = -5,
    DCP_ERROR_OUT_OF_MEMORY = -6,
    DCP_ERROR_FILE_NOT_FOUND = -7,
    DCP_ERROR_INVALID_FORMAT = -8,
    DCP_ERROR_DECODER_FAILED = -9,
    DCP_ERROR_ENCODER_FAILED = -10,
    DCP_ERROR_MODEL_NOT_LOADED = -11,
    DCP_ERROR_INFERENCE_FAILED = -12,
    DCP_ERROR_INVALID_STATE = -13,
    DCP_ERROR_DENORMAL_DETECTED = -14,
    DCP_ERROR_CLIP_DETECTED = -15,
    DCP_ERROR_TIMING_DRIFT = -16,
    DCP_ERROR_UNSUPPORTED_FORMAT = -17,
    DCP_ERROR_LICENSE_INVALID = -18,
    DCP_ERROR_INTERNAL = -999
} dcp_error_t;

// =============================================================================
// Audio Buffer (Real-Time Safe)
// =============================================================================

typedef struct {
    dcp_float* channels[DCP_MAX_CHANNELS];  // Non-interleaved, channel-first
    dcp_int num_channels;
    dcp_int num_frames;
    dcp_int sample_rate;
    dcp_int64_t timestamp_ns;  // Sample-accurate timestamp
} dcp_audio_buffer_t;

// =============================================================================
// Gain Breakpoint (Sample-Accurate)
// =============================================================================

typedef struct {
    dcp_int frame_index;      // Absolute frame index from stream start
    dcp_float gain_db;        // Gain in decibels
    dcp_float curve;          // 0=linear, 1=smoothstep, 2=exp, 3=log, 4=scurve
} dcp_gain_breakpoint_t;

// =============================================================================
// Effect Specification
// =============================================================================

typedef enum {
    DCP_EFFECT_NONE = 0,
    DCP_EFFECT_ZDF_HPF = 1,
    DCP_EFFECT_ZDF_LPF = 2,
    DCP_EFFECT_ZDF_BPF = 3,
    DCP_EFFECT_ZDF_HPF_SWEEP = 4,
    DCP_EFFECT_ZDF_LPF_SWEEP = 5,
    DCP_EFFECT_ZDF_BPF_SWEEP = 6,
    DCP_EFFECT_FDN_FREEZE = 7,
    DCP_EFFECT_SIDECHAIN_DUCK = 8,
    DCP_EFFECT_SIDECHAIN_TRIGGER = 9,
    DCP_EFFECT_DUB_DELAY = 10,
    DCP_EFFECT_STEREO_WIDTH = 11,
    DCP_EFFECT_HARMONIC_SHIFT = 12,
    DCP_EFFECT_VOCAL_FORMANT = 13,
    DCP_EFFECT_GENERATIVE_RISER = 14,
    DCP_EFFECT_GENERATIVE_CRASH = 15,
    DCP_EFFECT_GENERATIVE_808 = 16,
    DCP_EFFECT_MULTIBAND_COMP = 17,
    DCP_EFFECT_SATURATION = 18,
    DCP_EFFECT_CLIP_GUARD = 18
} dcp_effect_type_t;

typedef struct {
    dcp_effect_type_t type;
    dcp_int frame_start;
    dcp_int frame_end;
    dcp_float params[16];  // Effect-specific parameters
    dcp_int param_count;
} dcp_effect_spec_t;

// =============================================================================
// Stem Automation
// =============================================================================

typedef struct {
    dcp_char source[16];                    // "A" or "B"
    dcp_char stem_name[32];                 // "vocals", "drums", "bass", "other"
    dcp_gain_breakpoint_t breakpoints[DCP_MAX_BREAKPOINTS];
    dcp_int breakpoint_count;
    dcp_effect_spec_t effects[DCP_MAX_EFFECTS];
    dcp_int effect_count;
} dcp_stem_automation_t;

// =============================================================================
// Transition Plan
// =============================================================================

typedef enum {
    DCP_STRATEGY_CONSERVATIVE = 0,
    DCP_STRATEGY_CROSSFADE = 1,
    DCP_STRATEGY_STEM_LEVEL = 2,
    DCP_STRATEGY_RHYTHMIC = 3,
    DCP_STRATEGY_ATMOSPHERIC = 4,
    DCP_STRATEGY_OPTIMIZED = 5,
    DCP_STRATEGY_STUDIO_REMIX = 6  // Song C
} dcp_strategy_t;

typedef enum {
    DCP_ARCHETYPE_VACUUM_BREATH_DROP = 0,
    DCP_ARCHETYPE_CONTINUOUS_CROSS_FILTER = 1,
    DCP_ARCHETYPE_POLYRHYTHMIC_4_3_PIVOT = 2,
    DCP_ARCHETYPE_GHOST_WASH_SUSPENSION = 3,
    DCP_ARCHETYPE_ADAPTIVE = 4
} dcp_archetype_t;

typedef struct {
    dcp_strategy_t strategy;
    dcp_archetype_t archetype;
    dcp_int total_bars;
    dcp_int lead_in_bars;
    dcp_int ghost_bars;
    dcp_float t_impact;           // Impact time (frames)
    dcp_float t_breath;           // Breath window start (frames)
    dcp_float t_vacuum;           // Vacuum window start (frames)
    dcp_float t_intro;            // Intro/build start (frames)
    dcp_float effective_bar_frames; // Frames per bar at effective tempo
    dcp_float warp_ratio_b;       // Song B time-stretch ratio
    dcp_int cue_offset_b_frames;  // Song B cue offset
    dcp_int launchpad_frame;      // Song A launchpad frame
    dcp_int impact_frame;         // Song B impact frame
    dcp_float master_bpm;
    dcp_float tempo_a;
    dcp_float tempo_b;
    dcp_float confidence;
    dcp_bool phase_locked;
    dcp_bool low_end_monogamy;
    dcp_bool harmonic_modulation;
    dcp_int key_transition_semitones;
    dcp_int bass_root_a;
    dcp_int bass_root_b;
} dcp_transition_plan_t;

// =============================================================================
// Track Musical IR (Intermediate Representation)
// =============================================================================

typedef struct {
    dcp_float bpm;
    dcp_float confidence;
    dcp_int downbeat_frames[DCP_MAX_DOWNBEATS];
    dcp_int downbeat_count;
    dcp_int beat_frames[DCP_MAX_BEATS];
    dcp_int beat_count;
    dcp_float accent_profile[16];  // 16-step per bar
} dcp_metrical_hypothesis_t;

typedef struct {
    dcp_int phrase_index;
    dcp_int bar_start;
    dcp_int bar_end;
    dcp_int frame_start;
    dcp_int frame_end;
    dcp_int structural_role;  // 0=intro, 1=verse, 2=peak, 3=breakdown, 4=outro
    dcp_float boundary_salience;
    dcp_float chroma[12];     // L2-normalized
    dcp_int bass_root_pc;     // 0-11
    dcp_float vocal_activity;
    dcp_bool vocal_resolves;
    dcp_float low_end_weight;
    dcp_float drum_crest;
} dcp_phrase_ir_t;

typedef struct {
    dcp_char track_id[64];
    dcp_int sample_rate;
    dcp_int duration_frames;
    dcp_float base_bpm;
    dcp_metrical_hypothesis_t hypotheses[4];  // primary, half, double, 4:3
    dcp_int hypothesis_count;
    dcp_phrase_ir_t phrases[DCP_MAX_PHRASES];
    dcp_int phrase_count;
} dcp_track_ir_t;

// =============================================================================
// Composer Plan (Output of Musical Brain)
// =============================================================================

typedef struct {
    dcp_archetype_t archetype;
    dcp_int total_bars;
    dcp_int lead_in_bars;
    dcp_int ghost_bars;
    dcp_int launchpad_phrase_idx;
    dcp_int impact_phrase_idx;
    dcp_int launchpad_frame;
    dcp_int impact_frame;
    dcp_int key_shift_semitones;
    dcp_int bass_root_a;
    dcp_int bass_root_b;
    dcp_float t_impact;
    dcp_float t_breath;
    dcp_float t_vacuum;
    dcp_float t_intro;
    dcp_float effective_bar_frames;
    dcp_float warp_ratio_b;
    dcp_int cue_offset_b_frames;
    dcp_float master_bpm;
    dcp_bool phase_locked;
    dcp_bool low_end_monogamy;
    dcp_bool harmonic_modulation;
    dcp_float confidence;
    dcp_char rationale[256];
} dcp_composer_plan_t;

// =============================================================================
// Configuration
// =============================================================================

typedef struct {
    // Audio
    dcp_int target_sample_rate;       // 0 = auto
    dcp_int max_channels;
    dcp_int render_block_size;        // Frames per process block (64, 128, 256, 512)

    // Stem Separation
    dcp_bool prefer_neural_stems;     // Try ONNX Demucs first
    dcp_bool cache_stems;             // Cache to disk
    dcp_char cache_dir[256];

    // Musical Analysis
    dcp_float min_tempo_confidence;   // 0.0-1.0
    dcp_int min_downbeats;            // Minimum for phrase detection
    dcp_bool enable_genre_classification;

    // Composer
    dcp_int max_bridge_bars;          // 4-32
    dcp_int min_bridge_bars;
    dcp_bool enable_harmonic_modulation;
    dcp_bool enable_generative_layer; // Riser, crash, 808, chops

    // DSP
    dcp_bool oversample_2x;           // 2x for non-linear stages
    dcp_bool oversample_4x;           // 4x for saturation
    dcp_float dither_amount;          // TPDF dither (-1 = auto)
    dcp_bool noise_shaping;           // 1st-order noise shaping
    dcp_bool clip_guard;              // Per-node clip protection

    // Master Bus
    dcp_bool master_bus_enabled;
    dcp_float master_lufs_target;     // -14 for streaming
    dcp_float master_true_peak_ceiling; // -1.0 dBTP
    dcp_bool master_saturation;       // Parallel tube/tape

    // Performance
    dcp_int max_threads;              // 0 = auto
    dcp_bool realtime_mode;           // Zero allocations, fixed latency
    dcp_int max_latency_ms;           // For buffering decisions

    // Telemetry
    dcp_bool telemetry_enabled;
    dcp_char telemetry_endpoint[256];
} dcp_config_t;

// =============================================================================
// Context Handles (Opaque)
// =============================================================================

typedef struct dcp_context_s* dcp_context_t;
typedef struct dcp_stem_separator_s* dcp_stem_separator_t;
typedef struct dcp_composer_s* dcp_composer_t;
typedef struct dcp_pipeline_s* dcp_pipeline_t;
typedef struct dcp_dsp_s* dcp_dsp_t;

#ifdef __cplusplus
}
#endif

#endif // DCP_TYPES_H
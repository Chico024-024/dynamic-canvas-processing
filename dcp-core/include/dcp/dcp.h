/**
 * DCP Core Public API - Apple Music Grade
 * Main entry point for the Dynamic Canvas Processing library
 */

#ifndef DCP_H
#define DCP_H

#include "dcp_types.h"
#include "dcp_config.h"
#include "dcp_errors.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Version Information
// =============================================================================

#define DCP_VERSION_MAJOR 1
#define DCP_VERSION_MINOR 0
#define DCP_VERSION_PATCH 0
#define DCP_VERSION_STRING "1.0.0"

/**
 * Get library version
 * @param major Output major version
 * @param minor Output minor version
 * @param patch Output patch version
 */
void dcp_version(int* major, int* minor, int* patch);

/**
 * Get version string
 * @return Static version string
 */
const char* dcp_version_string(void);

// =============================================================================
// Global Initialization / Shutdown
// =============================================================================

/**
 * Initialize library (call once at startup)
 * @param config Optional configuration (NULL for defaults)
 * @return DCP_OK on success
 */
dcp_error_t dcp_initialize(const dcp_config_t* config);

/**
 * Shutdown library (call once at shutdown)
 */
void dcp_shutdown(void);

/**
 * Check if library is initialized
 * @return DCP_TRUE if initialized
 */
dcp_bool dcp_is_initialized(void);

// =============================================================================
// Context Management
// =============================================================================

/**
 * Create processing context
 * @param config Configuration (NULL for global defaults)
 * @param out_context Output context handle
 * @return DCP_OK on success
 */
dcp_error_t dcp_context_create(const dcp_config_t* config, dcp_context_t* out_context);

/**
 * Destroy context
 * @param context Context to destroy
 */
void dcp_context_destroy(dcp_context_t context);

/**
 * Get context configuration
 * @param context Context handle
 * @return Configuration pointer (valid while context exists)
 */
const dcp_config_t* dcp_context_get_config(dcp_context_t context);

// =============================================================================
// Audio Buffer Management
// =============================================================================

/**
 * Create audio buffer
 * @param context Context handle
 * @param num_channels Number of channels (1-8)
 * @param num_frames Number of frames
 * @param sample_rate Sample rate
 * @param out_buffer Output buffer handle
 * @return DCP_OK on success
 */
dcp_error_t dcp_buffer_create(
    dcp_context_t context,
    dcp_int num_channels,
    dcp_int num_frames,
    dcp_int sample_rate,
    dcp_audio_buffer_t* out_buffer
);

/**
 * Destroy audio buffer
 * @param buffer Buffer to destroy
 */
void dcp_buffer_destroy(dcp_audio_buffer_t* buffer);

/**
 * Clear audio buffer (zero samples)
 * @param buffer Buffer to clear
 */
void dcp_buffer_clear(dcp_audio_buffer_t* buffer);

/**
 * Copy audio buffer
 * @param src Source buffer
 * @param dst Destination buffer (must have same layout)
 * @return DCP_OK on success
 */
dcp_error_t dcp_buffer_copy(const dcp_audio_buffer_t* src, dcp_audio_buffer_t* dst);

// =============================================================================
// Stem Separation
// =============================================================================

/**
 * Create stem separator
 * @param context Context handle
 * @param out_separator Output separator handle
 * @return DCP_OK on success
 */
dcp_error_t dcp_stem_separator_create(dcp_context_t context, dcp_stem_separator_t* out_separator);

/**
 * Destroy stem separator
 * @param separator Separator to destroy
 */
void dcp_stem_separator_destroy(dcp_stem_separator_t separator);

/**
 * Separate audio into stems
 * @param separator Separator handle
 * @param input Input audio buffer
 * @param stems Output stem buffers (must be pre-allocated)
 * @param stem_count Number of stems (max DCP_MAX_STEMS)
 * @return DCP_OK on success
 */
dcp_error_t dcp_stem_separator_separate(
    dcp_stem_separator_t separator,
    const dcp_audio_buffer_t* input,
    dcp_audio_buffer_t* stems,
    dcp_int* stem_count
);

// =============================================================================
// Musical Analysis
// =============================================================================

/**
 * Create musical analyzer
 * @param context Context handle
 * @return Analyzer handle (opaque)
 */
typedef struct dcp_analyzer_s* dcp_analyzer_t;

dcp_error_t dcp_analyzer_create(dcp_context_t context, dcp_analyzer_t* out_analyzer);
void dcp_analyzer_destroy(dcp_analyzer_t analyzer);

/**
 * Analyze track and produce Musical IR
 * @param analyzer Analyzer handle
 * @param input Input audio buffer
 * @param out_ir Output Musical IR
 * @return DCP_OK on success
 */
dcp_error_t dcp_analyzer_analyze(
    dcp_analyzer_t analyzer,
    const dcp_audio_buffer_t* input,
    dcp_track_ir_t* out_ir
);

// =============================================================================
// Composer (Song C)
// =============================================================================

/**
 * Create composer
 * @param context Context handle
 * @param out_composer Output composer handle
 * @return DCP_OK on success
 */
dcp_error_t dcp_composer_create(dcp_context_t context, dcp_composer_t* out_composer);
void dcp_composer_destroy(dcp_composer_t composer);

/**
 * Compose transition bridge (Song C)
 * @param composer Composer handle
 * @param ir_a Musical IR for Song A
 * @param ir_b Musical IR for Song B
 * @param out_plan Output composer plan
 * @return DCP_OK on success
 */
dcp_error_t dcp_composer_compose(
    dcp_composer_t composer,
    const dcp_track_ir_t* ir_a,
    const dcp_track_ir_t* ir_b,
    dcp_composer_plan_t* out_plan
);

// =============================================================================
// Pipeline (Rendering)
// =============================================================================

/**
 * Create rendering pipeline
 * @param context Context handle
 * @param plan Transition plan
 * @param out_pipeline Output pipeline handle
 * @return DCP_OK on success
 */
dcp_error_t dcp_pipeline_create(
    dcp_context_t context,
    const dcp_transition_plan_t* plan,
    dcp_pipeline_t* out_pipeline
);
void dcp_pipeline_destroy(dcp_pipeline_t pipeline);

/**
 * Render transition
 * @param pipeline Pipeline handle
 * @param stems_a Song A stems (array of DCP_MAX_STEMS buffers, NULL for missing)
 * @param stems_b Song B stems (array of DCP_MAX_STEMS buffers, NULL for missing)
 * @param output Output buffer (pre-allocated)
 * @return DCP_OK on success
 */
dcp_error_t dcp_pipeline_render(
    dcp_pipeline_t pipeline,
    const dcp_audio_buffer_t* stems_a,
    const dcp_audio_buffer_t* stems_b,
    dcp_audio_buffer_t* output
);

/**
 * Process single block (real-time streaming)
 * @param pipeline Pipeline handle
 * @param input Input block
 * @param output Output block (same size as input)
 * @return DCP_OK on success, DCP_ERROR_INVALID_STATE if pipeline not ready
 */
dcp_error_t dcp_pipeline_process_block(
    dcp_pipeline_t pipeline,
    const dcp_audio_buffer_t* input,
    dcp_audio_buffer_t* output
);

// =============================================================================
// DSP Effects (Low-Level)
// =============================================================================

/**
 * Create DSP effect processor
 * @param context Context handle
 * @param out_dsp Output DSP handle
 * @return DCP_OK on success
 */
dcp_error_t dcp_dsp_create(dcp_context_t context, dcp_dsp_t* out_dsp);
void dcp_dsp_destroy(dcp_dsp_t dsp);

/**
 * ZDF State Variable Filter process
 * @param dsp DSP handle
 * @param input Input buffer
 * @param output Output buffer
 * @param cutoff_hz Filter cutoff frequency
 * @param resonance_q Resonance/Q factor (0.5-12.0)
 * @param filter_type 0=LPF, 1=HPF, 2=BPF, 3=HPF4
 * @param mode 0=static, 1=sweep
 * @param cutoff_end_hz End cutoff for sweep mode
 * @return DCP_OK on success
 */
dcp_error_t dcp_dsp_zdf_svf(
    dcp_dsp_t dsp,
    const dcp_audio_buffer_t* input,
    dcp_audio_buffer_t* output,
    dcp_float cutoff_hz,
    dcp_float resonance_q,
    dcp_int filter_type,
    dcp_int mode,
    dcp_float cutoff_end_hz
);

/**
 * FDN Freeze process
 * @param dsp DSP handle
 * @param input Input buffer
 * @param output Output buffer
 * @param freeze_engage_frame Frame to engage freeze
 * @param freeze_release_frame Frame to release freeze
 * @param feedback Feedback gain (0.0-0.95)
 * @param wet_mix Wet mix (0.0-1.0)
 * @param dry_mix Dry mix (0.0-1.0)
 * @return DCP_OK on success
 */
dcp_error_t dcp_dsp_fdn_freeze(
    dcp_dsp_t dsp,
    const dcp_audio_buffer_t* input,
    dcp_audio_buffer_t* output,
    dcp_int freeze_engage_frame,
    dcp_int freeze_release_frame,
    dcp_float feedback,
    dcp_float wet_mix,
    dcp_float dry_mix
);

/**
 * Sidechain compressor process
 * @param dsp DSP handle
 * @param target Target buffer (to be ducked)
 * @param key Key buffer (trigger)
 * @param output Output buffer
 * @param bpm Tempo for release timing
 * @param threshold_db Threshold in dB
 * @param ratio Compression ratio
 * @param attack_ms Attack time in ms
 * @param release_beat_fraction Release as beat fraction
 * @return DCP_OK on success
 */
dcp_error_t dcp_dsp_sidechain(
    dcp_dsp_t dsp,
    const dcp_audio_buffer_t* target,
    const dcp_audio_buffer_t* key,
    dcp_audio_buffer_t* output,
    dcp_float bpm,
    dcp_float threshold_db,
    dcp_float ratio,
    dcp_float attack_ms,
    dcp_float release_beat_fraction
);

/**
 * Phase vocoder time-stretch
 * @param dsp DSP handle
 * @param input Input buffer
 * @param output Output buffer
 * @param stretch_ratio Stretch ratio (0.5-2.0)
 * @return DCP_OK on success
 */
dcp_error_t dcp_dsp_time_stretch(
    dcp_dsp_t dsp,
    const dcp_audio_buffer_t* input,
    dcp_audio_buffer_t* output,
    dcp_float stretch_ratio
);

// =============================================================================
// Master Bus Processing
// =============================================================================

/**
 * Process master bus (glue compressor + limiter + dither)
 * @param dsp DSP handle
 * @param input Input buffer
 * @param output Output buffer
 * @param threshold_db Compressor threshold
 * @param ratio Compressor ratio
 * @param attack_ms Attack time
 * @param release_ms Release time
 * @param ceiling_dbfs True peak ceiling
 * @param lufs_target Target LUFS
 * @param saturation_enabled Enable parallel saturation
 * @param dither_amount Dither amount (-1=auto)
 * @param noise_shaping Enable noise shaping
 * @return DCP_OK on success
 */
dcp_error_t dcp_dsp_master_bus(
    dcp_dsp_t dsp,
    const dcp_audio_buffer_t* input,
    dcp_audio_buffer_t* output,
    dcp_float threshold_db,
    dcp_float ratio,
    dcp_float attack_ms,
    dcp_float release_ms,
    dcp_float ceiling_dbfs,
    dcp_float lufs_target,
    dcp_bool saturation_enabled,
    dcp_float dither_amount,
    dcp_bool noise_shaping
);

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Convert dB to linear gain
 */
static inline dcp_float dcp_db_to_linear(dcp_float db) {
    return powf(10.0f, db * 0.05f);
}

/**
 * Convert linear gain to dB
 */
static inline dcp_float dcp_linear_to_db(dcp_float linear) {
    return 20.0f * log10f(fmaxf(linear, 1e-12f));
}

/**
 * Get optimal block size for sample rate
 */
dcp_int dcp_optimal_block_size(dcp_int sample_rate);

/**
 * Calculate frames from time
 */
static inline dcp_int dcp_time_to_frames(dcp_float time_sec, dcp_int sample_rate) {
    return (dcp_int)lrintf(time_sec * sample_rate);
}

/**
 * Calculate time from frames
 */
static inline dcp_float dcp_frames_to_time(dcp_int frames, dcp_int sample_rate) {
    return (dcp_float)frames / (dcp_float)sample_rate;
}

#ifdef __cplusplus
}
#endif

#endif // DCP_H
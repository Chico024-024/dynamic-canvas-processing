/**
 * Transient-Decoupled Sidechain Compressor - Apple Music Grade
 * Decoupled peak/RMS detection, fast attack, beat-sync release
 * Real-time safe, zero allocations, sample-accurate
 */

#ifndef DCP_SIDECHAIN_H
#define DCP_SIDECHAIN_H

#include "../dcp_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Configuration
// =============================================================================

typedef struct {
    // Detection
    dcp_float attack_ms;              // Attack time in ms (0.1 - 100)
    dcp_float release_ms;             // Release time in ms (1 - 5000)
    dcp_float release_beat_fraction;  // Alternative: release as beat fraction (0.0625 - 1.0)
    dcp_float bpm;                    // Tempo for beat-sync release

    // Threshold & Ratio
    dcp_float threshold_db;           // Threshold in dB (-60 to 0)
    dcp_float ratio;                  // Compression ratio (1.0 - inf, use 20.0 for limiting)
    dcp_float knee_width_db;          // Knee width in dB (0 = hard knee)

    // Detection blend
    dcp_float peak_rms_blend;         // 0 = pure RMS, 1 = pure peak (0.5-0.9 typical)
    dcp_float rms_window_ms;          // RMS window size (2 - 50ms)

    // Sidechain filter
    dcp_bool sc_filter_enabled;       // Enable sidechain filter
    dcp_float sc_hp_hz;               // Sidechain high-pass (20-200 Hz)
    dcp_float sc_lp_hz;               // Sidechain low-pass (100-500 Hz)

    // Stereo linking
    dcp_bool stereo_link;             // Link L/R channels

    // Lookahead
    dcp_float lookahead_ms;           // Lookahead for catch transient (0 - 5ms)

    // Makeup gain
    dcp_float makeup_gain_db;         // Auto makeup gain (0 = auto)

    // Block processing
    dcp_int block_size;               // Internal block size for envelope (16-128)
} dcp_sidechain_config_t;

// =============================================================================
// Sidechain Handle
// =============================================================================

typedef struct dcp_sidechain_s* dcp_sidechain_t;

// =============================================================================
// API
// =============================================================================

/**
 * Create sidechain compressor
 * @param sc Output sidechain handle
 * @param config Configuration
 * @param sample_rate Sample rate in Hz
 * @param max_channels Maximum channels (1-8)
 * @return DCP_OK on success
 */
dcp_error_t dcp_sidechain_create(
    dcp_sidechain_t* sc,
    const dcp_sidechain_config_t* config,
    dcp_int sample_rate,
    dcp_int max_channels
);

/**
 * Destroy sidechain compressor
 * @param sc Sidechain handle
 */
void dcp_sidechain_destroy(dcp_sidechain_t sc);

/**
 * Process sidechain compression
 * @param sc Sidechain handle
 * @param target Target audio (to be ducked)
 * @param key Key audio (sidechain trigger)
 * @param output Output buffer (can be same as target for in-place)
 * @param num_frames Number of frames
 * @return DCP_OK on success
 */
dcp_error_t dcp_sidechain_process(
    dcp_sidechain_t sc,
    const dcp_audio_buffer_t* target,
    const dcp_audio_buffer_t* key,
    dcp_audio_buffer_t* output,
    dcp_int num_frames
);

/**
 * Process with pre-computed key envelope (for efficiency)
 * @param sc Sidechain handle
 * @param target Target audio (to be ducked)
 * @param key_envelope Pre-computed key envelope (1 per frame)
 * @param output Output buffer
 * @param num_frames Number of frames
 * @return DCP_OK on success
 */
dcp_error_t dcp_sidechain_process_with_envelope(
    dcp_sidechain_t sc,
    const dcp_audio_buffer_t* target,
    const dcp_float* key_envelope,
    dcp_audio_buffer_t* output,
    dcp_int num_frames
);

/**
 * Generate key envelope from key audio (for reuse)
 * @param sc Sidechain handle
 * @param key Key audio
 * @param envelope Output envelope buffer (1 value per frame)
 * @param num_frames Number of frames
 * @return DCP_OK on success
 */
dcp_error_t dcp_sidechain_generate_envelope(
    dcp_sidechain_t sc,
    const dcp_audio_buffer_t* key,
    dcp_float* envelope,
    dcp_int num_frames
);

/**
 * Set BPM for beat-sync release (real-time safe)
 * @param sc Sidechain handle
 * @param bpm Tempo in BPM
 * @return DCP_OK on success
 */
dcp_error_t dcp_sidechain_set_bpm(dcp_sidechain_t sc, dcp_float bpm);

/**
 * Set threshold (real-time safe)
 * @param sc Sidechain handle
 * @param threshold_db Threshold in dB
 * @return DCP_OK on success
 */
dcp_error_t dcp_sidechain_set_threshold(dcp_sidechain_t sc, dcp_float threshold_db);

/**
 * Set ratio (real-time safe)
 * @param sc Sidechain handle
 * @param ratio Compression ratio
 * @return DCP_OK on success
 */
dcp_error_t dcp_sidechain_set_ratio(dcp_sidechain_t sc, dcp_float ratio);

/**
 * Reset sidechain state
 * @param sc Sidechain handle
 */
void dcp_sidechain_reset(dcp_sidechain_t sc);

/**
 * Get current gain reduction in dB (for metering)
 * @param sc Sidechain handle
 * @param channel Channel index
 * @return Gain reduction in dB (positive = reduction)
 */
dcp_float dcp_sidechain_get_gain_reduction(const dcp_sidechain_t sc, dcp_int channel);

#ifdef __cplusplus
}
#endif

#endif // DCP_SIDECHAIN_H
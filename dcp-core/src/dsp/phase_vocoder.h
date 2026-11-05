/**
 * Phase-Locked STFT Phase Vocoder - Apple Music Grade
 * Sample-accurate, real-time safe, bounded stretch (±8%)
 * Based on Laroche & Dolson, Puckette, and Zölzer
 */

#ifndef DCP_PHASE_VOCODER_H
#define DCP_PHASE_VOCODER_H

#include "../dcp_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Configuration
// =============================================================================

typedef struct {
    // STFT parameters
    dcp_int fft_size;           // Power of 2 (128 - 4096)
    dcp_int hop_size;           // FFT size / 4 typical
    dcp_int oversampling;       // 1, 2, or 4x for reduced aliasing

    // Stretch bounds
    dcp_float min_stretch;      // 0.92 (8% faster)
    dcp_float max_stretch;      // 1.08 (8% slower)

    // Phase locking
    dcp_bool phase_locking;     // Enable phase locking
    dcp_float phase_lock_threshold; // 0.0-1.0

    // Window
    typedef enum {
        DCP_WINDOW_HANN = 0,
        DCP_WINDOW_KAISER = 1,
        DCP_WINDOW_BLACKMAN_HARRIS = 2
    } dcp_window_type_t;

    dcp_window_type_t window_type;
    dcp_float kaiser_beta;      // For Kaiser window

    // Quality
    dcp_bool high_quality;      // Extra phase unwrapping passes
} dcp_phase_vocoder_config_t;

// =============================================================================
// Phase Vocoder Handle
// =============================================================================

typedef struct dcp_phase_vocoder_s* dcp_phase_vocoder_t;

// =============================================================================
// API
// =============================================================================

/**
 * Create phase vocoder
 * @param pv Output phase vocoder handle
 * @param config Configuration
 * @param sample_rate Sample rate in Hz
 * @param max_frames Maximum frames per process block
 * @return DCP_OK on success
 */
dcp_error_t dcp_phase_vocoder_create(
    dcp_phase_vocoder_t* pv,
    const dcp_phase_vocoder_config_t* config,
    dcp_int sample_rate,
    dcp_int max_frames
);

/**
 * Destroy phase vocoder
 * @param pv Phase vocoder handle
 */
void dcp_phase_vocoder_destroy(dcp_phase_vocoder_t pv);

/**
 * Time-stretch audio with constant ratio
 * @param pv Phase vocoder handle
 * @param input Input buffer
 * @param output Output buffer (pre-allocated)
 * @param stretch_ratio Stretch ratio (0.92 - 1.08)
 * @return DCP_OK on success
 */
dcp_error_t dcp_phase_vocoder_stretch_constant(
    dcp_phase_vocoder_t pv,
    const dcp_audio_buffer_t* input,
    dcp_audio_buffer_t* output,
    dcp_float stretch_ratio
);

/**
 * Time-stretch audio with time-varying ratio
 * @param pv Phase vocoder handle
 * @param input Input buffer
 * @param output Output buffer (pre-allocated)
 * @param stretch_ratios Array of stretch ratios (one per output frame)
 * @param ratio_count Number of ratios
 * @return DCP_OK on success
 */
dcp_error_t dcp_phase_vocoder_stretch_variable(
    dcp_phase_vocoder_t pv,
    const dcp_audio_buffer_t* input,
    dcp_audio_buffer_t* output,
    const dcp_float* stretch_ratios,
    dcp_int ratio_count
);

/**
 * Process single frame (streaming, advanced use)
 * @param pv Phase vocoder handle
 * @param input_frame Input frame (deinterleaved, 1 per channel)
 * @param output_frame Output frame (deinterleaved)
 * @param stretch_ratio Current stretch ratio
 * @return DCP_OK on success
 */
dcp_error_t dcp_phase_vocoder_process_frame(
    dcp_phase_vocoder_t pv,
    const dcp_float* input_frame,
    dcp_float* output_frame,
    dcp_float stretch_ratio
);

/**
 * Reset phase vocoder state
 * @param pv Phase vocoder handle
 */
void dcp_phase_vocoder_reset(dcp_phase_vocoder_t pv);

/**
 * Get current latency in frames
 * @param pv Phase vocoder handle
 * @return Latency in frames
 */
dcp_int dcp_phase_vocoder_get_latency(const dcp_phase_vocoder_t pv);

#ifdef __cplusplus
}
#endif

#endif // DCP_PHASE_VOCODER_H
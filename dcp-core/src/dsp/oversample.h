/**
 * Oversampling (Polyphase IIR) - Apple Music Grade
 * 2x, 4x, 8x oversampling for aliasing-free non-linear processing
 * Polyphase IIR filters, real-time safe, zero allocations
 */

#ifndef DCP_OVERSAMPLE_H
#define DCP_OVERSAMPLE_H

#include "../dcp_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Configuration
// =============================================================================

typedef enum {
    DCP_OVERSAMPLE_2X = 2,
    DCP_OVERSAMPLE_4X = 4,
    DCP_OVERSAMPLE_8X = 8
} dcp_oversample_factor_t;

typedef enum {
    DCP_OVERSAMPLE_LINEAR = 0,
    DCP_OVERSAMPLE_MINIMUM_PHASE = 1,
    DCP_OVERSAMPLE_LINEAR_PHASE = 2
} dcp_oversample_mode_t;

typedef struct {
    dcp_oversample_factor_t factor;
    dcp_oversample_mode_t mode;
    dcp_int filter_order;        // 4, 6, 8, 10 (higher = better alias rejection)
    dcp_float stopband_attenuation_db;  // Target stopband attenuation
    dcp_bool zero_latency;       // Use minimum-phase for zero latency
} dcp_oversample_config_t;

// =============================================================================
// Oversampling Handle
// =============================================================================

typedef struct dcp_oversample_s* dcp_oversample_t;

// =============================================================================
// API
// =============================================================================

/**
 * Create oversampler
 * @param os Output oversampler handle
 * @param config Configuration
 * @param sample_rate Base sample rate
 * @param num_channels Number of channels
 * @return DCP_OK on success
 */
dcp_error_t dcp_oversample_create(
    dcp_oversample_t* os,
    const dcp_oversample_config_t* config,
    dcp_int sample_rate,
    dcp_int num_channels
);

/**
 * Destroy oversampler
 * @param os Oversampler handle
 */
void dcp_oversample_destroy(dcp_oversample_t os);

/**
 * Process: upsample, process callback, downsample
 * @param os Oversampler handle
 * @param input Input buffer at base sample rate
 * @param output Output buffer at base sample rate
 * @param num_frames Number of frames at base sample rate
 * @param process_cb Callback to process at high sample rate
 * @param user_data User data passed to callback
 * @return DCP_OK on success
 */
dcp_error_t dcp_oversample_process(
    dcp_oversample_t os,
    const dcp_audio_buffer_t* input,
    dcp_audio_buffer_t* output,
    dcp_int num_frames,
    dcp_error_t (*process_cb)(dcp_audio_buffer_t* high_rate, void* user_data),
    void* user_data
);

/**
 * Upsample only (input at base rate, output at high rate)
 * @param os Oversampler handle
 * @param input Input buffer at base rate
 * @param output Output buffer at high rate (size = num_frames * factor)
 * @param num_frames Input frames at base rate
 * @return DCP_OK on success
 */
dcp_error_t dcp_oversample_upsample(
    dcp_oversample_t os,
    const dcp_audio_buffer_t* input,
    dcp_audio_buffer_t* output,
    dcp_int num_frames
);

/**
 * Downsample only (input at high rate, output at base rate)
 * @param os Oversampler handle
 * @param input Input buffer at high rate (size = num_frames * factor)
 * @param output Output buffer at base rate
 * @param num_frames Output frames at base rate
 * @return DCP_OK on success
 */
dcp_error_t dcp_oversample_downsample(
    dcp_oversample_t os,
    const dcp_audio_buffer_t* input,
    dcp_audio_buffer_t* output,
    dcp_int num_frames
);

/**
 * Reset oversampler state
 * @param os Oversampler handle
 */
void dcp_oversample_reset(dcp_oversample_t os);

/**
 * Get latency in frames (at base rate)
 * @param os Oversampler handle
 * @return Latency in frames
 */
dcp_int dcp_oversample_get_latency(const dcp_oversample_t os);

#ifdef __cplusplus
}
#endif

#endif // DCP_OVERSAMPLE_H
/**
 * FDN Reverb (Householder 4×4) - Apple Music Grade
 * Feedback Delay Network with unitary Householder matrix
 * Finite-sample Hann crossfade for exact freeze state lock (α = 1.0)
 */

#ifndef DCP_FDN_REVERB_H
#define DCP_FDN_REVERB_H

#include "../dcp_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// FDN Configuration
// =============================================================================

typedef struct {
    // Delay line lengths (mutually coprime for modal density)
    dcp_int delay_lengths[4];

    // Feedback matrix type
    typedef enum {
        DCP_FDN_HOUSEHOLDER = 0,  // Unitary Householder (default)
        DCP_FDN_HADAMARD = 1,     // Hadamard
        DCP_FDN_CUSTOM = 2
    } dcp_fdn_matrix_type_t;

    dcp_fdn_matrix_type_t matrix_type;

    // Damping filters (per delay line)
    dcp_float damping_lp_hz[4];   // Low-pass cutoff per line
    dcp_float damping_hp_hz[4];   // High-pass cutoff per line

    // Global feedback gain (0.0 - 0.99)
    dcp_float feedback_gain;

    // Output mixing
    dcp_float wet_mix;
    dcp_float dry_mix;

    // Freeze parameters
    dcp_bool freeze_enabled;
    dcp_int freeze_engage_frame;
    dcp_int freeze_release_frame;
    dcp_int freeze_ramp_frames;   // Hann ramp frames (default 1024)

    // Soft limiter
    dcp_bool soft_limiter;
    dcp_float limiter_threshold;  // 0.9 default
} dcp_fdn_config_t;

// =============================================================================
// FDN State
// =============================================================================

typedef struct dcp_fdn_state_s* dcp_fdn_state_t;
typedef struct dcp_fdn_s* dcp_fdn_t;

// =============================================================================
// API
// =============================================================================

/**
 * Create FDN reverb
 * @param fdn Output FDN handle
 * @param config FDN configuration
 * @param sample_rate Sample rate in Hz
 * @param max_frames Maximum frames per process block
 * @return DCP_OK on success
 */
dcp_error_t dcp_fdn_create(
    dcp_fdn_t* fdn,
    const dcp_fdn_config_t* config,
    dcp_int sample_rate,
    dcp_int max_frames
);

/**
 * Destroy FDN
 * @param fdn FDN handle
 */
void dcp_fdn_destroy(dcp_fdn_t fdn);

/**
 * Process audio block
 * @param fdn FDN handle
 * @param input Input buffer
 * @param output Output buffer
 * @param num_frames Number of frames
 * @return DCP_OK on success
 */
dcp_error_t dcp_fdn_process(
    dcp_fdn_t fdn,
    const dcp_audio_buffer_t* input,
    dcp_audio_buffer_t* output,
    dcp_int num_frames
);

/**
 * Engage freeze at specific frame (sample-accurate)
 * @param fdn FDN handle
 * @param frame Frame index to engage freeze
 * @return DCP_OK on success
 */
dcp_error_t dcp_fdn_engage_freeze(dcp_fdn_t fdn, dcp_int frame);

/**
 * Release freeze at specific frame (sample-accurate)
 * @param fdn FDN handle
 * @param frame Frame index to release freeze
 * @return DCP_OK on success
 */
dcp_error_t dcp_fdn_release_freeze(dcp_fdn_t fdn, dcp_int frame);

/**
 * Set freeze state directly
 * @param fdn FDN handle
 * @param enabled Freeze state
 * @param ramp_frames Crossfade frames (0 = instant)
 * @return DCP_OK on success
 */
dcp_error_t dcp_fdn_set_freeze(dcp_fdn_t fdn, dcp_bool enabled, dcp_int ramp_frames);

/**
 * Set feedback gain (real-time safe)
 * @param fdn FDN handle
 * @param gain Feedback gain (0.0 - 0.99)
 * @return DCP_OK on success
 */
dcp_error_t dcp_fdn_set_feedback(dcp_fdn_t fdn, dcp_float gain);

/**
 * Set damping filters (real-time safe)
 * @param fdn FDN handle
 * @param lp_hz Low-pass cutoffs for 4 lines
 * @param hp_hz High-pass cutoffs for 4 lines
 * @return DCP_OK on success
 */
dcp_error_t dcp_fdn_set_damping(
    dcp_fdn_t fdn,
    const dcp_float lp_hz[4],
    const dcp_float hp_hz[4]
);

/**
 * Reset FDN state
 * @param fdn FDN handle
 */
void dcp_fdn_reset(dcp_fdn_t fdn);

/**
 * Get current freeze alpha (0.0 - 1.0)
 * @param fdn FDN handle
 * @return Freeze alpha
 */
dcp_float dcp_fdn_get_freeze_alpha(const dcp_fdn_t fdn);

#ifdef __cplusplus
}
#endif

#endif // DCP_FDN_REVERB_H
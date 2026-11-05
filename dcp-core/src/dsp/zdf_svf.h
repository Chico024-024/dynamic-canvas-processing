/**
 * ZDF State Variable Filter - Apple Music Grade
 * Vadim Zavalishin topology-preserving transform (TPT)
 * Real-time safe, SIMD-ready, zero allocations
 */

#ifndef DCP_ZDF_SVF_H
#define DCP_ZDF_SVF_H

#include "../dcp_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Filter Types
// =============================================================================

typedef enum {
    DCP_ZDF_LPF = 0,    // Low-pass
    DCP_ZDF_HPF = 1,    // High-pass
    DCP_ZDF_BPF = 2,    // Band-pass
    DCP_ZDF_HPF4 = 3,   // 4th-order high-pass (2× HPF2 cascade)
    DCP_ZDF_LPF4 = 4,   // 4th-order low-pass
    DCP_ZDF_BPF4 = 5,   // 4th-order band-pass
    DCP_ZDF_NOTCH = 6,  // Notch
    DCP_ZDF_PEAK = 7,   // Peaking EQ
    DCP_ZDF_LOSHELF = 8, // Low shelf
    DCP_ZDF_HISHELF = 9  // High shelf
} dcp_zdf_filter_type_t;

// =============================================================================
// Filter State (Real-Time Safe, Cache-Friendly)
// =============================================================================

typedef struct {
    // Integration state (2 states per 2nd-order section)
    dcp_float s1, s2;      // State variables
    dcp_float s3, s4;      // For 4th-order cascade

    // Coefficients (updated per block)
    dcp_float g;           // Gain factor g = tan(pi * fc / sr)
    dcp_float R;           // Resonance factor
    dcp_float c1, c2, c3;  // Precomputed coefficients

    // Sweep support
    dcp_float g_start, g_end;
    dcp_float R_start, R_end;
    dcp_bool is_sweep;
} dcp_zdf_svf_state_t;

// =============================================================================
// Filter Handle
// =============================================================================

typedef struct {
    dcp_zdf_svf_state_t states[DCP_MAX_CHANNELS];
    dcp_int num_channels;
    dcp_int sample_rate;
    dcp_zdf_filter_type_t filter_type;
    dcp_float current_fc;
    dcp_float current_q;
} dcp_zdf_svf_t;

// =============================================================================
// API
// =============================================================================

/**
 * Create ZDF SVF filter
 * @param filter Output filter handle
 * @param num_channels Number of channels (1-8)
 * @param sample_rate Sample rate in Hz
 * @param filter_type Filter type
 * @return DCP_OK on success
 */
dcp_error_t dcp_zdf_svf_create(
    dcp_zdf_svf_t** filter,
    dcp_int num_channels,
    dcp_int sample_rate,
    dcp_zdf_filter_type_t filter_type
);

/**
 * Destroy filter
 * @param filter Filter to destroy
 */
void dcp_zdf_svf_destroy(dcp_zdf_svf_t* filter);

/**
 * Set filter parameters (real-time safe)
 * @param filter Filter handle
 * @param fc_hz Cutoff frequency in Hz (20 - sr/2 * 0.95)
 * @param q Resonance/Q factor (0.5 - 12.0)
 * @return DCP_OK on success
 */
dcp_error_t dcp_zdf_svf_set_params(
    dcp_zdf_svf_t* filter,
    dcp_float fc_hz,
    dcp_float q
);

/**
 * Start parameter sweep (real-time safe)
 * @param filter Filter handle
 * @param fc_start_hz Start cutoff frequency
 * @param fc_end_hz End cutoff frequency
 * @param q_start Start resonance
 * @param q_end End resonance
 * @param duration_frames Sweep duration in frames
 * @return DCP_OK on success
 */
dcp_error_t dcp_zdf_svf_start_sweep(
    dcp_zdf_svf_t* filter,
    dcp_float fc_start_hz,
    dcp_float fc_end_hz,
    dcp_float q_start,
    dcp_float q_end,
    dcp_int duration_frames
);

/**
 * Process audio block (real-time safe, zero allocations)
 * @param filter Filter handle
 * @param input Input buffer (interleaved or deinterleaved)
 * @param output Output buffer (can be same as input for in-place)
 * @param num_frames Number of frames to process
 * @param stride Channel stride (1 for interleaved, num_frames for deinterleaved)
 * @return DCP_OK on success
 */
dcp_error_t dcp_zdf_svf_process(
    dcp_zdf_svf_t* filter,
    const dcp_float* input,
    dcp_float* output,
    dcp_int num_frames,
    dcp_int stride
);

/**
 * Process deinterleaved (real-time safe)
 * @param filter Filter handle
 * @param input Array of channel pointers (input)
 * @param output Array of channel pointers (output)
 * @param num_frames Number of frames
 * @return DCP_OK on success
 */
dcp_error_t dcp_zdf_svf_process_deinterleaved(
    dcp_zdf_svf_t* filter,
    dcp_float* const* input,
    dcp_float* const* output,
    dcp_int num_frames
);

/**
 * Reset filter state
 * @param filter Filter handle
 */
void dcp_zdf_svf_reset(dcp_zdf_svf_t* filter);

/**
 * Get current frequency response at given frequency
 * @param filter Filter handle
 * @param freq_hz Frequency to evaluate
 * @param magnitude Output magnitude (linear)
 * @param phase Output phase (radians)
 */
void dcp_zdf_svf_get_response(
    const dcp_zdf_svf_t* filter,
    dcp_float freq_hz,
    dcp_float* magnitude,
    dcp_float* phase
);

#ifdef __cplusplus
}
#endif

#endif // DCP_ZDF_SVF_H
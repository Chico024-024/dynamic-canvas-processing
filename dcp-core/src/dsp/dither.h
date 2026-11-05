/**
 * Dither & Noise Shaping - Apple Music Grade
 * TPDF dither, 1st/2nd-order noise shaping
 * Real-time safe, zero allocations, psychoacoustically optimized
 */

#ifndef DCP_DITHER_H
#define DCP_DITHER_H

#include "../dcp_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Dither Types
// =============================================================================

typedef enum {
    DCP_DITHER_NONE = 0,
    DCP_DITHER_TPDF = 1,           // Triangular PDF (2 LSB)
    DCP_DITHER_RECTANGULAR = 2,    // Rectangular PDF (1 LSB)
    DCP_DITHER_GAUSSIAN = 3        // Gaussian PDF
} dcp_dither_type_t;

typedef enum {
    DCP_NS_NONE = 0,
    DCP_NS_1ST_ORDER = 1,          // 1st-order noise shaping
    DCP_NS_2ND_ORDER = 2,          // 2nd-order noise shaping
    DCP_NS_LIPPSHAW = 3            // Lipshitz/Vanderkooy 4th-order
} dcp_noise_shaping_t;

// =============================================================================
// Configuration
// =============================================================================

typedef struct {
    dcp_dither_type_t dither_type;
    dcp_noise_shaping_t noise_shaping;
    dcp_int target_bit_depth;      // 16, 24, 32
    dcp_float dither_amount;       // 1.0 = 1 LSB TPDF, -1 = auto
    dcp_float noise_shaping_gain;  // Noise shaping filter gain
    dcp_bool auto_dither;          // Auto-select based on bit depth
} dcp_dither_config_t;

// =============================================================================
// Dither Handle
// =============================================================================

typedef struct dcp_dither_s* dcp_dither_t;

// =============================================================================
// API
// =============================================================================

/**
 * Create dither processor
 * @param dither Output dither handle
 * @param config Configuration
 * @param num_channels Number of channels
 * @return DCP_OK on success
 */
dcp_error_t dcp_dither_create(
    dcp_dither_t* dither,
    const dcp_dither_config_t* config,
    dcp_int num_channels
);

/**
 * Destroy dither processor
 * @param dither Dither handle
 */
void dcp_dither_destroy(dcp_dither_t dither);

/**
 * Apply dither and noise shaping
 * @param dither Dither handle
 * @param input Input buffer (32-bit float, full scale = ±1.0)
 * @param output Output buffer (same size, dithered)
 * @param num_frames Number of frames
 * @return DCP_OK on success
 */
dcp_error_t dcp_dither_process(
    dcp_dither_t dither,
    const dcp_audio_buffer_t* input,
    dcp_audio_buffer_t* output,
    dcp_int num_frames
);

/**
 * Apply dither in-place
 * @param dither Dither handle
 * @param buffer Buffer to dither in-place
 * @param num_frames Number of frames
 * @return DCP_OK on success
 */
dcp_error_t dcp_dither_process_inplace(
    dcp_dither_t dither,
    dcp_audio_buffer_t* buffer,
    dcp_int num_frames
);

/**
 * Reset dither state
 * @param dither Dither handle
 */
void dcp_dither_reset(dcp_dither_t dither);

/**
 * Destroy dither processor
 * @param dither Dither handle
 */
void dcp_dither_destroy(dcp_dither_t dither);

#ifdef __cplusplus
}
#endif

#endif // DCP_DITHER_H
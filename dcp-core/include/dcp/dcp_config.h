/**
 * DCP Configuration - Apple Music Grade
 * Real-time safe, validated, serializable
 */

#ifndef DCP_CONFIG_H
#define DCP_CONFIG_H

#include "dcp_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Default Configuration Values
// =============================================================================

#define DCP_DEFAULT_SAMPLE_RATE 48000
#define DCP_DEFAULT_BLOCK_SIZE 256
#define DCP_DEFAULT_MAX_CHANNELS 2
#define DCP_DEFAULT_MIN_TEMPO_CONFIDENCE 0.7f
#define DCP_DEFAULT_MIN_DOWNBEATS 5
#define DCP_DEFAULT_MAX_BRIDGE_BARS 16
#define DCP_DEFAULT_MIN_BRIDGE_BARS 4
#define DCP_DEFAULT_MASTER_LUFS_TARGET -14.0f
#define DCP_DEFAULT_MASTER_TRUE_PEAK_CEILING -1.0f
#define DCP_DEFAULT_DITHER_AMOUNT -1.0f  // Auto
#define DCP_DEFAULT_MAX_THREADS 0
#define DCP_DEFAULT_MAX_LATENCY_MS 20

// =============================================================================
// Configuration Validation
// =============================================================================

/**
 * Validate configuration struct
 * @param config Configuration to validate
 * @return DCP_OK if valid, error code otherwise
 */
dcp_error_t dcp_config_validate(const dcp_config_t* config);

/**
 * Fill configuration with default values
 * @param config Configuration to initialize
 */
void dcp_config_defaults(dcp_config_t* config);

/**
 * Load configuration from JSON file
 * @param config Configuration to populate
 * @param path Path to JSON file
 * @return DCP_OK on success
 */
dcp_error_t dcp_config_load_json(dcp_config_t* config, const char* path);

/**
 * Save configuration to JSON file
 * @param config Configuration to save
 * @param path Path to JSON file
 * @return DCP_OK on success
 */
dcp_error_t dcp_config_save_json(const dcp_config_t* config, const char* path);

// =============================================================================
// Configuration Presets
// =============================================================================

typedef enum {
    DCP_PRESET_FAST = 0,        // Low quality, minimal CPU
    DCP_PRESET_BALANCED = 1,    // Default
    DCP_PRESET_HIGH_QUALITY = 2, // Best quality, more CPU
    DCP_PRESET_REALTIME = 3,    // Optimized for real-time (low latency)
    DCP_PRESET_OFFLINE = 4      // Offline render, max quality
} dcp_preset_t;

/**
 * Apply preset to configuration
 * @param config Configuration to modify
 * @param preset Preset to apply
 */
void dcp_config_apply_preset(dcp_config_t* config, dcp_preset_t preset);

// =============================================================================
// Real-Time Configuration Updates
// =============================================================================

/**
 * Update runtime-configurable parameters (safe to call from audio thread)
 * @param config Configuration to update
 * @param master_lufs_target New LUFS target (-24 to 0)
 * @param master_true_peak_ceiling New true peak ceiling (-6 to 0 dBTP)
 * @param master_saturation_enabled New saturation state
 * @param dither_amount New dither amount (-1=auto, 0=none, >0=specific)
 * @return DCP_OK on success
 */
dcp_error_t dcp_config_update_realtime(
    dcp_config_t* config,
    dcp_float master_lufs_target,
    dcp_float master_true_peak_ceiling,
    dcp_bool master_saturation_enabled,
    dcp_float dither_amount
);

#ifdef __cplusplus
}
#endif

#endif // DCP_CONFIG_H
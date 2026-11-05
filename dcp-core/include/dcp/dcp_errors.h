/**
 * DCP Error Handling - Apple Music Grade
 * Thread-safe, real-time safe error handling
 */

#ifndef DCP_ERRORS_H
#define DCP_ERRORS_H

#include "dcp_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Error String Mapping
// =============================================================================

/**
 * Get human-readable error string for error code
 * @param error Error code
 * @return Static string (do not free)
 */
const char* dcp_error_string(dcp_error_t error);

// =============================================================================
// Error Context (Thread-Local Storage)
// =============================================================================

/**
 * Get last error for current thread
 * @return Last error code, or DCP_OK if none
 */
dcp_error_t dcp_get_last_error(void);

/**
 * Set last error for current thread
 * @param error Error code to set
 */
void dcp_set_last_error(dcp_error_t error);

/**
 * Clear last error for current thread
 */
void dcp_clear_last_error(void);

// =============================================================================
// Error Callback (Optional)
// =============================================================================

typedef void (*dcp_error_callback_t)(dcp_error_t error, const char* file, int line, const char* function, void* user_data);

/**
 * Set global error callback
 * @param callback Callback function
 * @param user_data User data passed to callback
 */
void dcp_set_error_callback(dcp_error_callback_t callback, void* user_data);

// =============================================================================
// Assertion Macros (Debug Only)
// =============================================================================

#ifdef NDEBUG
    #define DCP_ASSERT(cond) ((void)0)
    #define DCP_CHECK_ERROR(err) ((void)0)
#else
    #include <assert.h>
    #define DCP_ASSERT(cond) assert(cond)
    #define DCP_CHECK_ERROR(err) do { \
        if ((err) != DCP_OK) { \
            dcp_set_last_error(err); \
            assert(false && "DCP Error"); \
        } \
    } while(0)
#endif

// =============================================================================
// Real-Time Safe Error Helpers
// =============================================================================

/**
 * Check pointer and set error if null (real-time safe: no allocation)
 * @param ptr Pointer to check
 * @param error_code Error code to set if null
 * @return true if ptr is not null
 */
static inline dcp_bool dcp_check_ptr(const void* ptr, dcp_error_t error_code) {
    if (!ptr) {
        dcp_set_last_error(error_code);
        return DCP_FALSE;
    }
    return DCP_TRUE;
}

/**
 * Check range and set error if out of bounds (real-time safe)
 * @param value Value to check
 * @param min Minimum allowed value
 * @param max Maximum allowed value
 * @param error_code Error code to set if out of bounds
 * @return true if in range
 */
static inline dcp_bool dcp_check_range(dcp_float value, dcp_float min, dcp_float max, dcp_error_t error_code) {
    if (value < min || value > max) {
        dcp_set_last_error(error_code);
        return DCP_FALSE;
    }
    return DCP_TRUE;
}

/**
 * Check int range and set error if out of bounds (real-time safe)
 */
static inline dcp_bool dcp_check_range_int(dcp_int value, dcp_int min, dcp_int max, dcp_error_t error_code) {
    if (value < min || value > max) {
        dcp_set_last_error(error_code);
        return DCP_FALSE;
    }
    return DCP_TRUE;
}

#ifdef __cplusplus
}
#endif

#endif // DCP_ERRORS_H
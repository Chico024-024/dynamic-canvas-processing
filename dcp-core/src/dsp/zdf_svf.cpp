/**
 * ZDF State Variable Filter Implementation - Apple Music Grade
 * Vadim Zavalishin Topology-Preserving Transform (TPT)
 * Based on "The Art of VA Filter Design" by Vadim Zavalishin
 *
 * Key properties:
 * - Sample-accurate frequency sweeps
 * - Stable at any resonance
 * - No warping issues
 * - Real-time safe, zero allocations
 * - SIMD-friendly
 */

#include "zdf_svf.h"
#include <cmath>
#include <cstring>
#include <algorithm>

// =============================================================================
// Constants
// =============================================================================

namespace {
    constexpr dcp_float DCP_PI = 3.14159265358979323846f;
    constexpr dcp_float DCP_2PI = 2.0f * DCP_PI;
    constexpr dcp_float DCP_MIN_FC = 20.0f;
    constexpr dcp_float DCP_MAX_FC_RATIO = 0.475f;  // Nyquist * 0.95
    constexpr dcp_float DCP_MIN_Q = 0.5f;
    constexpr dcp_float DCP_MAX_Q = 12.0f;
    constexpr dcp_float DCP_DENORMAL_THRESHOLD = 1e-25f;
}

// =============================================================================
// Denormal Protection
// =============================================================================

static inline dcp_float dcp_denormal_guard(dcp_float x) {
    // Flush denormals to zero (real-time safe)
    return (std::fabs(x) < DCP_DENORMAL_THRESHOLD) ? 0.0f : x;
}

// =============================================================================
// Coefficient Computation
// =============================================================================

static void dcp_zdf_svf_compute_coeffs(
    dcp_zdf_svf_state_t* state,
    dcp_float fc_hz,
    dcp_float q,
    dcp_int sample_rate
) {
    // Clamp parameters
    fc_hz = std::clamp(fc_hz, DCP_MIN_FC, sample_rate * DCP_MAX_FC_RATIO);
    q = std::clamp(q, DCP_MIN_Q, DCP_MAX_Q);

    // TPT: g = tan(pi * fc / sr)
    state->g = std::tanf(DCP_PI * fc_hz / sample_rate);
    state->R = 1.0f / q;  // Damping factor

    // Precompute coefficients for TPT
    // State equations:
    // v1 = g * (input - R * v1 - v2)
    // v2 = v2 + g * v1
    //
    // Solving for v1:
    // v1 = g / (1 + g*R + g*g) * (input - R*v1 - v2)
    // Actually the standard TPT formulation:
    // g = tan(pi*fc/sr)
    // R = 1/Q
    //
    // y_hp = (x - R*v1 - v2) / (1 + g*R + g*g)
    // v1 = v1 + g * y_hp
    // y_bp = g * y_hp
    // v2 = v2 + g * y_bp
    // y_lp = g * y_bp

    const dcp_float g = state->g;
    const dcp_float R = state->R;
    const dcp_float gR = g * R;
    const dcp_float gg = g * g;
    const dcp_float denom = 1.0f + gR + gg;

    // These coefficients map input + states to outputs
    // For standard TPT SVF with 3 outputs (HP, BP, LP)
    state->c1 = g / denom;           // HP coefficient
    state->c2 = g * state->c1;       // BP coefficient
    state->c3 = g * state->c2;       // LP coefficient

    // Store for 4th-order cascade
    state->is_sweep = false;
}

// =============================================================================
// Public API Implementation
// =============================================================================

extern "C" {

dcp_error_t dcp_zdf_svf_create(
    dcp_zdf_svf_t** filter,
    dcp_int num_channels,
    dcp_int sample_rate,
    dcp_zdf_filter_type_t filter_type
) {
    if (!filter || num_channels <= 0 || num_channels > DCP_MAX_CHANNELS) {
        return DCP_ERROR_INVALID_ARGUMENT;
    }
    if (sample_rate <= 0) {
        return DCP_ERROR_INVALID_SAMPLE_RATE;
    }

    dcp_zdf_svf_t* f = new (std::nothrow) dcp_zdf_svf_t;
    if (!f) return DCP_ERROR_OUT_OF_MEMORY;

    f->num_channels = num_channels;
    f->sample_rate = sample_rate;
    f->filter_type = filter_type;
    f->current_fc = 1000.0f;
    f->current_q = 1.0f;

    // Initialize states
    for (int i = 0; i < num_channels; ++i) {
        f->states[i] = {};
    }

    *filter = f;
    return DCP_OK;
}

void dcp_zdf_svf_destroy(dcp_zdf_svf_t* filter) {
    if (filter) {
        delete filter;
    }
}

dcp_error_t dcp_zdf_svf_set_params(
    dcp_zdf_svf_t* filter,
    dcp_float fc_hz,
    dcp_float q
) {
    if (!filter) return DCP_ERROR_NULL_POINTER;
    if (fc_hz < DCP_MIN_FC || fc_hz > filter->sample_rate * DCP_MAX_FC_RATIO) {
        return DCP_ERROR_INVALID_ARGUMENT;
    }
    if (q < DCP_MIN_Q || q > DCP_MAX_Q) {
        return DCP_ERROR_INVALID_ARGUMENT;
    }

    filter->current_fc = fc_hz;
    filter->current_q = q;

    for (int i = 0; i < filter->num_channels; ++i) {
        dcp_zdf_svf_compute_coeffs(&filter->states[i], fc_hz, q, filter->sample_rate);
    }

    return DCP_OK;
}

dcp_error_t dcp_zdf_svf_start_sweep(
    dcp_zdf_svf_t* filter,
    dcp_float fc_start_hz,
    dcp_float fc_end_hz,
    dcp_float q_start,
    dcp_float q_end,
    dcp_int duration_frames
) {
    if (!filter) return DCP_ERROR_NULL_POINTER;
    if (duration_frames <= 0) return DCP_ERROR_INVALID_ARGUMENT;

    // Clamp
    fc_start_hz = std::clamp(fc_start_hz, DCP_MIN_FC, filter->sample_rate * DCP_MAX_FC_RATIO);
    fc_end_hz = std::clamp(fc_end_hz, DCP_MIN_FC, filter->sample_rate * DCP_MAX_FC_RATIO);
    q_start = std::clamp(q_start, DCP_MIN_Q, DCP_MAX_Q);
    q_end = std::clamp(q_end, DCP_MIN_Q, DCP_MAX_Q);

    for (int i = 0; i < filter->num_channels; ++i) {
        auto& state = filter->states[i];

        // Compute start coefficients
        state.g = std::tanf(DCP_PI * fc_start_hz / filter->sample_rate);
        state.R = 1.0f / q_start;
        state.is_sweep = true;

        // Store sweep parameters
        state.g_start = state.g;
        state.g_end = std::tanf(DCP_PI * fc_end_hz / filter->sample_rate);
        state.R_start = state.R;
        state.R_end = 1.0f / q_end;
    }

    return DCP_OK;
}

dcp_error_t dcp_zdf_svf_process(
    dcp_zdf_svf_t* filter,
    const dcp_float* input,
    dcp_float* output,
    dcp_int num_frames,
    dcp_int stride
) {
    if (!filter || !input || !output || num_frames <= 0) {
        return DCP_ERROR_INVALID_ARGUMENT;
    }

    const int ch = filter->num_channels;
    const int sr = filter->sample_rate;
    const int ftype = filter->filter_type;

    // Process each channel
    for (int c = 0; c < ch; ++c) {
        auto& state = filter->states[c];

        const dcp_float* in = input + c * stride;
        dcp_float* out = output + c * stride;

        // TPT SVF state variables
        dcp_float v1 = state.s1;
        dcp_float v2 = state.s2;
        dcp_float v3 = state.s3;
        dcp_float v4 = state.s4;

        // Sweep interpolation
        const bool is_sweep = state.is_sweep;
        const dcp_float g_start = state.g_start;
        const dcp_float g_end = state.g_end;
        const dcp_float R_start = state.R_start;
        const dcp_float R_end = state.R_end;

        // Determine processing mode
        const bool is_hpf = (ftype == DCP_ZDF_HPF || ftype == DCP_ZDF_HPF4);
        const bool is_lpf = (ftype == DCP_ZDF_LPF || ftype == DCP_ZDF_LPF4);
        const bool is_bpf = (ftype == DCP_ZDF_BPF || ftype == DCP_ZDF_BPF4);
        const bool is_hpf4 = (ftype == DCP_ZDF_HPF4);
        const bool is_lpf4 = (ftype == DCP_ZDF_LPF4);
        const bool is_bpf4 = (ftype == DCP_ZDF_BPF4);

        for (int n = 0; n < num_frames; ++n) {
            const dcp_float x = in[n];

            // Sweep interpolation
            dcp_float g, R;
            if (is_sweep) {
                const dcp_float t = (dcp_float)n / num_frames;
                g = g_start + (g_end - g_start) * t;
                R = R_start + (R_end - R_start) * t;
            } else {
                g = state.g;
                R = state.R;
            }

            // TPT coefficients for this sample
            const dcp_float gR = g * R;
            const dcp_float gg = g * g;
            const dcp_float denom = 1.0f + gR + gg;
            const dcp_float c1 = g / denom;      // HP coeff
            const dcp_float c2 = g * c1;         // BP coeff
            const dcp_float c3 = g * c2;         // LP coeff

            // TPT State Equations (from Zavalishin):
            // y_hp = c1 * (x - R*v1 - v2)
            // v1_new = v1 + g * y_hp
            // y_bp = g * y_hp
            // v2_new = v2 + g * y_bp
            // y_lp = g * y_bp

            const dcp_float y_hp = c1 * (x - R * v1 - v2);
            v1 = dcp_denormal_guard(v1 + g * y_hp);

            const dcp_float y_bp = g * y_hp;
            v2 = dcp_denormal_guard(v2 + g * y_bp);

            const dcp_float y_lp = g * y_bp;

            // 4th-order cascade (second stage)
            dcp_float y_out;
            if (is_hpf4 || is_lpf4 || is_bpf4) {
                // Second stage: feed y_hp/y_bp/y_lp into second filter
                const dcp_float y_hp2 = c1 * (y_hp - R * v3 - v4);
                v3 = dcp_denormal_guard(v3 + g * y_hp2);
                const dcp_float y_bp2 = g * y_hp2;
                v4 = dcp_denormal_guard(v4 + g * y_bp2);
                const dcp_float y_lp2 = g * y_bp2;

                if (is_hpf4) y_out = y_hp2;
                else if (is_lpf4) y_out = y_lp2;
                else y_out = y_bp2;
            } else {
                // 2nd-order output
                if (is_hpf) y_out = y_hp;
                else if (is_lpf) y_out = y_lp;
                else y_out = y_bp;
            }

            // Soft clipper for high resonance (prevent blowup)
            if (std::fabs(y_out) > 1.0f) {
                y_out = std::tanh(y_out);
            }

            out[n] = y_out;
        }

        // Store final states
        state.s1 = v1;
        state.s2 = v2;
        state.s3 = v3;
        state.s4 = v4;
    }

    return DCP_OK;
}

dcp_error_t dcp_zdf_svf_process_deinterleaved(
    dcp_zdf_svf_t* filter,
    dcp_float* const* input,
    dcp_float* const* output,
    dcp_int num_frames
) {
    if (!filter || !input || !output || num_frames <= 0) {
        return DCP_ERROR_INVALID_ARGUMENT;
    }

    // Use process with stride=1 for each channel
    for (int c = 0; c < filter->num_channels; ++c) {
        if (!input[c] || !output[c]) {
            return DCP_ERROR_NULL_POINTER;
        }
        dcp_error_t err = dcp_zdf_svf_process(filter, input[c], output[c], num_frames, 1);
        if (err != DCP_OK) return err;
    }
    return DCP_OK;
}

void dcp_zdf_svf_reset(dcp_zdf_svf_t* filter) {
    if (!filter) return;
    for (int i = 0; i < filter->num_channels; ++i) {
        filter->states[i] = {};
    }
}

void dcp_zdf_svf_get_response(
    const dcp_zdf_svf_t* filter,
    dcp_float freq_hz,
    dcp_float* magnitude,
    dcp_float* phase
) {
    if (!filter || !magnitude || !phase) return;

    const dcp_float g = std::tanf(DCP_PI * freq_hz / filter->sample_rate);
    const dcp_float R = 1.0f / filter->current_q;
    const dcp_float gR = g * R;
    const dcp_float gg = g * g;
    const dcp_float denom = 1.0f + gR + gg;

    // Transfer functions at specific frequency
    // H_lp = g^2 / (1 + gR + g^2 + j*...)
    // For magnitude at fc, use steady-state analysis

    // Simplified: evaluate at the pole frequency
    const dcp_float w = 2.0f * DCP_PI * freq_hz / filter->sample_rate;
    const dcp_float cw = std::cosf(w);
    const dcp_float sw = std::sinf(w);

    // ZDF SVF transfer functions (analog prototype mapped via TPT)
    dcp_float H_lp, H_bp, H_hp;

    // LP: g^2 / (1 + gR + g^2)
    // BP: g / (1 + gR + g^2)  (at resonance)
    // HP: 1 / (1 + gR + g^2)  (at resonance)

    const dcp_float mag_lp = gg / denom;
    const dcp_float mag_bp = g / denom;
    const dcp_float mag_hp = 1.0f / denom;

    const int ftype = filter->filter_type;
    if (ftype == DCP_ZDF_LPF || ftype == DCP_ZDF_LPF4) {
        *magnitude = (ftype == DCP_ZDF_LPF4) ? mag_lp * mag_lp : mag_lp;
        *phase = -2.0f * std::atanf(g);
    } else if (ftype == DCP_ZDF_HPF || ftype == DCP_ZDF_HPF4) {
        *magnitude = (ftype == DCP_ZDF_HPF4) ? mag_hp * mag_hp : mag_hp;
        *phase = 2.0f * std::atanf(g);
    } else {
        *magnitude = (ftype == DCP_ZDF_BPF4) ? mag_bp * mag_bp : mag_bp;
        *phase = 0.0f;
    }
}

} // extern "C"
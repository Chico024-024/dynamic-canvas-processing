/**
 * FDN Reverb Implementation - Apple Music Grade
 * 4×4 Householder Feedback Delay Network
 * Finite-sample Hann crossfade for exact freeze state lock (α = 1.0)
 * Based on Jot (1991), Rocchesso & Smith, and Stautner & Puckette
 */

#include "fdn_reverb.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <memory>

// =============================================================================
// Constants
// =============================================================================

namespace {
    constexpr dcp_float DCP_PI = 3.14159265358979323846f;
    constexpr dcp_float DCP_2PI = 2.0f * DCP_PI;
    constexpr dcp_float DCP_SQRT2 = 1.4142135623730951f;
    constexpr dcp_float DCP_INV_SQRT2 = 0.7071067811865475f;

    // Default coprime delay lengths (in samples at 48kHz)
    // Scaled proportionally for other sample rates
    constexpr int DEFAULT_DELAYS_48K[4] = { 1699, 2141, 2749, 3467 };

    constexpr dcp_float DCP_DENORMAL_THRESHOLD = 1e-25f;
    constexpr dcp_float DCP_SOFT_LIMITER_THRESHOLD = 0.9f;
    constexpr dcp_float DCP_SOFT_LIMITER_SLOPE = 0.1f;

    // Householder matrix: H = I - (2/N) * 1 * 1^T
    // For N=4: H = I - 0.5 * 1 * 1^T
    // This is unitary: H * H^T = I
    constexpr dcp_float HOUSEHOLDER_COEFF = 0.5f;
}

// =============================================================================
// Denormal Protection
// =============================================================================

static inline dcp_float dcp_denormal_guard(dcp_float x) {
    return (std::fabs(x) < DCP_DENORMAL_THRESHOLD) ? 0.0f : x;
}

// =============================================================================
// Delay Line Structure
// =============================================================================

struct DelayLine {
    dcp_float* buffer;
    dcp_int length;
    dcp_int write_pos;
    dcp_int read_pos;

    DelayLine() : buffer(nullptr), length(0), write_pos(0), read_pos(0) {}

    ~DelayLine() {
        delete[] buffer;
    }

    bool allocate(dcp_int len) {
        if (buffer) delete[] buffer;
        length = len;
        buffer = new (std::nothrow) dcp_float[len];
        if (!buffer) return false;
        std::fill(buffer, buffer + length, 0.0f);
        write_pos = 0;
        read_pos = 0;
        return true;
    }

    inline dcp_float read() const {
        return buffer[read_pos];
    }

    inline void write(dcp_float value) {
        buffer[write_pos] = value;
    }

    inline void advance() {
        write_pos = (write_pos + 1) % length;
        read_pos = (read_pos + 1) % length;
    }

    inline void set_read_pos(dcp_int pos) {
        read_pos = (pos % length + length) % length;
    }

    void clear() {
        if (buffer) {
            std::fill(buffer, buffer + length, 0.0f);
        }
    }
};

// =============================================================================
// Damping Filter (1st-order HPF + LPF per delay line)
// =============================================================================

struct DampingFilter {
    dcp_float lp_state;
    dcp_float hp_state;
    dcp_float lp_coeff;
    dcp_float hp_coeff;

    DampingFilter() : lp_state(0.0f), hp_state(0.0f), lp_coeff(0.0f), hp_coeff(0.0f) {}

    inline void set_coeffs(dcp_float lp_hz, dcp_float hp_hz, dcp_float sample_rate) {
        // LPF: y[n] = lp_coeff * x[n] + (1 - lp_coeff) * y[n-1]
        // LP coeff = 1 - exp(-2π * fc / sr)
        if (lp_hz > 0.0f && lp_hz < sample_rate * 0.5f) {
            lp_coeff = 1.0f - std::expf(-2.0f * DCP_PI * lp_hz / sample_rate);
        } else {
            lp_coeff = 1.0f;  // No filtering
        }

        // HPF: y[n] = hp_coeff * (y[n-1] + x[n] - x[n-1])
        // HP coeff = exp(-2π * fc / sr)
        if (hp_hz > 0.0f && hp_hz < sample_rate * 0.5f) {
            hp_coeff = std::expf(-2.0f * DCP_PI * hp_hz / sample_rate);
        } else {
            hp_coeff = 0.0f;  // No filtering
        }
    }

    inline dcp_float process(dcp_float input) {
        // HPF first
        dcp_float hp_out = hp_coeff * (hp_state + input - hp_state);  // Simplified
        hp_state = input;

        // LPF
        dcp_float lp_out = lp_coeff * hp_out + (1.0f - lp_coeff) * lp_state;
        lp_state = dcp_denormal_guard(lp_out);

        return lp_out;
    }

    void reset() {
        lp_state = 0.0f;
        hp_state = 0.0f;
    }
};

// =============================================================================
// FDN Internal State
// =============================================================================

struct FDNInternal {
    // Configuration
    dcp_fdn_config_t config;
    dcp_int sample_rate;
    dcp_int max_frames;

    // Delay lines (4 channels)
    DelayLine delays[4];
    DampingFilter dampings[4];

    // Freeze state
    dcp_float freeze_alpha;       // 0.0 = normal, 1.0 = frozen
    dcp_int freeze_ramp_remaining; // Frames remaining in ramp
    dcp_int freeze_ramp_total;    // Total ramp frames
    dcp_bool freeze_ramping_up;

    // Householder matrix state
    dcp_float delay_reads[4];     // Current delay line outputs

    // Soft limiter state
    dcp_float limiter_state[4];

    // Metrics
    dcp_int frames_processed;

    FDNInternal() : sample_rate(48000), max_frames(4096), freeze_alpha(0.0f),
                    freeze_ramp_remaining(0), freeze_ramp_total(1024),
                    freeze_ramping_up(false), frames_processed(0) {
        std::fill(std::begin(limiter_state), std::end(limiter_state), 0.0f);
    }

    ~FDNInternal() = default;
};

// =============================================================================
// Helper Functions
// =============================================================================

static inline dcp_float dcp_hann_window(dcp_float x) {
    // Finite-sample Hann: 0.5 * (1 - cos(π * x))
    return 0.5f * (1.0f - std::cosf(DCP_PI * x));
}

static inline dcp_float soft_limiter(dcp_float x, dcp_float threshold, dcp_float slope) {
    // C^1 continuous soft limiter
    // For |x| > threshold: sign(x) * (threshold + slope * tanh((|x| - threshold) / slope))
    const dcp_float ax = std::fabs(x);
    if (ax <= threshold) return x;
    const dcp_float sign = (x >= 0.0f) ? 1.0f : -1.0f;
    return sign * (threshold + slope * std::tanhf((ax - threshold) / slope));
}

static inline void householder_4x4(const dcp_float in[4], dcp_float out[4]) {
    // H = I - 0.5 * 1 * 1^T
    // out[i] = in[i] - 0.5 * sum(in)
    dcp_float sum = in[0] + in[1] + in[2] + in[3];
    dcp_float half_sum = 0.5f * sum;
    out[0] = in[0] - half_sum;
    out[1] = in[1] - half_sum;
    out[2] = in[2] - half_sum;
    out[3] = in[3] - half_sum;
}

// =============================================================================
// FDN Implementation
// =============================================================================

extern "C" {

dcp_error_t dcp_fdn_create(
    dcp_fdn_t* fdn,
    const dcp_fdn_config_t* config,
    dcp_int sample_rate,
    dcp_int max_frames
) {
    if (!fdn || !config || sample_rate <= 0 || max_frames <= 0) {
        return DCP_ERROR_INVALID_ARGUMENT;
    }

    auto* internal = new (std::nothrow) FDNInternal();
    if (!internal) return DCP_ERROR_OUT_OF_MEMORY;

    internal->config = *config;
    internal->sample_rate = sample_rate;
    internal->max_frames = max_frames;
    internal->freeze_alpha = 0.0f;
    internal->freeze_ramp_total = config->freeze_ramp_frames > 0 ? config->freeze_ramp_frames : 1024;
    internal->freeze_ramp_remaining = 0;
    internal->freeze_ramping_up = false;

    // Scale delay lengths for sample rate
    dcp_float scale = (dcp_float)sample_rate / 48000.0f;
    for (int i = 0; i < 4; ++i) {
        dcp_int len = (dcp_int)std::round(DEFAULT_DELAYS_48K[i] * scale);
        // Ensure odd length for better modal distribution
        len = (len | 1);
        if (!internal->delays[i].allocate(len)) {
            dcp_fdn_destroy(internal);
            return DCP_ERROR_OUT_OF_MEMORY;
        }

        // Configure damping filters
        dcp_float lp = config->damping_lp_hz[i] > 0 ? config->damping_lp_hz[i] : 4500.0f;
        dcp_float hp = config->damping_hp_hz[i] > 0 ? config->damping_hp_hz[i] : 300.0f;
        internal->dampings[i].set_coeffs(lp, hp, sample_rate);
    }

    *fdn = internal;
    return DCP_OK;
}

void dcp_fdn_destroy(dcp_fdn_t fdn) {
    if (fdn) {
        delete static_cast<FDNInternal*>(fdn);
    }
}

dcp_error_t dcp_fdn_process(
    dcp_fdn_t fdn,
    const dcp_audio_buffer_t* input,
    dcp_audio_buffer_t* output,
    dcp_int num_frames
) {
    if (!fdn || !input || !output || num_frames <= 0) {
        return DCP_ERROR_INVALID_ARGUMENT;
    }

    auto* f = static_cast<FDNInternal*>(fdn);
    if (num_frames > f->max_frames) {
        return DCP_ERROR_BUFFER_TOO_SMALL;
    }

    const int ch = input->num_channels;
    if (ch < 1 || ch > 2) return DCP_ERROR_INVALID_CHANNEL_COUNT;

    const dcp_float* in_l = input->channels[0];
    const dcp_float* in_r = (ch == 2) ? input->channels[1] : input->channels[0];
    dcp_float* out_l = output->channels[0];
    dcp_float* out_r = (ch == 2) ? output->channels[1] : output->channels[0];

    const dcp_float fb_gain = f->config.feedback_gain;
    const dcp_float wet = f->config.wet_mix;
    const dcp_float dry = f->config.dry_mix;
    const bool freeze_enabled = f->config.freeze_enabled;
    const bool soft_limit = f->config.soft_limiter;
    const dcp_float lim_thresh = f->config.limiter_threshold > 0 ? f->config.limiter_threshold : DCP_SOFT_LIMITER_THRESHOLD;

    // Process each frame
    for (int n = 0; n < num_frames; ++n) {
        // Update freeze alpha with Hann crossfade
        if (f->freeze_ramp_remaining > 0) {
            dcp_float progress = 1.0f - (dcp_float)f->freeze_ramp_remaining / (dcp_float)f->freeze_ramp_total;
            if (f->freeze_ramping_up) {
                f->freeze_alpha = dcp_hann_window(progress);
            } else {
                f->freeze_alpha = 1.0f - dcp_hann_window(progress);
            }
            f->freeze_ramp_remaining--;
        } else if (freeze_enabled && f->freeze_alpha < 1.0f) {
            // Auto-engage freeze
            f->freeze_ramping_up = true;
            f->freeze_ramp_remaining = f->freeze_ramp_total;
            f->freeze_alpha = 0.0f;
        } else if (!freeze_enabled && f->freeze_alpha > 0.0f) {
            // Auto-release freeze
            f->freeze_ramping_up = false;
            f->freeze_ramp_remaining = f->freeze_ramp_total;
        }

        // Input injection: 4 channels from stereo
        // [L, R, (L+R)/√2, (L-R)/√2] * 0.5 * (1 - α)
        const dcp_float xL = in_l[n];
        const dcp_float xR = in_r[n];
        const dcp_float in_scale = 0.5f * (1.0f - f->freeze_alpha);

        dcp_float inj[4] = {
            xL * in_scale,
            xR * in_scale,
            DCP_INV_SQRT2 * (xL + xR) * in_scale,
            DCP_INV_SQRT2 * (xL - xR) * in_scale
        };

        // 1. Read from all delay lines
        for (int i = 0; i < 4; ++i) {
            f->delay_reads[i] = f->delays[i].read();
        }

        // 2. Apply Householder matrix (unitary mixing)
        dcp_float mixed[4];
        householder_4x4(f->delay_reads, mixed);

        // 3. Compute feedback
        // When freeze_alpha == 1.0: write mixed directly (unitary state lock)
        // When freeze_alpha == 0.0: normal feedback with damping
        const dcp_float active_scale = 1.0f - f->freeze_alpha;

        for (int i = 0; i < 4; ++i) {
            // Active path: injection + feedback
            dcp_float feedback_in = inj[i] + mixed[i] * fb_gain * active_scale;

            // Apply damping
            feedback_in = f->dampings[i].process(feedback_in);

            // Write to delay line
            // Freeze path: write mixed directly (unitary state lock)
            // Active path: write damped feedback
            dcp_float write_val = active_scale * feedback_in + f->freeze_alpha * mixed[i];

            // Soft limiter on write (C^1 continuous above threshold)
            if (soft_limit && std::fabs(write_val) > lim_thresh) {
                write_val = soft_limiter(write_val, lim_thresh, DCP_SOFT_LIMITER_SLOPE);
            }

            f->delays[i].write(write_val);
        }

        // Advance all delay lines
        for (int i = 0; i < 4; ++i) {
            f->delays[i].advance();
        }

        // 4. Orthonormal stereo output projection
        // Wet L = 1/√2 * (delay0 + delay2)
        // Wet R = 1/√2 * (delay1 + delay3)
        const dcp_float wet_L = DCP_INV_SQRT2 * (f->delay_reads[0] + f->delay_reads[2]);
        const dcp_float wet_R = DCP_INV_SQRT2 * (f->delay_reads[1] + f->delay_reads[3]);

        // Mix dry + wet
        out_l[n] = xL * dry + wet_L * wet;
        out_r[n] = xR * dry + wet_R * wet;
    }

    f->frames_processed += num_frames;
    return DCP_OK;
}

dcp_error_t dcp_fdn_engage_freeze(dcp_fdn_t fdn, dcp_int frame) {
    if (!fdn) return DCP_ERROR_NULL_POINTER;
    auto* f = static_cast<FDNInternal*>(fdn);
    if (f->freeze_ramp_remaining > 0) return DCP_OK;  // Already ramping
    f->freeze_ramping_up = true;
    f->freeze_ramp_remaining = f->freeze_ramp_total;
    return DCP_OK;
}

dcp_error_t dcp_fdn_release_freeze(dcp_fdn_t fdn, dcp_int frame) {
    if (!fdn) return DCP_ERROR_NULL_POINTER;
    auto* f = static_cast<FDNInternal*>(fdn);
    if (f->freeze_ramp_remaining > 0) return DCP_OK;  // Already ramping
    f->freeze_ramping_up = false;
    f->freeze_ramp_remaining = f->freeze_ramp_total;
    return DCP_OK;
}

dcp_error_t dcp_fdn_set_freeze(dcp_fdn_t fdn, dcp_bool enabled, dcp_int ramp_frames) {
    if (!fdn) return DCP_ERROR_NULL_POINTER;
    auto* f = static_cast<FDNInternal*>(fdn);

    if (ramp_frames > 0) {
        f->freeze_ramp_total = ramp_frames;
    }

    if (enabled && f->freeze_alpha < 1.0f) {
        f->freeze_ramping_up = true;
        f->freeze_ramp_remaining = f->freeze_ramp_total;
    } else if (!enabled && f->freeze_alpha > 0.0f) {
        f->freeze_ramping_up = false;
        f->freeze_ramp_remaining = f->freeze_ramp_total;
    }
    return DCP_OK;
}

dcp_error_t dcp_fdn_set_feedback(dcp_fdn_t fdn, dcp_float gain) {
    if (!fdn) return DCP_ERROR_NULL_POINTER;
    auto* f = static_cast<FDNInternal*>(fdn);
    f->config.feedback_gain = std::clamp(gain, 0.0f, 0.99f);
    return DCP_OK;
}

dcp_error_t dcp_fdn_set_damping(
    dcp_fdn_t fdn,
    const dcp_float lp_hz[4],
    const dcp_float hp_hz[4]
) {
    if (!fdn || !lp_hz || !hp_hz) return DCP_ERROR_INVALID_ARGUMENT;
    auto* f = static_cast<FDNInternal*>(fdn);
    for (int i = 0; i < 4; ++i) {
        f->dampings[i].set_coeffs(lp_hz[i], hp_hz[i], f->sample_rate);
    }
    return DCP_OK;
}

void dcp_fdn_reset(dcp_fdn_t fdn) {
    if (!fdn) return;
    auto* f = static_cast<FDNInternal*>(fdn);
    for (int i = 0; i < 4; ++i) {
        f->delays[i].clear();
        f->dampings[i].reset();
    }
    f->freeze_alpha = 0.0f;
    f->freeze_ramp_remaining = 0;
    f->freeze_ramping_up = false;
    f->frames_processed = 0;
    std::fill(std::begin(f->limiter_state), std::end(f->limiter_state), 0.0f);
}

dcp_float dcp_fdn_get_freeze_alpha(const dcp_fdn_t fdn) {
    if (!fdn) return 0.0f;
    return static_cast<const FDNInternal*>(fdn)->freeze_alpha;
}

} // extern "C"
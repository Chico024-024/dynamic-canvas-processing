// src/dcp/cpp/dcp_dsp_core.hpp
#pragma once

#include <array>
#include <vector>
#include <cmath>
#include <numbers>
#include <numeric>
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace dcp::dsp {

namespace detail {
    constexpr size_t gcd(size_t a, size_t b) noexcept {
        while (b != 0) {
            size_t t = b;
            b = a % b;
            a = t;
        }
        return a;
    }

    constexpr size_t nextPowerOfTwo(size_t v) noexcept {
        size_t p = 64;
        while (p < v) p <<= 1;
        return p;
    }
} // namespace detail

class FiniteStateFreezeFdn {
public:
    static constexpr size_t kNumChannels = 4;

    void prepare(double sampleRate, size_t freezeRampSamples = 1024) noexcept {
        sampleRate_ = std::max(8000.0, sampleRate);
        rampLength_ = std::max<size_t>(64, freezeRampSamples);
        const double scale = sampleRate_ / 48000.0;

        // Ensure mutually coprime delay lengths AFTER sample-rate scaling
        constexpr std::array<size_t, kNumChannels> kBaseDelays48k = {1699, 2141, 2749, 3467};
        size_t maxDelay = 512;

        for (size_t i = 0; i < kNumChannels; ++i) {
            size_t candidate = std::max<size_t>(128, static_cast<size_t>(kBaseDelays48k[i] * scale) | 1ULL);
            bool coprime = false;
            while (!coprime) {
                coprime = true;
                for (size_t j = 0; j < i; ++j) {
                    if (detail::gcd(candidate, delayLengths_[j]) != 1) {
                        coprime = false;
                        candidate += 2;
                        break;
                    }
                }
            }
            delayLengths_[i] = candidate;
            maxDelay = std::max(maxDelay, candidate);
        }

        const size_t pow2Size = detail::nextPowerOfTwo(maxDelay + 16);
        bufferMask_ = pow2Size - 1;
        for (auto& line : delayLines_) {
            line.assign(pow2Size, 0.0f);
        }

        const double w0 = std::tan(std::numbers::pi * 15.0 / sampleRate_);
        dcPole_ = static_cast<float>((1.0 - w0) / (1.0 + w0));
        dcGain_ = static_cast<float>(1.0 / (1.0 + w0));
        lpCoeff_ = static_cast<float>(std::exp(-2.0 * std::numbers::pi * 4500.0 / sampleRate_));

        reset();
    }

    void reset() noexcept {
        for (auto& line : delayLines_) {
            std::fill(line.begin(), line.end(), 0.0f);
        }
        lpState_.fill(0.0f);
        dcInState_.fill(0.0f);
        dcOutState_.fill(0.0f);
        writePos_ = 0;
        rampCounter_ = 0;
        freezeEngaged_ = false;
    }

    void setFreeze(bool freeze) noexcept {
        freezeEngaged_ = freeze;
    }

    void setParameters(float feedback, float hfDamp, float wet, float dry) noexcept {
        feedback_ = std::clamp(feedback, 0.0f, 0.96f);
        hfDamp_ = std::clamp(hfDamp, 0.0f, 0.95f);
        wet_ = std::clamp(wet, 0.0f, 1.0f);
        dry_ = std::clamp(dry, 0.0f, 1.0f);
    }

    // Exact L2 energy across the complete internal delay state vector M_0 + M_1 + M_2 + M_3
    double computeTotalStateEnergy() const noexcept {
        const size_t bufSize = delayLines_[0].size();
        double totalEnergy = 0.0;
        for (size_t i = 0; i < kNumChannels; ++i) {
            for (size_t d = 1; d <= delayLengths_[i]; ++d) {
                const size_t rIdx = (writePos_ + bufSize - d) & bufferMask_;
                const double val = static_cast<double>(delayLines_[i][rIdx]);
                totalEnergy += val * val;
            }
        }
        return totalEnergy;
    }

    void processBlock(const float* const* in, float* const* out, size_t numSamples) noexcept {
        const float* __restrict inL = in[0];
        const float* __restrict inR = in[1];
        float* __restrict outL = out[0];
        float* __restrict outR = out[1];
        const size_t bufSize = delayLines_[0].size();

        for (size_t n = 0; n < numSamples; ++n) {
            // Exact finite-sample raised-cosine (Hann) state transition
            if (freezeEngaged_ && rampCounter_ < rampLength_) {
                ++rampCounter_;
            } else if (!freezeEngaged_ && rampCounter_ > 0) {
                --rampCounter_;
            }

            std::array<float, kNumChannels> tankRead{};
            for (size_t i = 0; i < kNumChannels; ++i) {
                const size_t rIdx = (writePos_ + bufSize - delayLengths_[i]) & bufferMask_;
                tankRead[i] = delayLines_[i][rIdx];
            }

            // Orthogonal 4x4 Householder Matrix: ||mixed||_2 == ||tankRead||_2
            const float halfSum = 0.5f * (tankRead[0] + tankRead[1] + tankRead[2] + tankRead[3]);
            const std::array<float, kNumChannels> mixed = {
                tankRead[0] - halfSum,
                tankRead[1] - halfSum,
                tankRead[2] - halfSum,
                tankRead[3] - halfSum
            };

            if (rampCounter_ == rampLength_) {
                // HARD BIT-EXACT STATE DISCONNECT: Zero input, zero filter damping, exact unitary permutation
                for (size_t i = 0; i < kNumChannels; ++i) {
                    delayLines_[i][writePos_] = mixed[i];
                }
            } else {
                const float phase = static_cast<float>(rampCounter_) / static_cast<float>(rampLength_);
                const float alpha = 0.5f * (1.0f - std::cos(std::numbers::pi_v<float> * phase));
                const float inScale = 0.5f * (1.0f - alpha);

                const float xL = inL[n];
                const float xR = inR[n];
                const std::array<float, kNumChannels> inj = {
                    xL * inScale,
                    xR * inScale,
                    (xL + xR) * 0.70710678f * inScale,
                    (xL - xR) * 0.70710678f * inScale
                };

                for (size_t i = 0; i < kNumChannels; ++i) {
                    const float active = inj[i] + mixed[i] * feedback_;
                    lpState_[i] = (1.0f - lpCoeff_) * active + lpCoeff_ * lpState_[i];
                    const float damped = active - hfDamp_ * (active - lpState_[i]);

                    const float dcBlocked = dcGain_ * (damped - dcInState_[i]) + dcPole_ * dcOutState_[i];
                    dcInState_[i] = damped;
                    dcOutState_[i] = dcBlocked;

                    delayLines_[i][writePos_] = (1.0f - alpha) * dcBlocked + alpha * mixed[i];
                }
            }

            writePos_ = (writePos_ + 1) & bufferMask_;

            const float wetL = 0.70710678f * (tankRead[0] + tankRead[2]);
            const float wetR = 0.70710678f * (tankRead[1] + tankRead[3]);
            outL[n] = inL[n] * dry_ + wetL * wet_;
            outR[n] = inR[n] * dry_ + wetR * wet_;
        }
    }

private:
    double sampleRate_ = 48000.0;
    size_t bufferMask_ = 65535;
    size_t writePos_ = 0;
    size_t rampLength_ = 1024;
    size_t rampCounter_ = 0;
    bool freezeEngaged_ = false;

    std::array<std::vector<float>, kNumChannels> delayLines_;
    std::array<size_t, kNumChannels> delayLengths_{1699, 2141, 2749, 3467};
    std::array<float, kNumChannels> lpState_{};
    std::array<float, kNumChannels> dcInState_{};
    std::array<float, kNumChannels> dcOutState_{};

    float dcPole_ = 0.998f, dcGain_ = 0.999f, lpCoeff_ = 0.55f;
    float feedback_ = 0.88f, hfDamp_ = 0.35f, wet_ = 0.45f, dry_ = 1.0f;
};

} // namespace dcp::dsp


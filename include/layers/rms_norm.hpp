#pragma once

#include "layers/base.hpp"

#include <cstdint>

namespace lite_llm {

class RMSNorm : public WeightedUnaryLayer {
public:
    RMSNorm() = default;

    explicit RMSNorm(float eps);
    RMSNorm(Tensor weight, float eps);

    RMSNorm(const RMSNorm&) = delete;
    RMSNorm& operator=(const RMSNorm&) = delete;

    RMSNorm(RMSNorm&&) noexcept = default;
    RMSNorm& operator=(RMSNorm&&) noexcept = default;

    const char* name() const override {
        return "RMSNorm";
    }

    void load_weight(Tensor weight);

    void forward(const Tensor& input, Tensor& output) const override;

    float eps() const {
        return eps_;
    }

    int64_t hidden_size() const {
        return hidden_size_;
    }

private:
    float eps_ = 1e-6f;
    int64_t hidden_size_ = 0;
};

} // namespace lite_llm
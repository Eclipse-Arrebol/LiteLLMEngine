#pragma once

#include "layers/base.hpp"

#include <cstdint>

namespace lite_llm {

class Linear : public WeightedUnaryLayer {
public:
    Linear() = default;

    explicit Linear(Tensor weight);
    Linear(Tensor weight, Tensor bias);

    Linear(const Linear&) = delete;
    Linear& operator=(const Linear&) = delete;

    Linear(Linear&&) noexcept = default;
    Linear& operator=(Linear&&) noexcept = default;

    const char* name() const override {
        return "Linear";
    }

    void forward(const Tensor& input, Tensor& output) const override;

    bool has_bias() const {
        return has_bias_;
    }

    int64_t in_features() const {
        return in_features_;
    }

    int64_t out_features() const {
        return out_features_;
    }

private:
    Tensor bias_;

    bool has_bias_ = false;

    int64_t in_features_ = 0;
    int64_t out_features_ = 0;
};

} // namespace lite_llm
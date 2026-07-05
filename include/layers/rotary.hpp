#pragma once

#include "layers/base.hpp"
#include "core/tensor.hpp"

#include <cstdint>

namespace lite_llm {

class RotaryEmbedding : public Layer {
public:
    RotaryEmbedding() = default;

    RotaryEmbedding(int64_t head_dim, float rope_theta);

    RotaryEmbedding(const RotaryEmbedding&) = delete;
    RotaryEmbedding& operator=(const RotaryEmbedding&) = delete;

    RotaryEmbedding(RotaryEmbedding&&) noexcept = default;
    RotaryEmbedding& operator=(RotaryEmbedding&&) noexcept = default;

    const char* name() const override {
        return "RotaryEmbedding";
    }

    bool initialized() const override {
        return head_dim_ > 0 && rope_theta_ > 0.0f;
    }

    void apply(
        const Tensor& q,
        const Tensor& k,
        const Tensor& position_ids,
        Tensor& q_out,
        Tensor& k_out
    ) const;

    int64_t head_dim() const {
        return head_dim_;
    }

    float rope_theta() const {
        return rope_theta_;
    }

private:
    int64_t head_dim_ = 0;
    float rope_theta_ = 10000.0f;
};

} // namespace lite_llm
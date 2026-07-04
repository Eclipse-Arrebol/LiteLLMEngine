#include "layers/rotary.hpp"

#include <stdexcept>

namespace lite_llm {

RotaryEmbedding::RotaryEmbedding(int64_t head_dim, float rope_theta)
    : head_dim_(head_dim),
      rope_theta_(rope_theta) {
    if (head_dim_ <= 0) {
        throw std::runtime_error("RotaryEmbedding head_dim must be positive");
    }

    if (rope_theta_ <= 0.0f) {
        throw std::runtime_error("RotaryEmbedding rope_theta must be positive");
    }
}

void RotaryEmbedding::apply(
    const Tensor& q,
    const Tensor& k,
    const Tensor& position_ids,
    Tensor& q_out,
    Tensor& k_out
) const {
    if (!initialized()) {
        throw std::runtime_error("RotaryEmbedding::apply called before initialized");
    }

    (void)q;
    (void)k;
    (void)position_ids;
    (void)q_out;
    (void)k_out;

    throw std::runtime_error("RotaryEmbedding::apply not implemented yet");
}

} // namespace lite_llm
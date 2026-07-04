#include "layers/embedding.hpp"

#include <stdexcept>
#include <utility>

namespace lite_llm {

Embedding::Embedding(Tensor weight)
    : WeightedUnaryLayer(std::move(weight)) {
    if (weight_.shape().size() != 2) {
        throw std::runtime_error("Embedding weight must be 2D");
    }

    if (weight_.dtype() != DType::FP32) {
        throw std::runtime_error("Embedding weight must be FP32");
    }

    vocab_size_ = weight_.shape()[0];
    hidden_size_ = weight_.shape()[1];

    if (vocab_size_ <= 0 || hidden_size_ <= 0) {
        throw std::runtime_error("Embedding got invalid weight shape");
    }
}

void Embedding::forward(const Tensor& input_ids, Tensor& output) const {
    if (!initialized()) {
        throw std::runtime_error("Embedding::forward called before weight is initialized");
    }

    (void)input_ids;
    (void)output;

    throw std::runtime_error("Embedding::forward not implemented yet");
}

} // namespace lite_llm
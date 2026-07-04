#include "layers/rms_norm.hpp"

#include <stdexcept>
#include <utility>

namespace lite_llm {

RMSNorm::RMSNorm(float eps)
    : eps_(eps) {
    if (eps_ <= 0.0f) {
        throw std::runtime_error("RMSNorm eps must be positive");
    }
}

RMSNorm::RMSNorm(Tensor weight, float eps)
    : WeightedUnaryLayer(std::move(weight)),
      eps_(eps) {
    if (eps_ <= 0.0f) {
        throw std::runtime_error("RMSNorm eps must be positive");
    }

    if (weight_.shape().size() != 1) {
        throw std::runtime_error("RMSNorm weight must be 1D");
    }

    if (weight_.dtype() != DType::FP32) {
        throw std::runtime_error("RMSNorm weight must be FP32");
    }

    hidden_size_ = weight_.shape()[0];

    if (hidden_size_ <= 0) {
        throw std::runtime_error("RMSNorm got invalid hidden_size");
    }
}

void RMSNorm::load_weight(Tensor weight) {
    if (weight.shape().size() != 1) {
        throw std::runtime_error("RMSNorm weight must be 1D");
    }

    if (weight.dtype() != DType::FP32) {
        throw std::runtime_error("RMSNorm weight must be FP32");
    }

    int64_t hidden_size = weight.shape()[0];

    if (hidden_size <= 0) {
        throw std::runtime_error("RMSNorm got invalid hidden_size");
    }

    weight_ = std::move(weight);
    hidden_size_ = hidden_size;
}

void RMSNorm::forward(const Tensor& input, Tensor& output) const {
    if (!initialized()) {
        throw std::runtime_error("RMSNorm::forward called before weight is initialized");
    }

    (void)input;
    (void)output;

    throw std::runtime_error("RMSNorm::forward not implemented yet");
}

} // namespace lite_llm
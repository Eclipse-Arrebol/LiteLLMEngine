#include "layers/linear.hpp"

#include <stdexcept>
#include <utility>

namespace lite_llm {

Linear::Linear(Tensor weight)
    : WeightedUnaryLayer(std::move(weight)) {
    if (weight_.shape().size() != 2) {
        throw std::runtime_error("Linear weight must be 2D");
    }

    if (weight_.dtype() != DType::FP32) {
        throw std::runtime_error("Linear weight must be FP32");
    }

    out_features_ = weight_.shape()[0];
    in_features_ = weight_.shape()[1];

    if (out_features_ <= 0 || in_features_ <= 0) {
        throw std::runtime_error("Linear got invalid weight shape");
    }
}

Linear::Linear(Tensor weight, Tensor bias)
    : Linear(std::move(weight)) {
    if (bias.shape().size() != 1) {
        throw std::runtime_error("Linear bias must be 1D");
    }

    if (bias.dtype() != DType::FP32) {
        throw std::runtime_error("Linear bias must be FP32");
    }

    if (bias.device() != weight_.device()) {
        throw std::runtime_error("Linear bias and weight must be on same device");
    }

    if (bias.shape()[0] != out_features_) {
        throw std::runtime_error("Linear bias shape mismatch");
    }

    bias_ = std::move(bias);
    has_bias_ = true;
}


void Linear::load_weight(Tensor weight) {
    if (weight.shape().size() != 2) {
        throw std::runtime_error("Linear weight must be 2D");
    }

    if (weight.dtype() != DType::FP32) {
        throw std::runtime_error("Linear weight must be FP32");
    }

    int64_t out_features = weight.shape()[0];
    int64_t in_features = weight.shape()[1];

    if (out_features <= 0 || in_features <= 0) {
        throw std::runtime_error("Linear got invalid weight shape");
    }

    weight_ = std::move(weight);
    out_features_ = out_features;
    in_features_ = in_features;

    bias_ = Tensor();
    has_bias_ = false;
}

void Linear::load_bias(Tensor bias) {
    if (!initialized()) {
        throw std::runtime_error("Linear::load_bias called before weight is initialized");
    }

    if (bias.shape().size() != 1) {
        throw std::runtime_error("Linear bias must be 1D");
    }

    if (bias.dtype() != DType::FP32) {
        throw std::runtime_error("Linear bias must be FP32");
    }

    if (bias.device() != weight_.device()) {
        throw std::runtime_error("Linear bias and weight must be on same device");
    }

    if (bias.shape()[0] != out_features_) {
        throw std::runtime_error("Linear bias shape mismatch");
    }

    bias_ = std::move(bias);
    has_bias_ = true;
}


void Linear::forward(const Tensor& input, Tensor& output) const {
    if (!initialized()) {
        throw std::runtime_error("Linear::forward called before weight is initialized");
    }

    (void)input;
    (void)output;

    throw std::runtime_error("Linear::forward not implemented yet");
}

} // namespace lite_llm
#pragma once

#include "layers/base.hpp"

#include <cstdint>

namespace lite_llm {

class Embedding : public WeightedUnaryLayer {
public:
    Embedding() = default;

    explicit Embedding(Tensor weight);

    Embedding(const Embedding&) = delete;
    Embedding& operator=(const Embedding&) = delete;

    Embedding(Embedding&&) noexcept = default;
    Embedding& operator=(Embedding&&) noexcept = default;

    const char* name() const override {
        return "Embedding";
    }

    void forward(const Tensor& input_ids, Tensor& output) const override;

    int64_t vocab_size() const {
        return vocab_size_;
    }

    int64_t hidden_size() const {
        return hidden_size_;
    }

private:
    int64_t vocab_size_ = 0;
    int64_t hidden_size_ = 0;
};

} // namespace lite_llm
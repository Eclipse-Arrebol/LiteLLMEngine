#pragma once

#include "core/tensor.hpp"

#include <cstdint>
#include <vector>

namespace lite_llm {

int32_t argmax_last_token(const Tensor& logits);

std::vector<int32_t> argmax_each_row(const Tensor& logits);

}  // namespace lite_llm

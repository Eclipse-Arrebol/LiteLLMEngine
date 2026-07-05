#pragma once

#include "core/tensor.hpp"

#include <cstdint>

namespace lite_llm {

int32_t argmax_last_token(const Tensor& logits);

}  // namespace lite_llm
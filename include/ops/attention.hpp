#pragma once

#include "core/tensor.hpp"

namespace lite_llm {

void flash_attention(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    Tensor& output
);

}  // namespace lite_llm
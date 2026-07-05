#pragma once

#include "core/tensor.hpp"

namespace lite_llm {

void silu_and_mul(
    const Tensor& gate,
    const Tensor& up,
    Tensor& output
);

}  // namespace lite_llm
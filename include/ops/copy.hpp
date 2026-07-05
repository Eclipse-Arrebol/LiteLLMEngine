#pragma once

#include "core/tensor.hpp"

namespace lite_llm {

void tensor_copy(
    const Tensor& input,
    Tensor& output
);

}  // namespace lite_llm
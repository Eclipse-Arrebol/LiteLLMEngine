#pragma once

#include "core/tensor.hpp"

namespace lite_llm {

void tensor_add(const Tensor& a, const Tensor& b, Tensor& out);

} // namespace lite_llm
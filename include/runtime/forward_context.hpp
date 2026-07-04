#pragma once

#include "core/tensor.hpp"

#include <cstdint>

namespace lite_llm {

struct ForwardContext {
    const Tensor* position_ids = nullptr;

    int64_t seq_len = 0;
    int64_t past_len = 0;

    bool use_cache = false;
};

} // namespace lite_llm
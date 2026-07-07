#pragma once

#include "core/tensor.hpp"

#include <cstdint>

namespace lite_llm {
class ModelKVCache;
struct ForwardContext {
    const Tensor* position_ids = nullptr;

    int64_t seq_len = 0;
    int64_t past_len = 0;

    bool use_cache = false;

    ModelKVCache* kv_cache = nullptr;
    int64_t layer_idx = -1;
};

} // namespace lite_llm
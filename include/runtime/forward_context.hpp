// include/model/forward_context.hpp

#pragma once

#include "core/tensor.hpp"

#include <cstdint>

namespace lite_llm {

class ModelKVCache;
class ModelPagedKVCache;
class BlockTableManager;

struct ForwardContext {
    const Tensor* position_ids = nullptr;

    int64_t seq_len = 0;
    int64_t past_len = 0;

    bool use_cache = false;

    ModelKVCache* kv_cache = nullptr;
    int64_t layer_idx = -1;

    bool use_paged_kv_cache = false;
    ModelPagedKVCache* paged_kv_cache = nullptr;
    BlockTableManager* block_table_manager = nullptr;
    int64_t table_idx = -1;
    int64_t kv_seq_len = 0;
};

}  // namespace lite_llm
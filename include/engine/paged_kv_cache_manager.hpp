// include/engine/paged_kv_cache_manager.hpp
#pragma once

#include "core/device.hpp"
#include "core/dtype.hpp"
#include "core/tensor.hpp"
#include "engine/block_table_manager.hpp"
#include "engine/paged_kv_cache.hpp"
#include "engine/request.hpp"
#include "model/model_config.hpp"
#include "runtime/forward_context.hpp"

#include <cstdint>

namespace lite_llm {

class PagedKVCacheManager {
public:
    PagedKVCacheManager(
        const ModelConfig& config,
        Device device,
        DType dtype,
        int64_t capacity,
        int64_t page_size
    );

    int64_t capacity() const;
    int64_t page_size() const;
    int64_t num_blocks() const;

    ModelPagedKVCache& paged_kv_cache();
    const ModelPagedKVCache& paged_kv_cache() const;

    BlockTableManager& block_table_manager();
    const BlockTableManager& block_table_manager() const;

    void ensure_blocks(
        GenerationRequest& request,
        int64_t target_len
    );

    void ensure_device_blocks(
        GenerationRequest& request
    );

    void release(
        GenerationRequest& request
    );

    void fill_forward_context(
        ForwardContext& context,
        GenerationRequest& request,
        const Tensor* position_ids,
        int64_t seq_len
    );

private:
    ModelPagedKVCache paged_kv_cache_;
    BlockTableManager block_table_manager_;
    int64_t capacity_ = 0;
};

}  // namespace lite_llm
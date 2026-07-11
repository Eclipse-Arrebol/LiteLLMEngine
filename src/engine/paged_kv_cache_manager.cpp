// src/engine/paged_kv_cache_manager.cpp

#include "engine/paged_kv_cache_manager.hpp"

#include <stdexcept>

namespace lite_llm {

PagedKVCacheManager::PagedKVCacheManager(
    const ModelConfig& config,
    Device device,
    DType dtype,
    int64_t capacity,
    int64_t page_size
)
    : paged_kv_cache_(
          config.num_hidden_layers,
          capacity,
          page_size,
          config.num_key_value_heads,
          config.head_dim,
          dtype,
          device
      ),
      block_table_manager_(paged_kv_cache_.num_blocks()),
      capacity_(capacity) {
    if (config.num_hidden_layers <= 0) {
        throw std::runtime_error("PagedKVCacheManager num_hidden_layers must be positive");
    }

    if (config.num_key_value_heads <= 0) {
        throw std::runtime_error("PagedKVCacheManager num_key_value_heads must be positive");
    }

    if (config.head_dim <= 0) {
        throw std::runtime_error("PagedKVCacheManager head_dim must be positive");
    }

    if (capacity <= 0) {
        throw std::runtime_error("PagedKVCacheManager capacity must be positive");
    }

    if (page_size <= 0) {
        throw std::runtime_error("PagedKVCacheManager page_size must be positive");
    }
}

int64_t PagedKVCacheManager::capacity() const {
    return capacity_;
}

int64_t PagedKVCacheManager::page_size() const {
    return paged_kv_cache_.page_size();
}

int64_t PagedKVCacheManager::num_blocks() const {
    return paged_kv_cache_.num_blocks();
}

ModelPagedKVCache& PagedKVCacheManager::paged_kv_cache() {
    return paged_kv_cache_;
}

const ModelPagedKVCache& PagedKVCacheManager::paged_kv_cache() const {
    return paged_kv_cache_;
}

BlockTableManager& PagedKVCacheManager::block_table_manager() {
    return block_table_manager_;
}

const BlockTableManager& PagedKVCacheManager::block_table_manager() const {
    return block_table_manager_;
}

void PagedKVCacheManager::ensure_blocks(
    GenerationRequest& request,
    int64_t target_len
) {
    if (target_len < 0) {
        throw std::runtime_error("PagedKVCacheManager target_len must be non-negative");
    }

    if (target_len > capacity_) {
        throw std::runtime_error("PagedKVCacheManager target_len exceeds capacity");
    }

    request.ensure_blocks(
        block_table_manager_,
        target_len,
        paged_kv_cache_.page_size()
    );
}

void PagedKVCacheManager::ensure_device_blocks(
    GenerationRequest& request
) {
    ensure_blocks(
        request,
        request.device_len()
    );
}

void PagedKVCacheManager::release(
    GenerationRequest& request
) {
    request.release_blocks(block_table_manager_);
}

void PagedKVCacheManager::fill_forward_context(
    ForwardContext& context,
    GenerationRequest& request,
    const Tensor* position_ids,
    int64_t seq_len
) {
    if (position_ids == nullptr) {
        throw std::runtime_error("PagedKVCacheManager position_ids is null");
    }

    if (seq_len <= 0) {
        throw std::runtime_error("PagedKVCacheManager seq_len must be positive");
    }

    if (request.table_idx < 0) {
        throw std::runtime_error("PagedKVCacheManager request has no table_idx");
    }

    context.position_ids = position_ids;
    context.seq_len = seq_len;
    context.past_len = request.cached_len;

    context.use_cache = true;
    context.kv_cache = nullptr;
    context.layer_idx = -1;

    context.use_paged_kv_cache = true;
    context.paged_kv_cache = &paged_kv_cache_;
    context.block_table_manager = &block_table_manager_;
    context.table_idx = request.table_idx;
}

}  // namespace lite_llm
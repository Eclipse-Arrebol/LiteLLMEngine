// include/ops/attention.hpp

#pragma once

#include "core/tensor.hpp"
#include "engine/block_table_manager.hpp"
#include "engine/paged_kv_cache.hpp"

#include <cstdint>
#include <vector>

namespace lite_llm {

void flash_attention(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    Tensor& output
);

void flash_attention_kv_cache(
    const Tensor& q,
    const Tensor& key_cache,
    const Tensor& value_cache,
    int64_t kv_seq_len,
    Tensor& output
);

void flash_attention_paged_kv_cache_cuda(
    const Tensor& q,
    const ModelPagedKVCache& paged_kv_cache,
    const BlockTableManager& table_manager,
    int64_t table_idx,
    int64_t layer_idx,
    int64_t kv_seq_len,
    Tensor& output
);

void flash_attention_paged_kv_cache_batch_cuda(
    const Tensor& q,
    const ModelPagedKVCache& paged_kv_cache,
    const BlockTableManager& table_manager,
    const std::vector<int64_t>& table_indices,
    int64_t layer_idx,
    const std::vector<int64_t>& kv_seq_lens,
    Tensor& output
);

}  // namespace lite_llm

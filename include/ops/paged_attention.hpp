// include/ops/paged_attention.hpp
#pragma once

#include "core/tensor.hpp"
#include "engine/block_table_manager.hpp"
#include "engine/paged_kv_cache.hpp"

#include <cstdint>
#include <vector>

namespace lite_llm {

void paged_attention_decode_cpu(
    const Tensor& query,
    const ModelPagedKVCache& paged_kv_cache,
    const BlockTableManager& table_manager,
    int64_t table_idx,
    int64_t layer_idx,
    int64_t kv_seq_len,
    Tensor& output
);

void paged_attention_decode_batch_gather(
    const Tensor& query,
    const ModelPagedKVCache& paged_kv_cache,
    const BlockTableManager& table_manager,
    const std::vector<int64_t>& table_indices,
    int64_t layer_idx,
    const std::vector<int64_t>& kv_seq_lens,
    Tensor& output
);

}  // namespace lite_llm

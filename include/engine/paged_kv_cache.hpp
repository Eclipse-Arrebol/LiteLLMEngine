// include/engine/paged_kv_cache.hpp
#pragma once

#include "core/device.hpp"
#include "core/dtype.hpp"
#include "core/tensor.hpp"

#include <cstdint>
#include <vector>

namespace lite_llm {

struct PageEntry {
    int32_t key_block_id = -1;
    int32_t value_block_id = -1;
};

struct LayerPagedKVCache {
    Tensor key_pool;
    Tensor value_pool;

    // logical page -> physical block
    std::vector<PageEntry> page_table;

    LayerPagedKVCache(
        int64_t num_blocks,
        int64_t page_size,
        int64_t num_kv_heads,
        int64_t head_dim,
        DType dtype,
        Device device
    );

    int64_t physical_key_token_index(
        int64_t logical_token_index,
        int64_t page_size
    ) const;

    int64_t physical_value_token_index(
        int64_t logical_token_index,
        int64_t page_size
    ) const;
};

class ModelPagedKVCache {
public:
    ModelPagedKVCache(
        int64_t num_layers,
        int64_t capacity,
        int64_t page_size,
        int64_t num_kv_heads,
        int64_t head_dim,
        DType dtype,
        Device device
    );

    LayerPagedKVCache& layer(int64_t layer_idx);
    const LayerPagedKVCache& layer(int64_t layer_idx) const;

    void update_layer(
        int64_t layer_idx,
        const Tensor& key,
        const Tensor& value
    );

    void advance(int64_t num_tokens);
    void reset();

    int64_t current_len() const;
    int64_t capacity() const;
    int64_t page_size() const;
    int64_t num_blocks() const;
    int64_t num_layers() const;
    int64_t num_kv_heads() const;
    int64_t head_dim() const;

private:
    void check_layer_idx(int64_t layer_idx) const;
    void check_kv_shape(const Tensor& key, const Tensor& value) const;
    void check_can_append(int64_t seq_len) const;

    int64_t num_layers_ = 0;
    int64_t capacity_ = 0;
    int64_t page_size_ = 0;
    int64_t num_blocks_ = 0;
    int64_t num_kv_heads_ = 0;
    int64_t head_dim_ = 0;

    DType dtype_;
    Device device_;

    int64_t current_len_ = 0;

    std::vector<LayerPagedKVCache> layers_;
};

}  // namespace lite_llm
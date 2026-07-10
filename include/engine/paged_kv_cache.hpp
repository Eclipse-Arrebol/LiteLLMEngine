// include/engine/paged_kv_cache.hpp
#pragma once

#include "core/device.hpp"
#include "core/dtype.hpp"
#include "core/tensor.hpp"
#include "engine/block_table_manager.hpp"

#include <cstdint>
#include <vector>

namespace lite_llm {

struct LayerPagedKVCache {
    Tensor key_pool;
    Tensor value_pool;

    LayerPagedKVCache(
        int64_t num_blocks,
        int64_t page_size,
        int64_t num_kv_heads,
        int64_t head_dim,
        DType dtype,
        Device device
    );
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
        const BlockTableManager& table_manager,
        int64_t table_idx,
        int64_t start_pos,
        const Tensor& key,
        const Tensor& value
    );

    void reset();

    int64_t capacity() const;
    int64_t page_size() const;
    int64_t num_blocks() const;
    int64_t num_layers() const;
    int64_t num_kv_heads() const;
    int64_t head_dim() const;

private:
    void check_layer_idx(int64_t layer_idx) const;
    void check_kv_shape(const Tensor& key, const Tensor& value) const;
    void check_write_range(int64_t start_pos, int64_t seq_len) const;

private:
    int64_t num_layers_ = 0;
    int64_t capacity_ = 0;
    int64_t page_size_ = 0;
    int64_t num_blocks_ = 0;
    int64_t num_kv_heads_ = 0;
    int64_t head_dim_ = 0;

    DType dtype_ = DType::FP32;
    Device device_ = Device::CPU;

    std::vector<LayerPagedKVCache> layers_;
};

}  // namespace lite_llm
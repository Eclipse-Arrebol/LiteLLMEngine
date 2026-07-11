// src/engine/paged_kv_cache.cpp

#include "engine/paged_kv_cache.hpp"

#include <stdexcept>


namespace lite_llm {

namespace {

int64_t ceil_div(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

}  // namespace

LayerPagedKVCache::LayerPagedKVCache(
    int64_t num_blocks,
    int64_t page_size,
    int64_t num_kv_heads,
    int64_t head_dim,
    DType dtype,
    Device device
)
    : key_pool(
          std::vector<int64_t>{
              num_blocks * page_size,
              num_kv_heads,
              head_dim
          },
          dtype,
          device
      ),
      value_pool(
          std::vector<int64_t>{
              num_blocks * page_size,
              num_kv_heads,
              head_dim
          },
          dtype,
          device
      ) {
    if (num_blocks <= 0) {
        throw std::runtime_error("num_blocks must be positive");
    }
    if (page_size <= 0) {
        throw std::runtime_error("page_size must be positive");
    }

    key_pool.zero_();
    value_pool.zero_();
}

ModelPagedKVCache::ModelPagedKVCache(
    int64_t num_layers,
    int64_t capacity,
    int64_t page_size,
    int64_t num_kv_heads,
    int64_t head_dim,
    DType dtype,
    Device device
) {
    if (num_layers <= 0) {
        throw std::runtime_error("num_layers must be positive");
    }

    if (capacity <= 0) {
        throw std::runtime_error("capacity must be positive");
    }

    if (page_size <= 0) {
        throw std::runtime_error("page_size must be positive");
    }

    if (num_kv_heads <= 0) {
        throw std::runtime_error("num_kv_heads must be positive");
    }

    if (head_dim <= 0) {
        throw std::runtime_error("head_dim must be positive");
    }

    num_layers_ = num_layers;
    capacity_ = capacity;
    page_size_ = page_size;
    num_blocks_ = ceil_div(capacity_, page_size_);
    num_kv_heads_ = num_kv_heads;
    head_dim_ = head_dim;
    dtype_ = dtype;
    device_ = device;

    layers_.reserve(static_cast<size_t>(num_layers_));

    for (int64_t i = 0; i < num_layers_; ++i) {
        layers_.emplace_back(
            num_blocks_,
            page_size_,
            num_kv_heads_,
            head_dim_,
            dtype_,
            device_
        );
    }
}

LayerPagedKVCache& ModelPagedKVCache::layer(int64_t layer_idx) {
    check_layer_idx(layer_idx);
    return layers_[static_cast<size_t>(layer_idx)];
}

const LayerPagedKVCache& ModelPagedKVCache::layer(
    int64_t layer_idx
) const {
    check_layer_idx(layer_idx);
    return layers_[static_cast<size_t>(layer_idx)];
}

void ModelPagedKVCache::update_layer(
    int64_t layer_idx,
    const BlockTableManager& table_manager,
    int64_t table_idx,
    int64_t start_pos,
    const Tensor& key,
    const Tensor& value
) {
    check_layer_idx(layer_idx);
    check_kv_shape(key, value);

    const int64_t seq_len = key.shape()[0];
    check_write_range(start_pos, seq_len);

    LayerPagedKVCache& layer_cache = layer(layer_idx);

    const size_t row_bytes =
        static_cast<size_t>(num_kv_heads_ * head_dim_) * dtype_size(dtype_);

    for (int64_t token_idx = 0; token_idx < seq_len; ++token_idx) {
        const int64_t logical_token_index = start_pos + token_idx;

        const int64_t physical_token_index =
            table_manager.physical_token_index(
                table_idx,
                logical_token_index,
                page_size_
            );

        const size_t src_offset =
            static_cast<size_t>(token_idx) * row_bytes;

        const size_t dst_offset =
            static_cast<size_t>(physical_token_index) * row_bytes;

        layer_cache.key_pool.copy_from_tensor(
            key,
            dst_offset,
            src_offset,
            row_bytes
        );

        layer_cache.value_pool.copy_from_tensor(
            value,
            dst_offset,
            src_offset,
            row_bytes
        );
    }
}

/**
 * @brief 就从k pool里面把离散的数据整理成连续的output输出出来
 * 
 * @param layer_idx 
 * @param table_manager 
 * @param table_idx 
 * @param start_pos 
 * @param seq_len 
 * @param output 
 */
void ModelPagedKVCache::gather_layer_key(
    int64_t layer_idx,
    const BlockTableManager& table_manager,
    int64_t table_idx,
    int64_t start_pos,
    int64_t seq_len,
    Tensor& output
) const {
    check_layer_idx(layer_idx);
    check_write_range(start_pos, seq_len);
    check_gather_output_shape(output, seq_len);

    const LayerPagedKVCache& layer_cache = layer(layer_idx);

    const size_t row_bytes =
        static_cast<size_t>(num_kv_heads_ * head_dim_) * dtype_size(dtype_);

    for (int64_t token_idx = 0; token_idx < seq_len; ++token_idx) {
        const int64_t logical_token_index = start_pos + token_idx;

        const int64_t physical_token_index =
            table_manager.physical_token_index(
                table_idx,
                logical_token_index,
                page_size_
            );

        const size_t src_offset =
            static_cast<size_t>(physical_token_index) * row_bytes;

        const size_t dst_offset =
            static_cast<size_t>(token_idx) * row_bytes;

        output.copy_from_tensor(
            layer_cache.key_pool,
            dst_offset,
            src_offset,
            row_bytes
        );
    }
}

/**
 * @brief 就从k pool里面把离散的数据整理成连续的output输出出来
 * 
 * @param layer_idx 
 * @param table_manager 
 * @param table_idx 
 * @param start_pos 
 * @param seq_len 
 * @param output 
 */
void ModelPagedKVCache::gather_layer_value(
    int64_t layer_idx,
    const BlockTableManager& table_manager,
    int64_t table_idx,
    int64_t start_pos,
    int64_t seq_len,
    Tensor& output
) const {
    check_layer_idx(layer_idx);
    check_write_range(start_pos, seq_len);
    check_gather_output_shape(output, seq_len);

    const LayerPagedKVCache& layer_cache = layer(layer_idx);

    const size_t row_bytes =
        static_cast<size_t>(num_kv_heads_ * head_dim_) * dtype_size(dtype_);

    for (int64_t token_idx = 0; token_idx < seq_len; ++token_idx) {
        const int64_t logical_token_index = start_pos + token_idx;

        const int64_t physical_token_index =
            table_manager.physical_token_index(
                table_idx,
                logical_token_index,
                page_size_
            );

        const size_t src_offset =
            static_cast<size_t>(physical_token_index) * row_bytes;

        const size_t dst_offset =
            static_cast<size_t>(token_idx) * row_bytes;

        output.copy_from_tensor(
            layer_cache.value_pool,
            dst_offset,
            src_offset,
            row_bytes
        );
    }
}

void ModelPagedKVCache::reset() {
    for (auto& layer_cache : layers_) {
        layer_cache.key_pool.zero_();
        layer_cache.value_pool.zero_();
    }
}

int64_t ModelPagedKVCache::capacity() const {
    return capacity_;
}

int64_t ModelPagedKVCache::page_size() const {
    return page_size_;
}

int64_t ModelPagedKVCache::num_blocks() const {
    return num_blocks_;
}

int64_t ModelPagedKVCache::num_layers() const {
    return num_layers_;
}

int64_t ModelPagedKVCache::num_kv_heads() const {
    return num_kv_heads_;
}

int64_t ModelPagedKVCache::head_dim() const {
    return head_dim_;
}

void ModelPagedKVCache::check_layer_idx(int64_t layer_idx) const {
    if (layer_idx < 0 || layer_idx >= num_layers_) {
        throw std::runtime_error("ModelPagedKVCache layer_idx out of range");
    }
}

void ModelPagedKVCache::check_kv_shape(
    const Tensor& key,
    const Tensor& value
) const {
    if (key.dtype() != dtype_ || value.dtype() != dtype_) {
        throw std::runtime_error("paged kv cache dtype mismatch");
    }

    if (key.device() != device_ || value.device() != device_) {
        throw std::runtime_error("paged kv cache device mismatch");
    }

    if (key.shape().size() != 3 || value.shape().size() != 3) {
        throw std::runtime_error("paged kv cache expects 3D key/value");
    }

    if (key.shape()[0] != value.shape()[0]) {
        throw std::runtime_error("paged kv cache key/value seq_len mismatch");
    }

    if (key.shape()[1] != num_kv_heads_ ||
        value.shape()[1] != num_kv_heads_) {
        throw std::runtime_error("paged kv cache num_kv_heads mismatch");
    }

    if (key.shape()[2] != head_dim_ ||
        value.shape()[2] != head_dim_) {
        throw std::runtime_error("paged kv cache head_dim mismatch");
    }
}

void ModelPagedKVCache::check_write_range(
    int64_t start_pos,
    int64_t seq_len
) const {
    if (start_pos < 0) {
        throw std::runtime_error("paged kv cache start_pos must be non-negative");
    }

    if (seq_len < 0) {
        throw std::runtime_error("paged kv cache seq_len must be non-negative");
    }

    if (start_pos + seq_len > capacity_) {
        throw std::runtime_error("paged kv cache write range exceeds capacity");
    }
}

/**
 * @brief 检测输出的output是不是和模型指定的一样
 * 
 * @param output 
 * @param seq_len 
 */
void ModelPagedKVCache::check_gather_output_shape(
    const Tensor& output,
    int64_t seq_len
) const {
    if (output.dtype() != dtype_) {
        throw std::runtime_error("paged kv cache gather output dtype mismatch");
    }

    if (output.device() != device_) {
        throw std::runtime_error("paged kv cache gather output device mismatch");
    }

    if (output.shape().size() != 3) {
        throw std::runtime_error("paged kv cache gather output expects 3D tensor");
    }

    if (output.shape()[0] != seq_len) {
        throw std::runtime_error("paged kv cache gather output seq_len mismatch");
    }

    if (output.shape()[1] != num_kv_heads_) {
        throw std::runtime_error("paged kv cache gather output num_kv_heads mismatch");
    }

    if (output.shape()[2] != head_dim_) {
        throw std::runtime_error("paged kv cache gather output head_dim mismatch");
    }
}

}  // namespace lite_llm
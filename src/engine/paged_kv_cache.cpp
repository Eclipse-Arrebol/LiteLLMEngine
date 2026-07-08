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
          {num_blocks * page_size, num_kv_heads, head_dim},
          dtype,
          device
      ),
      value_pool(
          {num_blocks * page_size, num_kv_heads, head_dim},
          dtype,
          device
      ) {
    if (num_blocks <= 0) {
        throw std::runtime_error("num_blocks must be positive");
    }
    if (page_size <= 0) {
        throw std::runtime_error("page_size must be positive");
    }

    page_table.reserve(static_cast<size_t>(num_blocks));

    for (int64_t i = 0; i < num_blocks; ++i) {
        PageEntry entry;
        entry.key_block_id = static_cast<int32_t>(i);
        entry.value_block_id = static_cast<int32_t>(i);
        page_table.push_back(entry);
    }

    key_pool.zero_();
    value_pool.zero_();
}

int64_t LayerPagedKVCache::physical_key_token_index(
    int64_t logical_token_index,
    int64_t page_size
) const {
    const int64_t logical_page = logical_token_index / page_size;
    const int64_t page_offset = logical_token_index % page_size;

    if (logical_page < 0 ||
        logical_page >= static_cast<int64_t>(page_table.size())) {
        throw std::runtime_error("logical page out of range for key cache");
    }

    const int64_t physical_block = page_table[logical_page].key_block_id;
    return physical_block * page_size + page_offset;
}

int64_t LayerPagedKVCache::physical_value_token_index(
    int64_t logical_token_index,
    int64_t page_size
) const {
    const int64_t logical_page = logical_token_index / page_size;
    const int64_t page_offset = logical_token_index % page_size;

    if (logical_page < 0 ||
        logical_page >= static_cast<int64_t>(page_table.size())) {
        throw std::runtime_error("logical page out of range for value cache");
    }

    const int64_t physical_block = page_table[logical_page].value_block_id;
    return physical_block * page_size + page_offset;
}

ModelPagedKVCache::ModelPagedKVCache(
    int64_t num_layers,
    int64_t capacity,
    int64_t page_size,
    int64_t num_kv_heads,
    int64_t head_dim,
    DType dtype,
    Device device
)
    : num_layers_(num_layers),
      capacity_(capacity),
      page_size_(page_size),
      num_blocks_(ceil_div(capacity, page_size)),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim),
      dtype_(dtype),
      device_(device) {
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

const LayerPagedKVCache& ModelPagedKVCache::layer(int64_t layer_idx) const {
    check_layer_idx(layer_idx);
    return layers_[static_cast<size_t>(layer_idx)];
}

void ModelPagedKVCache::update_layer(
    int64_t layer_idx,
    const Tensor& key,
    const Tensor& value
) {
    check_layer_idx(layer_idx);
    check_kv_shape(key, value);

    const int64_t seq_len = key.shape()[0];
    check_can_append(seq_len);

    LayerPagedKVCache& layer_cache = layer(layer_idx);

    const size_t row_bytes =
        static_cast<size_t>(num_kv_heads_ * head_dim_) * dtype_size(dtype_);

    for (int64_t token_idx = 0; token_idx < seq_len; ++token_idx) {
        const int64_t logical_token_index = current_len_ + token_idx;

        const int64_t physical_key_index =
            layer_cache.physical_key_token_index(
                logical_token_index,
                page_size_
            );

        const int64_t physical_value_index =
            layer_cache.physical_value_token_index(
                logical_token_index,
                page_size_
            );

        const size_t src_offset =
            static_cast<size_t>(token_idx) * row_bytes;

        const size_t key_dst_offset =
            static_cast<size_t>(physical_key_index) * row_bytes;

        const size_t value_dst_offset =
            static_cast<size_t>(physical_value_index) * row_bytes;

        layer_cache.key_pool.copy_from_tensor(
            key,
            key_dst_offset,
            src_offset,
            row_bytes
        );

        layer_cache.value_pool.copy_from_tensor(
            value,
            value_dst_offset,
            src_offset,
            row_bytes
        );
    }
}

void ModelPagedKVCache::advance(int64_t num_tokens) {
    if (num_tokens < 0) {
        throw std::runtime_error("advance num_tokens must be non-negative");
    }

    if (current_len_ + num_tokens > capacity_) {
        throw std::runtime_error("paged kv cache capacity exceeded in advance");
    }

    current_len_ += num_tokens;
}

void ModelPagedKVCache::reset() {
    current_len_ = 0;

    for (auto& layer_cache : layers_) {
        layer_cache.key_pool.zero_();
        layer_cache.value_pool.zero_();
    }
}

int64_t ModelPagedKVCache::current_len() const {
    return current_len_;
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

void ModelPagedKVCache::check_can_append(int64_t seq_len) const {
    if (seq_len < 0) {
        throw std::runtime_error("seq_len must be non-negative");
    }

    if (current_len_ + seq_len > capacity_) {
        throw std::runtime_error("paged kv cache capacity exceeded");
    }
}

}  // namespace lite_llm
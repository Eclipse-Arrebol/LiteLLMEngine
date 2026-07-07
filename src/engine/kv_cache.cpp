#include "engine/kv_cache.hpp"

#include <cstring>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

namespace lite_llm {
namespace {

int64_t checked_seq_len_from_kv(
    const Tensor& key,
    const Tensor& value,
    int64_t num_kv_heads,
    int64_t head_dim,
    DType dtype,
    Device device
) {
    if (key.dtype() != dtype || value.dtype() != dtype) {
        throw std::runtime_error("KVCache dtype mismatch");
    }

    if (key.device() != device || value.device() != device) {
        throw std::runtime_error("KVCache device mismatch");
    }

    const auto& key_shape = key.shape();
    const auto& value_shape = value.shape();

    if (key_shape.size() != 3) {
        throw std::runtime_error("KVCache key must be 3D");
    }

    if (value_shape.size() != 3) {
        throw std::runtime_error("KVCache value must be 3D");
    }

    if (key_shape != value_shape) {
        throw std::runtime_error("KVCache key/value shape mismatch");
    }

    if (key_shape[0] <= 0) {
        throw std::runtime_error("KVCache seq_len must be positive");
    }

    if (key_shape[1] != num_kv_heads) {
        throw std::runtime_error("KVCache num_kv_heads mismatch");
    }

    if (key_shape[2] != head_dim) {
        throw std::runtime_error("KVCache head_dim mismatch");
    }

    return key_shape[0];
}

void copy_kv_rows(
    Tensor& dst,
    const Tensor& src,
    int64_t dst_token_offset,
    int64_t seq_len,
    int64_t num_kv_heads,
    int64_t head_dim,
    DType dtype
) {
    const size_t row_bytes =
        static_cast<size_t>(num_kv_heads) *
        static_cast<size_t>(head_dim) *
        dtype_size(dtype);

    const size_t dst_offset_bytes =
        static_cast<size_t>(dst_token_offset) * row_bytes;

    const size_t src_offset_bytes = 0;

    const size_t bytes =
        static_cast<size_t>(seq_len) * row_bytes;

    dst.copy_from_tensor(
        src,
        dst_offset_bytes,
        src_offset_bytes,
        bytes
    );
}

}  // namespace



ModelKVCache::ModelKVCache(
    const ModelConfig& config,
    Device device,
    DType dtype,
    int64_t initial_capacity
)
    : num_layers_(config.num_hidden_layers),
      num_kv_heads_(config.num_key_value_heads),
      head_dim_(config.head_dim),
      current_len_(0),
      capacity_(initial_capacity),
      device_(device),
      dtype_(dtype) {
    if (num_layers_ <= 0) {
        throw std::runtime_error("ModelKVCache num_layers must be positive");
    }

    if (num_kv_heads_ <= 0) {
        throw std::runtime_error("ModelKVCache num_kv_heads must be positive");
    }

    if (head_dim_ <= 0) {
        throw std::runtime_error("ModelKVCache head_dim must be positive");
    }

    if (initial_capacity < 0) {
        throw std::runtime_error("ModelKVCache initial_capacity must be non-negative");
    }

    layers_.resize(static_cast<size_t>(num_layers_));

    if (initial_capacity > 0) {
        for (auto& layer : layers_) {
            layer.key = Tensor(
                {initial_capacity, num_kv_heads_, head_dim_},
                dtype_,
                device_
            );

            layer.value = Tensor(
                {initial_capacity, num_kv_heads_, head_dim_},
                dtype_,
                device_
            );

            layer.current_len = 0;
            layer.capacity = initial_capacity;
        }
    }
}

LayerKVCache& ModelKVCache::layer(int64_t layer_idx) {
    if (layer_idx < 0 || layer_idx >= num_layers_) {
        throw std::runtime_error("ModelKVCache layer_idx out of range");
    }

    return layers_[static_cast<size_t>(layer_idx)];
}

const LayerKVCache& ModelKVCache::layer(int64_t layer_idx) const {
    if (layer_idx < 0 || layer_idx >= num_layers_) {
        throw std::runtime_error("ModelKVCache layer_idx out of range");
    }

    return layers_[static_cast<size_t>(layer_idx)];
}

int64_t ModelKVCache::num_layers() const {
    return num_layers_;
}

int64_t ModelKVCache::current_len() const {
    return current_len_;
}

int64_t ModelKVCache::capacity() const {
    return capacity_;
}

int64_t ModelKVCache::num_kv_heads() const {
    return num_kv_heads_;
}

int64_t ModelKVCache::head_dim() const {
    return head_dim_;
}

void ModelKVCache::reset() {
    current_len_ = 0;

    for (auto& layer : layers_) {
        layer.current_len = 0;
    }
}

void ModelKVCache::ensure_capacity(int64_t required_capacity) {
    if (required_capacity <= capacity_) {
        return;
    }

    if (required_capacity <= 0) {
        throw std::runtime_error(
            "ModelKVCache required_capacity must be positive"
        );
    }

    int64_t new_capacity = capacity_ > 0 ? capacity_ : 16;

    while (new_capacity < required_capacity) {
        new_capacity *= 2;
    }

    std::vector<LayerKVCache> new_layers;
    new_layers.resize(static_cast<size_t>(num_layers_));

    for (int64_t layer_idx = 0; layer_idx < num_layers_; ++layer_idx) {
        LayerKVCache& new_layer =
            new_layers[static_cast<size_t>(layer_idx)];

        new_layer.key = Tensor(
            {new_capacity, num_kv_heads_, head_dim_},
            dtype_,
            device_
        );

        new_layer.value = Tensor(
            {new_capacity, num_kv_heads_, head_dim_},
            dtype_,
            device_
        );

        new_layer.current_len = current_len_;
        new_layer.capacity = new_capacity;

        if (capacity_ > 0 && current_len_ > 0) {
            const LayerKVCache& old_layer =
                layers_[static_cast<size_t>(layer_idx)];

            copy_kv_rows(
                new_layer.key,
                old_layer.key,
                0,
                current_len_,
                num_kv_heads_,
                head_dim_,
                dtype_
            );

            copy_kv_rows(
                new_layer.value,
                old_layer.value,
                0,
                current_len_,
                num_kv_heads_,
                head_dim_,
                dtype_
            );
        }
    }

    layers_ = std::move(new_layers);
    capacity_ = new_capacity;
}

void ModelKVCache::update_layer(
    int64_t layer_idx,
    const Tensor& new_key,
    const Tensor& new_value
) {
    if (layer_idx < 0 || layer_idx >= num_layers_) {
        throw std::runtime_error("ModelKVCache layer_idx out of range");
    }

    const int64_t seq_len = checked_seq_len_from_kv(
        new_key,
        new_value,
        num_kv_heads_,
        head_dim_,
        dtype_,
        device_
    );

    const int64_t required_capacity = current_len_ + seq_len;

    ensure_capacity(required_capacity);

    LayerKVCache& layer_cache = layer(layer_idx);

    copy_kv_rows(
        layer_cache.key,
        new_key,
        current_len_,
        seq_len,
        num_kv_heads_,
        head_dim_,
        dtype_
    );

    copy_kv_rows(
        layer_cache.value,
        new_value,
        current_len_,
        seq_len,
        num_kv_heads_,
        head_dim_,
        dtype_
    );

    layer_cache.current_len = current_len_ + seq_len;
}


/**
 * @brief  所有层都写完后，推进整体长度。
 * 
 * @param seq_len 
 */
void ModelKVCache::advance(int64_t seq_len) {
    if (seq_len <= 0) {
        throw std::runtime_error("ModelKVCache advance seq_len must be positive");
    }

    const int64_t new_len = current_len_ + seq_len;

    if (new_len > capacity_) {
        throw std::runtime_error("ModelKVCache advance exceeds capacity");
    }

    current_len_ = new_len;

    for (auto& layer : layers_) {
        layer.current_len = current_len_;
    }
}

}  // namespace lite_llm
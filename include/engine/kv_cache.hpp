#pragma once

#include "core/device.hpp"
#include "core/dtype.hpp"
#include "core/tensor.hpp"
#include "model/model_config.hpp"

#include <cstdint>
#include <vector>

namespace lite_llm {

struct LayerKVCache {
    Tensor key;
    Tensor value;

    int64_t current_len = 0;
    int64_t capacity = 0;
};

class ModelKVCache {
public:
    ModelKVCache() = default;

    ModelKVCache(
        const ModelConfig& config,
        Device device,
        DType dtype = DType::FP32,
        int64_t initial_capacity = 0
    );

    ModelKVCache(const ModelKVCache&) = delete;
    ModelKVCache& operator=(const ModelKVCache&) = delete;

    ModelKVCache(ModelKVCache&&) noexcept = default;
    ModelKVCache& operator=(ModelKVCache&&) noexcept = default;

    LayerKVCache& layer(int64_t layer_idx);
    const LayerKVCache& layer(int64_t layer_idx) const;

    int64_t num_layers() const;
    int64_t current_len() const;
    int64_t capacity() const;

    int64_t num_kv_heads() const;
    int64_t head_dim() const;

    void reset();

    // 先只声明，下一步再实现
    void update_layer(
        int64_t layer_idx,
        const Tensor& new_key,
        const Tensor& new_value
    );

    void advance(int64_t seq_len);

private:
    void ensure_capacity(int64_t required_capacity);

private:
    std::vector<LayerKVCache> layers_;

    int64_t num_layers_ = 0;
    int64_t num_kv_heads_ = 0;
    int64_t head_dim_ = 0;

    int64_t current_len_ = 0;
    int64_t capacity_ = 0;

    Device device_ = Device::CPU;
    DType dtype_ = DType::FP32;
};

}  // namespace lite_llm
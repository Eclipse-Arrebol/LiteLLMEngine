#include "engine/kv_cache.hpp"

#include "core/device.hpp"
#include "core/dtype.hpp"
#include "model/model_config.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool cond, const std::string& msg) {
    if (!cond) {
        throw std::runtime_error(msg);
    }
}

void require_eq_i64(
    int64_t actual,
    int64_t expected,
    const std::string& name
) {
    if (actual != expected) {
        throw std::runtime_error(
            name + " mismatch, actual=" + std::to_string(actual) +
            ", expected=" + std::to_string(expected)
        );
    }
}

void require_shape(
    const lite_llm::Tensor& tensor,
    const std::vector<int64_t>& expected,
    const std::string& name
) {
    const auto& shape = tensor.shape();

    if (shape.size() != expected.size()) {
        throw std::runtime_error(name + " rank mismatch");
    }

    for (size_t i = 0; i < expected.size(); ++i) {
        if (shape[i] != expected[i]) {
            throw std::runtime_error(
                name + " shape mismatch at dim " + std::to_string(i) +
                ", actual=" + std::to_string(shape[i]) +
                ", expected=" + std::to_string(expected[i])
            );
        }
    }
}

lite_llm::ModelConfig make_test_config() {
    lite_llm::ModelConfig config;

    config.num_hidden_layers = 28;
    config.num_key_value_heads = 8;
    config.head_dim = 128;

    return config;
}

}  // namespace

int main() {
    const lite_llm::ModelConfig config = make_test_config();

    constexpr int64_t initial_capacity = 16;

    lite_llm::ModelKVCache cache(
        config,
        lite_llm::Device::CPU,
        lite_llm::DType::FP32,
        initial_capacity
    );

    require_eq_i64(cache.num_layers(), 28, "num_layers");
    require_eq_i64(cache.num_kv_heads(), 8, "num_kv_heads");
    require_eq_i64(cache.head_dim(), 128, "head_dim");
    require_eq_i64(cache.current_len(), 0, "current_len");
    require_eq_i64(cache.capacity(), initial_capacity, "capacity");

    for (int64_t layer_idx = 0; layer_idx < cache.num_layers(); ++layer_idx) {
        const lite_llm::LayerKVCache& layer = cache.layer(layer_idx);

        require_eq_i64(layer.current_len, 0, "layer current_len");
        require_eq_i64(layer.capacity, initial_capacity, "layer capacity");

        require_shape(
            layer.key,
            {initial_capacity, 8, 128},
            "layer key"
        );

        require_shape(
            layer.value,
            {initial_capacity, 8, 128},
            "layer value"
        );
    }

    cache.reset();

    require_eq_i64(cache.current_len(), 0, "current_len after reset");

    for (int64_t layer_idx = 0; layer_idx < cache.num_layers(); ++layer_idx) {
        const lite_llm::LayerKVCache& layer = cache.layer(layer_idx);
        require_eq_i64(layer.current_len, 0, "layer current_len after reset");
    }

    bool caught_low = false;
    try {
        (void)cache.layer(-1);
    } catch (const std::exception&) {
        caught_low = true;
    }

    require(caught_low, "cache.layer(-1) should throw");

    bool caught_high = false;
    try {
        (void)cache.layer(28);
    } catch (const std::exception&) {
        caught_high = true;
    }

    require(caught_high, "cache.layer(28) should throw");

    std::cout << "[test_kv_cache] passed\n";
    return 0;
}
#include "engine/kv_cache.hpp"

#include "core/device.hpp"
#include "core/dtype.hpp"
#include "core/tensor.hpp"
#include "model/model_config.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lite_llm;

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

void require_close(
    float actual,
    float expected,
    const std::string& name
) {
    const float diff = std::fabs(actual - expected);

    if (diff > 1e-5f) {
        throw std::runtime_error(
            name + " mismatch, actual=" + std::to_string(actual) +
            ", expected=" + std::to_string(expected)
        );
    }
}

ModelConfig make_test_config() {
    ModelConfig config;

    config.num_hidden_layers = 2;
    config.num_key_value_heads = 2;
    config.head_dim = 3;

    return config;
}

Tensor make_fp32_tensor(
    const std::vector<int64_t>& shape,
    const std::vector<float>& data,
    Device device
) {
    Tensor tensor(
        shape,
        DType::FP32,
        device
    );

    require_eq_i64(
        static_cast<int64_t>(tensor.numel()),
        static_cast<int64_t>(data.size()),
        "tensor numel"
    );

    tensor.copy_from_cpu(
        data.data(),
        data.size() * sizeof(float)
    );

    return tensor;
}

std::vector<float> copy_tensor_to_cpu(const Tensor& tensor) {
    require(
        tensor.dtype() == DType::FP32,
        "copy_tensor_to_cpu only supports FP32"
    );

    std::vector<float> data(tensor.numel());

    tensor.copy_to_cpu(
        data.data(),
        data.size() * sizeof(float)
    );

    return data;
}

int64_t flat_index(
    int64_t token_idx,
    int64_t head_idx,
    int64_t dim_idx,
    int64_t num_heads,
    int64_t head_dim
) {
    return (token_idx * num_heads + head_idx) * head_dim + dim_idx;
}

void check_cache_token_block(
    const std::vector<float>& cache_data,
    int64_t token_offset,
    int64_t seq_len,
    int64_t num_heads,
    int64_t head_dim,
    const std::vector<float>& expected,
    const std::string& name
) {
    require_eq_i64(
        static_cast<int64_t>(expected.size()),
        seq_len * num_heads * head_dim,
        name + " expected size"
    );

    for (int64_t t = 0; t < seq_len; ++t) {
        for (int64_t h = 0; h < num_heads; ++h) {
            for (int64_t d = 0; d < head_dim; ++d) {
                const int64_t cache_idx = flat_index(
                    token_offset + t,
                    h,
                    d,
                    num_heads,
                    head_dim
                );

                const int64_t expected_idx = flat_index(
                    t,
                    h,
                    d,
                    num_heads,
                    head_dim
                );

                require_close(
                    cache_data[static_cast<size_t>(cache_idx)],
                    expected[static_cast<size_t>(expected_idx)],
                    name
                );
            }
        }
    }
}

void run_kv_cache_update_test(Device device) {
    const std::string device_name =
        device == Device::CPU ? "cpu" : "cuda";

    std::cout << "[test_kv_cache_update_" << device_name << "] start\n";

    const ModelConfig config = make_test_config();

    constexpr int64_t num_layers = 2;
    constexpr int64_t num_heads = 2;
    constexpr int64_t head_dim = 3;
    constexpr int64_t initial_capacity = 2;

    ModelKVCache cache(
        config,
        device,
        DType::FP32,
        initial_capacity
    );

    require_eq_i64(cache.num_layers(), num_layers, "num_layers");
    require_eq_i64(cache.current_len(), 0, "initial current_len");
    require_eq_i64(cache.capacity(), initial_capacity, "initial capacity");

    std::vector<float> key_1 = {
        100, 101, 102,
        103, 104, 105,

        106, 107, 108,
        109, 110, 111,
    };

    std::vector<float> value_1 = {
        200, 201, 202,
        203, 204, 205,

        206, 207, 208,
        209, 210, 211,
    };

    Tensor key_tensor_1 = make_fp32_tensor(
        {2, num_heads, head_dim},
        key_1,
        device
    );

    Tensor value_tensor_1 = make_fp32_tensor(
        {2, num_heads, head_dim},
        value_1,
        device
    );

    cache.update_layer(0, key_tensor_1, value_tensor_1);

    require_eq_i64(
        cache.current_len(),
        0,
        "current_len after first update_layer"
    );

    // 如果你后面删掉了 LayerKVCache::current_len，这一行也删掉
    require_eq_i64(
        cache.layer(0).current_len,
        2,
        "layer 0 current_len after first update"
    );

    cache.advance(2);

    require_eq_i64(cache.current_len(), 2, "current_len after first advance");
    require_eq_i64(cache.capacity(), 2, "capacity after first update");

    {
        const std::vector<float> cache_key =
            copy_tensor_to_cpu(cache.layer(0).key);

        const std::vector<float> cache_value =
            copy_tensor_to_cpu(cache.layer(0).value);

        check_cache_token_block(
            cache_key,
            0,
            2,
            num_heads,
            head_dim,
            key_1,
            "key_1 data"
        );

        check_cache_token_block(
            cache_value,
            0,
            2,
            num_heads,
            head_dim,
            value_1,
            "value_1 data"
        );
    }

    std::vector<float> key_2 = {
        300, 301, 302,
        303, 304, 305,

        306, 307, 308,
        309, 310, 311,

        312, 313, 314,
        315, 316, 317,
    };

    std::vector<float> value_2 = {
        400, 401, 402,
        403, 404, 405,

        406, 407, 408,
        409, 410, 411,

        412, 413, 414,
        415, 416, 417,
    };

    Tensor key_tensor_2 = make_fp32_tensor(
        {3, num_heads, head_dim},
        key_2,
        device
    );

    Tensor value_tensor_2 = make_fp32_tensor(
        {3, num_heads, head_dim},
        value_2,
        device
    );

    cache.update_layer(0, key_tensor_2, value_tensor_2);

    require(
        cache.capacity() >= 5,
        "capacity should expand to at least 5"
    );

    require_eq_i64(
        cache.current_len(),
        2,
        "current_len before second advance"
    );

    cache.advance(3);

    require_eq_i64(cache.current_len(), 5, "current_len after second advance");

    {
        const std::vector<float> cache_key =
            copy_tensor_to_cpu(cache.layer(0).key);

        const std::vector<float> cache_value =
            copy_tensor_to_cpu(cache.layer(0).value);

        check_cache_token_block(
            cache_key,
            0,
            2,
            num_heads,
            head_dim,
            key_1,
            "old key data after expansion"
        );

        check_cache_token_block(
            cache_value,
            0,
            2,
            num_heads,
            head_dim,
            value_1,
            "old value data after expansion"
        );

        check_cache_token_block(
            cache_key,
            2,
            3,
            num_heads,
            head_dim,
            key_2,
            "new key data after expansion"
        );

        check_cache_token_block(
            cache_value,
            2,
            3,
            num_heads,
            head_dim,
            value_2,
            "new value data after expansion"
        );
    }

    cache.reset();

    require_eq_i64(cache.current_len(), 0, "current_len after reset");

    std::cout << "[test_kv_cache_update_" << device_name << "] passed\n";
}

void test_kv_cache_shape_mismatch() {
    std::cout << "[test_kv_cache_shape_mismatch] start\n";

    ModelConfig config = make_test_config();

    ModelKVCache cache(
        config,
        Device::CPU,
        DType::FP32,
        2
    );

    std::vector<float> key_cpu = {
        1, 2, 3,
        4, 5, 6,
    };

    std::vector<float> value_cpu = {
        1, 2, 3,
        4, 5, 6,
    };

    Tensor key = make_fp32_tensor(
        {1, 2, 3},
        key_cpu,
        Device::CPU
    );

    // 错误：value shape 和 key shape 不一致
    Tensor value = make_fp32_tensor(
        {2, 1, 3},
        value_cpu,
        Device::CPU
    );

    bool caught = false;

    try {
        cache.update_layer(0, key, value);
    } catch (const std::runtime_error&) {
        caught = true;
    }

    if (!caught) {
        throw std::runtime_error("Expected KVCache shape mismatch error");
    }

    std::cout << "[test_kv_cache_shape_mismatch] passed\n";
}

}  // namespace

int main() {
    try {
        run_kv_cache_update_test(Device::CPU);
        run_kv_cache_update_test(Device::CUDA);

        test_kv_cache_shape_mismatch();
    } catch (const std::exception& e) {
        std::cerr << "[test_kv_cache_update] failed: "
                  << e.what()
                  << "\n";
        return 1;
    }

    std::cout << "[test_kv_cache_update] all passed\n";
    return 0;
}
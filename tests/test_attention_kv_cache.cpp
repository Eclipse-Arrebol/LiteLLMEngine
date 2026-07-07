#include "ops/attention.hpp"

#include "core/device.hpp"
#include "core/dtype.hpp"
#include "core/tensor.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lite_llm;

namespace {

void check_close(float a, float b, float tol = 1e-4f) {
    if (std::fabs(a - b) > tol) {
        throw std::runtime_error(
            "check_close failed: got " + std::to_string(a) +
            ", expected " + std::to_string(b)
        );
    }
}

Tensor make_fp32_tensor(
    const std::vector<int64_t>& shape,
    const std::vector<float>& data,
    Device device
) {
    Tensor tensor(shape, DType::FP32, device);

    if (tensor.numel() != data.size()) {
        throw std::runtime_error("make_fp32_tensor numel mismatch");
    }

    tensor.copy_from_cpu(
        data.data(),
        data.size() * sizeof(float)
    );

    return tensor;
}

std::vector<float> tensor_to_cpu(const Tensor& tensor) {
    if (tensor.dtype() != DType::FP32) {
        throw std::runtime_error("tensor_to_cpu only supports FP32");
    }

    std::vector<float> data(tensor.numel(), 0.0f);

    tensor.copy_to_cpu(
        data.data(),
        data.size() * sizeof(float)
    );

    return data;
}

int64_t offset3(
    int64_t token,
    int64_t head,
    int64_t dim,
    int64_t num_heads,
    int64_t head_dim
) {
    return (token * num_heads + head) * head_dim + dim;
}

std::vector<float> reference_flash_attention_kv_cache(
    const std::vector<float>& q,
    const std::vector<float>& key_cache,
    const std::vector<float>& value_cache,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t kv_seq_len
) {
    std::vector<float> output(
        static_cast<size_t>(num_q_heads * head_dim),
        0.0f
    );

    const int64_t group_size = num_q_heads / num_kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    for (int64_t q_head = 0; q_head < num_q_heads; ++q_head) {
        const int64_t kv_head = q_head / group_size;

        std::vector<float> scores(static_cast<size_t>(kv_seq_len), 0.0f);

        float max_score = -std::numeric_limits<float>::infinity();

        for (int64_t token = 0; token < kv_seq_len; ++token) {
            float dot = 0.0f;

            for (int64_t d = 0; d < head_dim; ++d) {
                const int64_t q_idx = offset3(
                    0,
                    q_head,
                    d,
                    num_q_heads,
                    head_dim
                );

                const int64_t k_idx = offset3(
                    token,
                    kv_head,
                    d,
                    num_kv_heads,
                    head_dim
                );

                dot += q[static_cast<size_t>(q_idx)] *
                       key_cache[static_cast<size_t>(k_idx)];
            }

            const float score = dot * scale;
            scores[static_cast<size_t>(token)] = score;
            max_score = std::max(max_score, score);
        }

        float denom = 0.0f;

        for (int64_t token = 0; token < kv_seq_len; ++token) {
            denom += std::exp(scores[static_cast<size_t>(token)] - max_score);
        }

        for (int64_t token = 0; token < kv_seq_len; ++token) {
            const float weight =
                std::exp(scores[static_cast<size_t>(token)] - max_score) /
                denom;

            for (int64_t d = 0; d < head_dim; ++d) {
                const int64_t out_idx = offset3(
                    0,
                    q_head,
                    d,
                    num_q_heads,
                    head_dim
                );

                const int64_t v_idx = offset3(
                    token,
                    kv_head,
                    d,
                    num_kv_heads,
                    head_dim
                );

                output[static_cast<size_t>(out_idx)] +=
                    weight * value_cache[static_cast<size_t>(v_idx)];
            }
        }
    }

    return output;
}

void run_flash_attention_kv_cache_test(Device device) {
    const std::string device_name =
        device == Device::CPU ? "cpu" : "cuda";

    std::cout << "[test_attention_kv_cache_" << device_name << "] start\n";

    constexpr int64_t num_q_heads = 4;
    constexpr int64_t num_kv_heads = 2;
    constexpr int64_t head_dim = 2;
    constexpr int64_t capacity = 5;
    constexpr int64_t kv_seq_len = 3;

    // q shape: [1, 4, 2]
    std::vector<float> q_cpu = {
        1.0f, 0.0f,
        0.5f, 1.0f,
        0.0f, 1.0f,
        1.0f, 1.0f,
    };

    // key_cache shape: [5, 2, 2]
    // 只使用前 3 个 token，后 2 个 token 故意填大值，验证 kv_seq_len 生效。
    std::vector<float> key_cache_cpu = {
        // token 0
        1.0f, 0.0f,
        0.0f, 1.0f,

        // token 1
        0.0f, 1.0f,
        1.0f, 0.0f,

        // token 2
        1.0f, 1.0f,
        1.0f, 1.0f,

        // token 3, invalid area
        100.0f, 100.0f,
        100.0f, 100.0f,

        // token 4, invalid area
        -100.0f, -100.0f,
        -100.0f, -100.0f,
    };

    // value_cache shape: [5, 2, 2]
    std::vector<float> value_cache_cpu = {
        // token 0
        1.0f, 10.0f,
        2.0f, 20.0f,

        // token 1
        3.0f, 30.0f,
        4.0f, 40.0f,

        // token 2
        5.0f, 50.0f,
        6.0f, 60.0f,

        // token 3, invalid area
        1000.0f, 1000.0f,
        1000.0f, 1000.0f,

        // token 4, invalid area
        -1000.0f, -1000.0f,
        -1000.0f, -1000.0f,
    };

    const std::vector<float> expected =
        reference_flash_attention_kv_cache(
            q_cpu,
            key_cache_cpu,
            value_cache_cpu,
            num_q_heads,
            num_kv_heads,
            head_dim,
            kv_seq_len
        );

    Tensor q(
        {1, num_q_heads, head_dim},
        DType::FP32,
        device
    );
    q.copy_from_cpu(q_cpu.data(), q_cpu.size() * sizeof(float));

    Tensor key_cache(
        {capacity, num_kv_heads, head_dim},
        DType::FP32,
        device
    );
    key_cache.copy_from_cpu(
        key_cache_cpu.data(),
        key_cache_cpu.size() * sizeof(float)
    );

    Tensor value_cache(
        {capacity, num_kv_heads, head_dim},
        DType::FP32,
        device
    );
    value_cache.copy_from_cpu(
        value_cache_cpu.data(),
        value_cache_cpu.size() * sizeof(float)
    );

    Tensor output(
        {1, num_q_heads, head_dim},
        DType::FP32,
        device
    );
    output.zero_();

    flash_attention_kv_cache(
        q,
        key_cache,
        value_cache,
        kv_seq_len,
        output
    );

    const std::vector<float> output_cpu = tensor_to_cpu(output);

    if (output_cpu.size() != expected.size()) {
        throw std::runtime_error("output size mismatch");
    }

    for (size_t i = 0; i < expected.size(); ++i) {
        check_close(output_cpu[i], expected[i]);
    }

    std::cout << "[test_attention_kv_cache_" << device_name << "] passed\n";
}

void test_flash_attention_kv_cache_invalid_kv_seq_len() {
    std::cout << "[test_attention_kv_cache_invalid_kv_seq_len] start\n";

    constexpr int64_t num_q_heads = 2;
    constexpr int64_t num_kv_heads = 1;
    constexpr int64_t head_dim = 2;
    constexpr int64_t capacity = 2;

    Tensor q(
        {1, num_q_heads, head_dim},
        DType::FP32,
        Device::CPU
    );

    Tensor key_cache(
        {capacity, num_kv_heads, head_dim},
        DType::FP32,
        Device::CPU
    );

    Tensor value_cache(
        {capacity, num_kv_heads, head_dim},
        DType::FP32,
        Device::CPU
    );

    Tensor output(
        {1, num_q_heads, head_dim},
        DType::FP32,
        Device::CPU
    );

    bool caught_zero = false;

    try {
        flash_attention_kv_cache(
            q,
            key_cache,
            value_cache,
            0,
            output
        );
    } catch (const std::runtime_error&) {
        caught_zero = true;
    }

    if (!caught_zero) {
        throw std::runtime_error(
            "Expected flash_attention_kv_cache kv_seq_len=0 error"
        );
    }

    bool caught_too_large = false;

    try {
        flash_attention_kv_cache(
            q,
            key_cache,
            value_cache,
            capacity + 1,
            output
        );
    } catch (const std::runtime_error&) {
        caught_too_large = true;
    }

    if (!caught_too_large) {
        throw std::runtime_error(
            "Expected flash_attention_kv_cache kv_seq_len > capacity error"
        );
    }

    std::cout << "[test_attention_kv_cache_invalid_kv_seq_len] passed\n";
}

}  // namespace

int main() {
    try {
        run_flash_attention_kv_cache_test(Device::CPU);
        run_flash_attention_kv_cache_test(Device::CUDA);

        test_flash_attention_kv_cache_invalid_kv_seq_len();
    } catch (const std::exception& e) {
        std::cerr << "[test_attention_kv_cache] failed: "
                  << e.what()
                  << "\n";
        return 1;
    }

    std::cout << "[test_attention_kv_cache] all passed\n";
    return 0;
}
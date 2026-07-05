#include "ops/attention.hpp"

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

int64_t offset3(
    int64_t token,
    int64_t head,
    int64_t dim,
    int64_t num_heads,
    int64_t head_dim
) {
    return (token * num_heads + head) * head_dim + dim;
}

std::vector<float> reference_causal_attention(
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    int64_t num_tokens,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim
) {
    std::vector<float> output(
        num_tokens * num_q_heads * head_dim,
        0.0f
    );

    const int64_t group_size = num_q_heads / num_kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::vector<float> scores(num_tokens, 0.0f);

    for (int64_t token = 0; token < num_tokens; ++token) {
        for (int64_t q_head = 0; q_head < num_q_heads; ++q_head) {
            const int64_t kv_head = q_head / group_size;

            float max_score = -std::numeric_limits<float>::infinity();

            for (int64_t key_token = 0; key_token <= token; ++key_token) {
                float dot = 0.0f;

                for (int64_t d = 0; d < head_dim; ++d) {
                    dot += q[offset3(token, q_head, d, num_q_heads, head_dim)] *
                           k[offset3(key_token, kv_head, d, num_kv_heads, head_dim)];
                }

                const float score = dot * scale;
                scores[key_token] = score;

                if (score > max_score) {
                    max_score = score;
                }
            }

            float sum_exp = 0.0f;

            for (int64_t key_token = 0; key_token <= token; ++key_token) {
                const float e = std::exp(scores[key_token] - max_score);
                scores[key_token] = e;
                sum_exp += e;
            }

            for (int64_t d = 0; d < head_dim; ++d) {
                float out = 0.0f;

                for (int64_t key_token = 0; key_token <= token; ++key_token) {
                    const float prob = scores[key_token] / sum_exp;

                    out += prob *
                           v[offset3(key_token, kv_head, d, num_kv_heads, head_dim)];
                }

                output[offset3(token, q_head, d, num_q_heads, head_dim)] = out;
            }
        }
    }

    return output;
}

Tensor make_float_tensor(
    const std::vector<int64_t>& shape,
    const std::vector<float>& data,
    Device device
) {
    Tensor tensor(shape, DType::FP32, device);
    tensor.copy_from_cpu(data.data(), data.size() * sizeof(float));
    return tensor;
}

void run_flash_attention_test(Device device) {
    const std::string device_name = device == Device::CPU ? "cpu" : "cuda";

    std::cout << "[test_flash_attention_" << device_name << "] start\n";

    constexpr int64_t num_tokens = 3;
    constexpr int64_t num_q_heads = 4;
    constexpr int64_t num_kv_heads = 2;
    constexpr int64_t head_dim = 4;

    // q shape: [3, 4, 4]
    std::vector<float> q_cpu = {
        // token 0
         1.0f,  0.0f,  0.5f, -1.0f,
         0.5f,  1.0f, -0.5f,  2.0f,
        -1.0f,  0.5f,  1.5f,  0.0f,
         2.0f, -1.0f,  0.0f,  0.5f,

        // token 1
         0.0f,  1.0f,  2.0f, -0.5f,
         1.5f, -0.5f,  0.0f,  1.0f,
        -0.5f,  2.0f, -1.0f,  0.5f,
         1.0f,  1.0f,  0.5f, -1.5f,

        // token 2
         2.0f, -1.0f,  1.0f,  0.0f,
        -1.0f,  0.0f,  1.0f,  2.0f,
         0.5f, -0.5f,  2.0f,  1.0f,
         1.5f,  0.5f, -1.0f,  0.0f,
    };

    // k shape: [3, 2, 4]
    std::vector<float> k_cpu = {
        // token 0
         1.0f,  0.0f, -1.0f,  0.5f,
         0.5f, -0.5f,  1.0f,  2.0f,

        // token 1
         0.0f,  1.5f,  0.5f, -1.0f,
         1.0f,  1.0f, -0.5f,  0.0f,

        // token 2
        -1.0f,  0.5f,  2.0f,  1.0f,
         0.0f, -1.5f,  1.5f,  0.5f,
    };

    // v shape: [3, 2, 4]
    std::vector<float> v_cpu = {
        // token 0
         1.0f,  2.0f,  3.0f,  4.0f,
        -1.0f,  0.5f,  1.5f,  2.5f,

        // token 1
         0.0f, -1.0f,  2.0f, -2.0f,
         3.0f,  1.0f, -0.5f,  0.0f,

        // token 2
         2.0f,  0.0f, -1.0f,  1.0f,
        -2.0f,  2.0f,  0.5f, -1.5f,
    };

    std::vector<float> expected = reference_causal_attention(
        q_cpu,
        k_cpu,
        v_cpu,
        num_tokens,
        num_q_heads,
        num_kv_heads,
        head_dim
    );

    Tensor q = make_float_tensor(
        {num_tokens, num_q_heads, head_dim},
        q_cpu,
        device
    );

    Tensor k = make_float_tensor(
        {num_tokens, num_kv_heads, head_dim},
        k_cpu,
        device
    );

    Tensor v = make_float_tensor(
        {num_tokens, num_kv_heads, head_dim},
        v_cpu,
        device
    );

    Tensor output(
        {num_tokens, num_q_heads, head_dim},
        DType::FP32,
        device
    );

    output.zero_();

    flash_attention(q, k, v, output);

    std::vector<float> output_cpu(expected.size(), 0.0f);
    output.copy_to_cpu(output_cpu.data(), output_cpu.size() * sizeof(float));

    for (size_t i = 0; i < expected.size(); ++i) {
        check_close(output_cpu[i], expected[i]);
    }

    std::cout << "[test_flash_attention_" << device_name << "] passed\n";
}

void test_flash_attention_single_token_causal() {
    std::cout << "[test_flash_attention_single_token_causal] start\n";

    constexpr int64_t num_tokens = 1;
    constexpr int64_t num_q_heads = 2;
    constexpr int64_t num_kv_heads = 1;
    constexpr int64_t head_dim = 4;

    std::vector<float> q_cpu = {
        1.0f, 2.0f, 3.0f, 4.0f,
        0.5f, 1.0f, 1.5f, 2.0f,
    };

    std::vector<float> k_cpu = {
        2.0f, 1.0f, 0.0f, -1.0f,
    };

    std::vector<float> v_cpu = {
        10.0f, 20.0f, 30.0f, 40.0f,
    };

    Tensor q = make_float_tensor(
        {num_tokens, num_q_heads, head_dim},
        q_cpu,
        Device::CPU
    );

    Tensor k = make_float_tensor(
        {num_tokens, num_kv_heads, head_dim},
        k_cpu,
        Device::CPU
    );

    Tensor v = make_float_tensor(
        {num_tokens, num_kv_heads, head_dim},
        v_cpu,
        Device::CPU
    );

    Tensor output(
        {num_tokens, num_q_heads, head_dim},
        DType::FP32,
        Device::CPU
    );

    flash_attention(q, k, v, output);

    std::vector<float> output_cpu(num_q_heads * head_dim, 0.0f);
    output.copy_to_cpu(output_cpu.data(), output_cpu.size() * sizeof(float));

    // 单 token causal attention，只能看自己，所以两个 q_head 输出都应该等于 v[0]
    for (int64_t head = 0; head < num_q_heads; ++head) {
        for (int64_t d = 0; d < head_dim; ++d) {
            check_close(
                output_cpu[offset3(0, head, d, num_q_heads, head_dim)],
                v_cpu[d]
            );
        }
    }

    std::cout << "[test_flash_attention_single_token_causal] passed\n";
}

void test_flash_attention_output_shape_mismatch() {
    std::cout << "[test_flash_attention_output_shape_mismatch] start\n";

    std::vector<float> q_cpu = {
        1.0f, 2.0f, 3.0f, 4.0f,
    };

    std::vector<float> k_cpu = {
        1.0f, 2.0f, 3.0f, 4.0f,
    };

    std::vector<float> v_cpu = {
        1.0f, 2.0f, 3.0f, 4.0f,
    };

    Tensor q = make_float_tensor({1, 1, 4}, q_cpu, Device::CPU);
    Tensor k = make_float_tensor({1, 1, 4}, k_cpu, Device::CPU);
    Tensor v = make_float_tensor({1, 1, 4}, v_cpu, Device::CPU);

    Tensor output({1, 4}, DType::FP32, Device::CPU);

    bool caught = false;

    try {
        flash_attention(q, k, v, output);
    } catch (const std::runtime_error&) {
        caught = true;
    }

    if (!caught) {
        throw std::runtime_error("Expected flash_attention output shape mismatch error");
    }

    std::cout << "[test_flash_attention_output_shape_mismatch] passed\n";
}

void test_flash_attention_gqa_head_mismatch() {
    std::cout << "[test_flash_attention_gqa_head_mismatch] start\n";

    std::vector<float> q_cpu(1 * 3 * 4, 1.0f);
    std::vector<float> k_cpu(1 * 2 * 4, 1.0f);
    std::vector<float> v_cpu(1 * 2 * 4, 1.0f);

    Tensor q = make_float_tensor({1, 3, 4}, q_cpu, Device::CPU);
    Tensor k = make_float_tensor({1, 2, 4}, k_cpu, Device::CPU);
    Tensor v = make_float_tensor({1, 2, 4}, v_cpu, Device::CPU);

    Tensor output({1, 3, 4}, DType::FP32, Device::CPU);

    bool caught = false;

    try {
        flash_attention(q, k, v, output);
    } catch (const std::runtime_error&) {
        caught = true;
    }

    if (!caught) {
        throw std::runtime_error("Expected flash_attention GQA head mismatch error");
    }

    std::cout << "[test_flash_attention_gqa_head_mismatch] passed\n";
}

}  // namespace

int main() {
    try {
        run_flash_attention_test(Device::CPU);
        run_flash_attention_test(Device::CUDA);

        test_flash_attention_single_token_causal();
        test_flash_attention_output_shape_mismatch();
        test_flash_attention_gqa_head_mismatch();
    } catch (const std::exception& e) {
        std::cerr << "[test_attention] failed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "[test_attention] all passed\n";
    return 0;
}
#include "layers/rotary.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lite_llm;

namespace {

void check_close(float a, float b, float tol = 1e-5f) {
    if (std::fabs(a - b) > tol) {
        throw std::runtime_error(
            "check_close failed: got " + std::to_string(a) +
            ", expected " + std::to_string(b)
        );
    }
}

float rope_inv_freq(int64_t dim_index, int64_t head_dim, float rope_theta) {
    return std::pow(
        rope_theta,
        -static_cast<float>(2 * dim_index) / static_cast<float>(head_dim)
    );
}

std::vector<float> reference_rotary_one(
    const std::vector<float>& input,
    const std::vector<int32_t>& position_ids,
    int64_t num_tokens,
    int64_t num_heads,
    int64_t head_dim,
    float rope_theta
) {
    std::vector<float> output(input.size(), 0.0f);

    const int64_t half_dim = head_dim / 2;

    for (int64_t token = 0; token < num_tokens; ++token) {
        const int32_t pos = position_ids[token];

        for (int64_t head = 0; head < num_heads; ++head) {
            const int64_t base = (token * num_heads + head) * head_dim;

            for (int64_t i = 0; i < half_dim; ++i) {
                const float inv_freq = rope_inv_freq(i, head_dim, rope_theta);
                const float angle = static_cast<float>(pos) * inv_freq;

                const float c = std::cos(angle);
                const float s = std::sin(angle);

                const float x1 = input[base + i];
                const float x2 = input[base + i + half_dim];

                output[base + i] = x1 * c - x2 * s;
                output[base + i + half_dim] = x2 * c + x1 * s;
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

Tensor make_int_tensor(
    const std::vector<int64_t>& shape,
    const std::vector<int32_t>& data,
    Device device
) {
    Tensor tensor(shape, DType::INT32, device);
    tensor.copy_from_cpu(data.data(), data.size() * sizeof(int32_t));
    return tensor;
}

void run_rotary_test(Device device) {
    const std::string device_name = device == Device::CPU ? "cpu" : "cuda";

    std::cout << "[test_rotary_" << device_name << "] start\n";

    constexpr int64_t num_tokens = 2;
    constexpr int64_t num_q_heads = 2;
    constexpr int64_t num_kv_heads = 1;
    constexpr int64_t head_dim = 4;
    constexpr float rope_theta = 10000.0f;

    std::vector<int32_t> position_ids_cpu = {
        0, 2,
    };

    // q shape: [num_tokens, num_q_heads, head_dim] = [2, 2, 4]
    std::vector<float> q_cpu = {
        // token 0, head 0
        1.0f, 2.0f, 3.0f, 4.0f,
        // token 0, head 1
        -1.0f, 0.5f, 2.0f, -0.5f,

        // token 1, head 0
        0.0f, 1.0f, -2.0f, 3.0f,
        // token 1, head 1
        4.0f, -1.0f, 0.25f, -0.75f,
    };

    // k shape: [num_tokens, num_kv_heads, head_dim] = [2, 1, 4]
    std::vector<float> k_cpu = {
        // token 0, head 0
        0.5f, -1.0f, 1.5f, 2.0f,

        // token 1, head 0
        -2.0f, 3.0f, 0.0f, 1.0f,
    };

    std::vector<float> expected_q = reference_rotary_one(
        q_cpu,
        position_ids_cpu,
        num_tokens,
        num_q_heads,
        head_dim,
        rope_theta
    );

    std::vector<float> expected_k = reference_rotary_one(
        k_cpu,
        position_ids_cpu,
        num_tokens,
        num_kv_heads,
        head_dim,
        rope_theta
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

    Tensor position_ids = make_int_tensor(
        {num_tokens},
        position_ids_cpu,
        device
    );

    Tensor q_out(
        {num_tokens, num_q_heads, head_dim},
        DType::FP32,
        device
    );

    Tensor k_out(
        {num_tokens, num_kv_heads, head_dim},
        DType::FP32,
        device
    );

    q_out.zero_();
    k_out.zero_();

    RotaryEmbedding rotary(head_dim, rope_theta);
    rotary.apply(q, k, position_ids, q_out, k_out);

    std::vector<float> q_out_cpu(expected_q.size(), 0.0f);
    std::vector<float> k_out_cpu(expected_k.size(), 0.0f);

    q_out.copy_to_cpu(q_out_cpu.data(), q_out_cpu.size() * sizeof(float));
    k_out.copy_to_cpu(k_out_cpu.data(), k_out_cpu.size() * sizeof(float));

    for (size_t i = 0; i < expected_q.size(); ++i) {
        check_close(q_out_cpu[i], expected_q[i]);
    }

    for (size_t i = 0; i < expected_k.size(); ++i) {
        check_close(k_out_cpu[i], expected_k[i]);
    }

    std::cout << "[test_rotary_" << device_name << "] passed\n";
}

void test_rotary_position_shape_mismatch() {
    std::cout << "[test_rotary_position_shape_mismatch] start\n";

    constexpr int64_t head_dim = 4;
    constexpr float rope_theta = 10000.0f;

    std::vector<float> q_cpu = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
    };

    std::vector<float> k_cpu = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
    };

    std::vector<int32_t> bad_position_ids_cpu = {
        0,
    };

    Tensor q = make_float_tensor({2, 1, 4}, q_cpu, Device::CPU);
    Tensor k = make_float_tensor({2, 1, 4}, k_cpu, Device::CPU);
    Tensor position_ids = make_int_tensor({1}, bad_position_ids_cpu, Device::CPU);

    Tensor q_out({2, 1, 4}, DType::FP32, Device::CPU);
    Tensor k_out({2, 1, 4}, DType::FP32, Device::CPU);

    RotaryEmbedding rotary(head_dim, rope_theta);

    bool caught = false;

    try {
        rotary.apply(q, k, position_ids, q_out, k_out);
    } catch (const std::runtime_error&) {
        caught = true;
    }

    if (!caught) {
        throw std::runtime_error("Expected RotaryEmbedding position_ids shape mismatch error");
    }

    std::cout << "[test_rotary_position_shape_mismatch] passed\n";
}

void test_rotary_output_shape_mismatch() {
    std::cout << "[test_rotary_output_shape_mismatch] start\n";

    constexpr int64_t head_dim = 4;
    constexpr float rope_theta = 10000.0f;

    std::vector<float> q_cpu = {
        1.0f, 2.0f, 3.0f, 4.0f,
    };

    std::vector<float> k_cpu = {
        1.0f, 2.0f, 3.0f, 4.0f,
    };

    std::vector<int32_t> position_ids_cpu = {
        0,
    };

    Tensor q = make_float_tensor({1, 1, 4}, q_cpu, Device::CPU);
    Tensor k = make_float_tensor({1, 1, 4}, k_cpu, Device::CPU);
    Tensor position_ids = make_int_tensor({1}, position_ids_cpu, Device::CPU);

    Tensor q_out({1, 4}, DType::FP32, Device::CPU);
    Tensor k_out({1, 1, 4}, DType::FP32, Device::CPU);

    RotaryEmbedding rotary(head_dim, rope_theta);

    bool caught = false;

    try {
        rotary.apply(q, k, position_ids, q_out, k_out);
    } catch (const std::runtime_error&) {
        caught = true;
    }

    if (!caught) {
        throw std::runtime_error("Expected RotaryEmbedding q_out shape mismatch error");
    }

    std::cout << "[test_rotary_output_shape_mismatch] passed\n";
}

}  // namespace

int main() {
    try {
        run_rotary_test(Device::CPU);
        run_rotary_test(Device::CUDA);

        test_rotary_position_shape_mismatch();
        test_rotary_output_shape_mismatch();
    } catch (const std::exception& e) {
        std::cerr << "[test_rotary] failed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "[test_rotary] all passed\n";
    return 0;
}
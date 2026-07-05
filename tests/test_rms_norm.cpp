#include "layers/rms_norm.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
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

std::vector<float> reference_rms_norm(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    int64_t num_tokens,
    int64_t hidden_size,
    float eps
) {
    std::vector<float> output(input.size(), 0.0f);

    for (int64_t t = 0; t < num_tokens; ++t) {
        const int64_t offset = t * hidden_size;

        float sum_sq = 0.0f;
        for (int64_t h = 0; h < hidden_size; ++h) {
            const float v = input[offset + h];
            sum_sq += v * v;
        }

        const float mean_sq = sum_sq / static_cast<float>(hidden_size);
        const float inv_rms = 1.0f / std::sqrt(mean_sq + eps);

        for (int64_t h = 0; h < hidden_size; ++h) {
            output[offset + h] = input[offset + h] * inv_rms * weight[h];
        }
    }

    return output;
}

void test_rms_norm_cpu() {
    std::cout << "[test_rms_norm_cpu] start\n";

    constexpr int64_t num_tokens = 2;
    constexpr int64_t hidden_size = 4;
    constexpr float eps = 1e-6f;

    std::vector<float> input_cpu = {
        1.0f, 2.0f, 3.0f, 4.0f,
        2.0f, 0.0f, -2.0f, 4.0f,
    };

    std::vector<float> weight_cpu = {
        1.0f, 1.5f, 2.0f, 0.5f,
    };

    std::vector<float> expected = reference_rms_norm(
        input_cpu,
        weight_cpu,
        num_tokens,
        hidden_size,
        eps
    );

    Tensor input({num_tokens, hidden_size}, DType::FP32, Device::CPU);
    input.copy_from_cpu(input_cpu.data(), input_cpu.size() * sizeof(float));

    Tensor weight({hidden_size}, DType::FP32, Device::CPU);
    weight.copy_from_cpu(weight_cpu.data(), weight_cpu.size() * sizeof(float));

    Tensor output({num_tokens, hidden_size}, DType::FP32, Device::CPU);
    output.zero_();

    RMSNorm rms_norm(eps);
    rms_norm.load_weight(std::move(weight));
    rms_norm.forward(input, output);

    std::vector<float> output_cpu(input_cpu.size());
    output.copy_to_cpu(output_cpu.data(), output_cpu.size() * sizeof(float));

    for (size_t i = 0; i < expected.size(); ++i) {
        check_close(output_cpu[i], expected[i]);
    }

    std::cout << "[test_rms_norm_cpu] passed\n";
}

void test_rms_norm_cuda() {
    std::cout << "[test_rms_norm_cuda] start\n";

    constexpr int64_t num_tokens = 2;
    constexpr int64_t hidden_size = 4;
    constexpr float eps = 1e-6f;

    std::vector<float> input_cpu = {
        1.0f, 2.0f, 3.0f, 4.0f,
        2.0f, 0.0f, -2.0f, 4.0f,
    };

    std::vector<float> weight_cpu = {
        1.0f, 1.5f, 2.0f, 0.5f,
    };

    std::vector<float> expected = reference_rms_norm(
        input_cpu,
        weight_cpu,
        num_tokens,
        hidden_size,
        eps
    );

    Tensor input({num_tokens, hidden_size}, DType::FP32, Device::CUDA);
    input.copy_from_cpu(input_cpu.data(), input_cpu.size() * sizeof(float));

    Tensor weight({hidden_size}, DType::FP32, Device::CUDA);
    weight.copy_from_cpu(weight_cpu.data(), weight_cpu.size() * sizeof(float));

    Tensor output({num_tokens, hidden_size}, DType::FP32, Device::CUDA);
    output.zero_();

    RMSNorm rms_norm(eps);
    rms_norm.load_weight(std::move(weight));
    rms_norm.forward(input, output);

    std::vector<float> output_cpu(input_cpu.size());
    output.copy_to_cpu(output_cpu.data(), output_cpu.size() * sizeof(float));

    for (size_t i = 0; i < expected.size(); ++i) {
        check_close(output_cpu[i], expected[i]);
    }

    std::cout << "[test_rms_norm_cuda] passed\n";
}

void test_rms_norm_shape_mismatch() {
    std::cout << "[test_rms_norm_shape_mismatch] start\n";

    constexpr float eps = 1e-6f;

    std::vector<float> input_cpu = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
    };

    std::vector<float> weight_cpu = {
        1.0f, 1.0f,
    };

    Tensor input({2, 3}, DType::FP32, Device::CPU);
    input.copy_from_cpu(input_cpu.data(), input_cpu.size() * sizeof(float));

    Tensor weight({2}, DType::FP32, Device::CPU);
    weight.copy_from_cpu(weight_cpu.data(), weight_cpu.size() * sizeof(float));

    Tensor output({2, 3}, DType::FP32, Device::CPU);

    RMSNorm rms_norm(eps);
    rms_norm.load_weight(std::move(weight));

    bool caught = false;

    try {
        rms_norm.forward(input, output);
    } catch (const std::runtime_error&) {
        caught = true;
    }

    if (!caught) {
        throw std::runtime_error("Expected RMSNorm shape mismatch error");
    }

    std::cout << "[test_rms_norm_shape_mismatch] passed\n";
}

}  // namespace

int main() {
    try {
        test_rms_norm_cpu();
        test_rms_norm_cuda();
        test_rms_norm_shape_mismatch();
    } catch (const std::exception& e) {
        std::cerr << "[test_rms_norm] failed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "[test_rms_norm] all passed\n";
    return 0;
}
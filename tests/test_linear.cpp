#include "layers/linear.hpp"

#include <cmath>
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

std::vector<float> reference_linear(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    const std::vector<float>* bias,
    int64_t m,
    int64_t in_features,
    int64_t out_features
) {
    std::vector<float> output(m * out_features, 0.0f);

    for (int64_t row = 0; row < m; ++row) {
        for (int64_t out_col = 0; out_col < out_features; ++out_col) {
            float sum = 0.0f;

            if (bias != nullptr) {
                sum += (*bias)[out_col];
            }

            for (int64_t k = 0; k < in_features; ++k) {
                sum += input[row * in_features + k] *
                       weight[out_col * in_features + k];
            }

            output[row * out_features + out_col] = sum;
        }
    }

    return output;
}

void run_linear_test(Device device, bool with_bias) {
    const std::string device_name = device == Device::CPU ? "cpu" : "cuda";
    const std::string bias_name = with_bias ? "with_bias" : "no_bias";

    std::cout << "[test_linear_" << device_name << "_" << bias_name << "] start\n";

    constexpr int64_t m = 2;
    constexpr int64_t in_features = 3;
    constexpr int64_t out_features = 4;

    // input shape: [M, in_features] = [2, 3]
    std::vector<float> input_cpu = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
    };

    // weight shape: [out_features, in_features] = [4, 3]
    std::vector<float> weight_cpu = {
        1.0f,  2.0f, 3.0f,
        4.0f,  5.0f, 6.0f,
       -1.0f,  0.0f, 1.0f,
        2.0f, -3.0f, 4.0f,
    };

    std::vector<float> bias_cpu = {
        0.5f, -1.0f, 10.0f, 2.0f,
    };

    const std::vector<float>* bias_ref = with_bias ? &bias_cpu : nullptr;

    std::vector<float> expected = reference_linear(
        input_cpu,
        weight_cpu,
        bias_ref,
        m,
        in_features,
        out_features
    );

    Tensor input({m, in_features}, DType::FP32, device);
    input.copy_from_cpu(input_cpu.data(), input_cpu.size() * sizeof(float));

    Tensor weight({out_features, in_features}, DType::FP32, device);
    weight.copy_from_cpu(weight_cpu.data(), weight_cpu.size() * sizeof(float));

    Tensor output({m, out_features}, DType::FP32, device);
    output.zero_();

    Linear linear;
    linear.load_weight(std::move(weight));

    if (with_bias) {
        Tensor bias({out_features}, DType::FP32, device);
        bias.copy_from_cpu(bias_cpu.data(), bias_cpu.size() * sizeof(float));
        linear.load_bias(std::move(bias));
    }

    linear.forward(input, output);

    std::vector<float> output_cpu(expected.size(), 0.0f);
    output.copy_to_cpu(output_cpu.data(), output_cpu.size() * sizeof(float));

    for (size_t i = 0; i < expected.size(); ++i) {
        check_close(output_cpu[i], expected[i]);
    }

    std::cout << "[test_linear_" << device_name << "_" << bias_name << "] passed\n";
}

void test_linear_shape_mismatch() {
    std::cout << "[test_linear_shape_mismatch] start\n";

    std::vector<float> input_cpu = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
    };

    std::vector<float> weight_cpu = {
        1.0f, 2.0f,
        3.0f, 4.0f,
    };

    Tensor input({2, 3}, DType::FP32, Device::CPU);
    input.copy_from_cpu(input_cpu.data(), input_cpu.size() * sizeof(float));

    // 错误：weight shape 是 [2, 2]，但 input in_features 是 3
    Tensor weight({2, 2}, DType::FP32, Device::CPU);
    weight.copy_from_cpu(weight_cpu.data(), weight_cpu.size() * sizeof(float));

    Tensor output({2, 2}, DType::FP32, Device::CPU);

    Linear linear;
    linear.load_weight(std::move(weight));

    bool caught = false;

    try {
        linear.forward(input, output);
    } catch (const std::runtime_error&) {
        caught = true;
    }

    if (!caught) {
        throw std::runtime_error("Expected Linear shape mismatch error");
    }

    std::cout << "[test_linear_shape_mismatch] passed\n";
}

void test_linear_bias_shape_mismatch() {
    std::cout << "[test_linear_bias_shape_mismatch] start\n";

    std::vector<float> input_cpu = {
        1.0f, 2.0f, 3.0f,
    };

    std::vector<float> weight_cpu = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
    };

    std::vector<float> bias_cpu = {
        1.0f,
    };

    Tensor input({1, 3}, DType::FP32, Device::CPU);
    input.copy_from_cpu(input_cpu.data(), input_cpu.size() * sizeof(float));

    Tensor weight({2, 3}, DType::FP32, Device::CPU);
    weight.copy_from_cpu(weight_cpu.data(), weight_cpu.size() * sizeof(float));

    Tensor bias({1}, DType::FP32, Device::CPU);
    bias.copy_from_cpu(bias_cpu.data(), bias_cpu.size() * sizeof(float));

    Tensor output({1, 2}, DType::FP32, Device::CPU);

    Linear linear;
    linear.load_weight(std::move(weight));

    bool caught = false;

    try {
        linear.load_bias(std::move(bias));
        linear.forward(input, output);
    } catch (const std::runtime_error&) {
        caught = true;
    }

    if (!caught) {
        throw std::runtime_error("Expected Linear bias shape mismatch error");
    }

    std::cout << "[test_linear_bias_shape_mismatch] passed\n";
}

}  // namespace

int main() {
    try {
        run_linear_test(Device::CPU, false);
        run_linear_test(Device::CPU, true);

        run_linear_test(Device::CUDA, false);
        run_linear_test(Device::CUDA, true);

        test_linear_shape_mismatch();
        test_linear_bias_shape_mismatch();
    } catch (const std::exception& e) {
        std::cerr << "[test_linear] failed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "[test_linear] all passed\n";
    return 0;
}
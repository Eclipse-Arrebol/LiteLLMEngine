#include "ops/copy.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lite_llm;

namespace {

void check_close(float a, float b, float tol = 1e-6f) {
    if (std::fabs(a - b) > tol) {
        throw std::runtime_error(
            "check_close failed: got " + std::to_string(a) +
            ", expected " + std::to_string(b)
        );
    }
}

void run_tensor_copy_test(Device device) {
    const std::string device_name = device == Device::CPU ? "cpu" : "cuda";

    std::cout << "[test_tensor_copy_" << device_name << "] start\n";

    std::vector<float> input_cpu = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
        10.0f, 11.0f, 12.0f,
    };

    // input shape: [2, 6]
    Tensor input({2, 6}, DType::FP32, device);
    input.copy_from_cpu(input_cpu.data(), input_cpu.size() * sizeof(float));

    // output shape: [2, 3, 2]
    // numel 一样，内存连续布局一样
    Tensor output({2, 3, 2}, DType::FP32, device);
    output.zero_();

    tensor_copy(input, output);

    std::vector<float> output_cpu(input_cpu.size(), 0.0f);
    output.copy_to_cpu(output_cpu.data(), output_cpu.size() * sizeof(float));

    for (size_t i = 0; i < input_cpu.size(); ++i) {
        check_close(output_cpu[i], input_cpu[i]);
    }

    std::cout << "[test_tensor_copy_" << device_name << "] passed\n";
}

void test_tensor_copy_numel_mismatch() {
    std::cout << "[test_tensor_copy_numel_mismatch] start\n";

    std::vector<float> input_cpu = {
        1.0f, 2.0f, 3.0f, 4.0f,
    };

    Tensor input({2, 2}, DType::FP32, Device::CPU);
    input.copy_from_cpu(input_cpu.data(), input_cpu.size() * sizeof(float));

    Tensor output({3, 2}, DType::FP32, Device::CPU);

    bool caught = false;

    try {
        tensor_copy(input, output);
    } catch (const std::runtime_error&) {
        caught = true;
    }

    if (!caught) {
        throw std::runtime_error("Expected tensor_copy numel mismatch error");
    }

    std::cout << "[test_tensor_copy_numel_mismatch] passed\n";
}

void test_tensor_copy_dtype_mismatch() {
    std::cout << "[test_tensor_copy_dtype_mismatch] start\n";

    std::vector<float> input_cpu = {
        1.0f, 2.0f, 3.0f, 4.0f,
    };

    Tensor input({4}, DType::FP32, Device::CPU);
    input.copy_from_cpu(input_cpu.data(), input_cpu.size() * sizeof(float));

    Tensor output({4}, DType::INT32, Device::CPU);

    bool caught = false;

    try {
        tensor_copy(input, output);
    } catch (const std::runtime_error&) {
        caught = true;
    }

    if (!caught) {
        throw std::runtime_error("Expected tensor_copy dtype mismatch error");
    }

    std::cout << "[test_tensor_copy_dtype_mismatch] passed\n";
}

void test_tensor_copy_device_mismatch() {
    std::cout << "[test_tensor_copy_device_mismatch] start\n";

    std::vector<float> input_cpu = {
        1.0f, 2.0f, 3.0f, 4.0f,
    };

    Tensor input({4}, DType::FP32, Device::CPU);
    input.copy_from_cpu(input_cpu.data(), input_cpu.size() * sizeof(float));

    Tensor output({4}, DType::FP32, Device::CUDA);

    bool caught = false;

    try {
        tensor_copy(input, output);
    } catch (const std::runtime_error&) {
        caught = true;
    }

    if (!caught) {
        throw std::runtime_error("Expected tensor_copy device mismatch error");
    }

    std::cout << "[test_tensor_copy_device_mismatch] passed\n";
}

}  // namespace

int main() {
    try {
        run_tensor_copy_test(Device::CPU);
        run_tensor_copy_test(Device::CUDA);

        test_tensor_copy_numel_mismatch();
        test_tensor_copy_dtype_mismatch();
        test_tensor_copy_device_mismatch();
    } catch (const std::exception& e) {
        std::cerr << "[test_copy] failed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "[test_copy] all passed\n";
    return 0;
}
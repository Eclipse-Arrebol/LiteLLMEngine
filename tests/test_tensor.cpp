#include "core/device.hpp"
#include "core/dtype.hpp"
#include "core/tensor.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error("Test failed: " + message);
    }
}

bool almost_equal(float a, float b) {
    return std::fabs(a - b) < 1e-6f;
}

void test_tensor_on_device(lite_llm::Device device) {
    std::vector<float> input = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };

    lite_llm::Tensor tensor({2, 3}, lite_llm::DType::FP32, device);

    expect(tensor.numel() == 6, "Tensor numel should be 6");
    expect(tensor.nbytes() == 24, "Tensor nbytes should be 24");
    expect(tensor.dtype() == lite_llm::DType::FP32, "Tensor dtype should be FP32");
    expect(tensor.device() == device, "Tensor device mismatch");

    tensor.copy_from_cpu(input.data(), input.size() * sizeof(float));

    std::vector<float> output(input.size(), 0.0f);
    tensor.copy_to_cpu(output.data(), output.size() * sizeof(float));

    for (size_t i = 0; i < input.size(); ++i) {
        expect(almost_equal(input[i], output[i]), "Tensor copy result mismatch");
    }

    tensor.zero_();

    std::vector<float> zero_output(input.size(), 1.0f);
    tensor.copy_to_cpu(zero_output.data(), zero_output.size() * sizeof(float));

    for (float value : zero_output) {
        expect(almost_equal(value, 0.0f), "Tensor zero_ result mismatch");
    }
}

bool has_cuda_device() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return err == cudaSuccess && device_count > 0;
}

} // namespace

int main() {
    try {
        std::cout << "[test_tensor] CPU test..." << std::endl;
        test_tensor_on_device(lite_llm::Device::CPU);
        std::cout << "[test_tensor] CPU test passed" << std::endl;

        if (has_cuda_device()) {
            std::cout << "[test_tensor] CUDA test..." << std::endl;
            test_tensor_on_device(lite_llm::Device::CUDA);
            std::cout << "[test_tensor] CUDA test passed" << std::endl;
        } else {
            std::cout << "[test_tensor] CUDA device not found, skipped" << std::endl;
        }

        std::cout << "[test_tensor] all tests passed" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
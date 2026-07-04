#include "core/device.hpp"
#include "core/dtype.hpp"
#include "core/tensor.hpp"
#include "ops/add.hpp"

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

bool has_cuda_device() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return err == cudaSuccess && device_count > 0;
}

void test_add_on_device(lite_llm::Device device) {
    std::vector<float> host_a = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };

    std::vector<float> host_b = {
        10.0f, 20.0f, 30.0f,
        40.0f, 50.0f, 60.0f
    };

    std::vector<float> expected = {
        11.0f, 22.0f, 33.0f,
        44.0f, 55.0f, 66.0f
    };

    lite_llm::Tensor a({2, 3}, lite_llm::DType::FP32, device);
    lite_llm::Tensor b({2, 3}, lite_llm::DType::FP32, device);
    lite_llm::Tensor out({2, 3}, lite_llm::DType::FP32, device);

    a.copy_from_cpu(host_a.data(), host_a.size() * sizeof(float));
    b.copy_from_cpu(host_b.data(), host_b.size() * sizeof(float));

    lite_llm::add(a, b, out);

    std::vector<float> host_out(expected.size(), 0.0f);
    out.copy_to_cpu(host_out.data(), host_out.size() * sizeof(float));

    for (size_t i = 0; i < expected.size(); ++i) {
        expect(
            almost_equal(host_out[i], expected[i]),
            "add result mismatch at index " + std::to_string(i)
        );
    }
}

} // namespace

int main() {
    try {
        std::cout << "[test_add] CPU test..." << std::endl;
        test_add_on_device(lite_llm::Device::CPU);
        std::cout << "[test_add] CPU test passed" << std::endl;

        if (has_cuda_device()) {
            std::cout << "[test_add] CUDA test..." << std::endl;
            test_add_on_device(lite_llm::Device::CUDA);
            std::cout << "[test_add] CUDA test passed" << std::endl;
        } else {
            std::cout << "[test_add] CUDA device not found, skipped" << std::endl;
        }

        std::cout << "[test_add] all tests passed" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
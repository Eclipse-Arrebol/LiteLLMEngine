#include "ops/silu_and_mul.hpp"

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

float silu_ref(float x) {
    return x / (1.0f + std::exp(-x));
}

std::vector<float> reference_silu_and_mul(
    const std::vector<float>& gate,
    const std::vector<float>& up
) {
    if (gate.size() != up.size()) {
        throw std::runtime_error("reference_silu_and_mul shape mismatch");
    }

    std::vector<float> output(gate.size(), 0.0f);

    for (size_t i = 0; i < gate.size(); ++i) {
        output[i] = silu_ref(gate[i]) * up[i];
    }

    return output;
}

void run_silu_and_mul_test(Device device) {
    const std::string device_name = device == Device::CPU ? "cpu" : "cuda";

    std::cout << "[test_silu_and_mul_" << device_name << "] start\n";

    std::vector<float> gate_cpu = {
        -3.0f, -1.0f, 0.0f, 1.0f,
         2.0f,  4.0f, 0.5f, -0.5f,
    };

    std::vector<float> up_cpu = {
        1.0f, 2.0f, -1.0f, 0.5f,
        3.0f, -2.0f, 4.0f, -3.0f,
    };

    std::vector<float> expected = reference_silu_and_mul(gate_cpu, up_cpu);

    Tensor gate({2, 4}, DType::FP32, device);
    gate.copy_from_cpu(gate_cpu.data(), gate_cpu.size() * sizeof(float));

    Tensor up({2, 4}, DType::FP32, device);
    up.copy_from_cpu(up_cpu.data(), up_cpu.size() * sizeof(float));

    Tensor output({2, 4}, DType::FP32, device);
    output.zero_();

    silu_and_mul(gate, up, output);

    std::vector<float> output_cpu(expected.size(), 0.0f);
    output.copy_to_cpu(output_cpu.data(), output_cpu.size() * sizeof(float));

    for (size_t i = 0; i < expected.size(); ++i) {
        check_close(output_cpu[i], expected[i]);
    }

    std::cout << "[test_silu_and_mul_" << device_name << "] passed\n";
}

void test_silu_and_mul_shape_mismatch() {
    std::cout << "[test_silu_and_mul_shape_mismatch] start\n";

    std::vector<float> gate_cpu = {
        1.0f, 2.0f, 3.0f, 4.0f,
    };

    std::vector<float> up_cpu = {
        1.0f, 2.0f,
    };

    Tensor gate({2, 2}, DType::FP32, Device::CPU);
    gate.copy_from_cpu(gate_cpu.data(), gate_cpu.size() * sizeof(float));

    Tensor up({1, 2}, DType::FP32, Device::CPU);
    up.copy_from_cpu(up_cpu.data(), up_cpu.size() * sizeof(float));

    Tensor output({2, 2}, DType::FP32, Device::CPU);

    bool caught = false;

    try {
        silu_and_mul(gate, up, output);
    } catch (const std::runtime_error&) {
        caught = true;
    }

    if (!caught) {
        throw std::runtime_error("Expected silu_and_mul shape mismatch error");
    }

    std::cout << "[test_silu_and_mul_shape_mismatch] passed\n";
}

void test_silu_and_mul_output_shape_mismatch() {
    std::cout << "[test_silu_and_mul_output_shape_mismatch] start\n";

    std::vector<float> gate_cpu = {
        1.0f, 2.0f, 3.0f, 4.0f,
    };

    std::vector<float> up_cpu = {
        1.0f, 2.0f, 3.0f, 4.0f,
    };

    Tensor gate({2, 2}, DType::FP32, Device::CPU);
    gate.copy_from_cpu(gate_cpu.data(), gate_cpu.size() * sizeof(float));

    Tensor up({2, 2}, DType::FP32, Device::CPU);
    up.copy_from_cpu(up_cpu.data(), up_cpu.size() * sizeof(float));

    Tensor output({4}, DType::FP32, Device::CPU);

    bool caught = false;

    try {
        silu_and_mul(gate, up, output);
    } catch (const std::runtime_error&) {
        caught = true;
    }

    if (!caught) {
        throw std::runtime_error("Expected silu_and_mul output shape mismatch error");
    }

    std::cout << "[test_silu_and_mul_output_shape_mismatch] passed\n";
}

}  // namespace

int main() {
    try {
        run_silu_and_mul_test(Device::CPU);
        run_silu_and_mul_test(Device::CUDA);

        test_silu_and_mul_shape_mismatch();
        test_silu_and_mul_output_shape_mismatch();
    } catch (const std::exception& e) {
        std::cerr << "[test_silu_and_mul] failed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "[test_silu_and_mul] all passed\n";
    return 0;
}
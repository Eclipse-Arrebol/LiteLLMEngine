#include "ops/argmax.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lite_llm;

namespace {

void run_argmax_last_token_test(Device device) {
    const std::string device_name = device == Device::CPU ? "cpu" : "cuda";

    std::cout << "[test_argmax_last_token_" << device_name << "] start\n";

    // logits shape: [num_tokens, vocab_size] = [3, 6]
    //
    // 只应该看最后一行：
    // last row = [0.1, 3.0, -1.0, 5.5, 2.0, 4.0]
    // argmax = 3
    std::vector<float> logits_cpu = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
        6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f,
        0.1f, 3.0f, -1.0f, 5.5f, 2.0f, 4.0f,
    };

    Tensor logits({3, 6}, DType::FP32, device);
    logits.copy_from_cpu(logits_cpu.data(), logits_cpu.size() * sizeof(float));

    const int32_t token_id = argmax_last_token(logits);

    if (token_id != 3) {
        throw std::runtime_error(
            "argmax_last_token failed: got " + std::to_string(token_id) +
            ", expected 3"
        );
    }

    std::cout << "[test_argmax_last_token_" << device_name << "] passed\n";
}

void run_argmax_tie_break_test(Device device) {
    const std::string device_name = device == Device::CPU ? "cpu" : "cuda";

    std::cout << "[test_argmax_tie_break_" << device_name << "] start\n";

    // 最后一行最大值 9.0 同时出现在 id=1 和 id=3。
    // 当前 CUDA kernel 写了 tie-break：取更小 id。
    // CPU 版本因为只在 value > best_value 时更新，也会保留更小 id。
    std::vector<float> logits_cpu = {
        0.0f, 0.0f, 0.0f, 0.0f,
        1.0f, 9.0f, 2.0f, 9.0f,
    };

    Tensor logits({2, 4}, DType::FP32, device);
    logits.copy_from_cpu(logits_cpu.data(), logits_cpu.size() * sizeof(float));

    const int32_t token_id = argmax_last_token(logits);

    if (token_id != 1) {
        throw std::runtime_error(
            "argmax_last_token tie-break failed: got " +
            std::to_string(token_id) + ", expected 1"
        );
    }

    std::cout << "[test_argmax_tie_break_" << device_name << "] passed\n";
}

void test_argmax_invalid_dtype() {
    std::cout << "[test_argmax_invalid_dtype] start\n";

    std::vector<int32_t> logits_cpu = {
        1, 2, 3,
    };

    Tensor logits({1, 3}, DType::INT32, Device::CPU);
    logits.copy_from_cpu(logits_cpu.data(), logits_cpu.size() * sizeof(int32_t));

    bool caught = false;

    try {
        (void)argmax_last_token(logits);
    } catch (const std::runtime_error&) {
        caught = true;
    }

    if (!caught) {
        throw std::runtime_error("Expected argmax_last_token dtype error");
    }

    std::cout << "[test_argmax_invalid_dtype] passed\n";
}

void test_argmax_invalid_shape() {
    std::cout << "[test_argmax_invalid_shape] start\n";

    std::vector<float> logits_cpu = {
        1.0f, 2.0f, 3.0f,
    };

    Tensor logits({3}, DType::FP32, Device::CPU);
    logits.copy_from_cpu(logits_cpu.data(), logits_cpu.size() * sizeof(float));

    bool caught = false;

    try {
        (void)argmax_last_token(logits);
    } catch (const std::runtime_error&) {
        caught = true;
    }

    if (!caught) {
        throw std::runtime_error("Expected argmax_last_token shape error");
    }

    std::cout << "[test_argmax_invalid_shape] passed\n";
}

void test_argmax_empty_tokens() {
    std::cout << "[test_argmax_empty_tokens] start\n";

    Tensor logits({0, 5}, DType::FP32, Device::CPU);

    bool caught = false;

    try {
        (void)argmax_last_token(logits);
    } catch (const std::runtime_error&) {
        caught = true;
    }

    if (!caught) {
        throw std::runtime_error("Expected argmax_last_token empty num_tokens error");
    }

    std::cout << "[test_argmax_empty_tokens] passed\n";
}

}  // namespace

int main() {
    try {
        run_argmax_last_token_test(Device::CPU);
        run_argmax_last_token_test(Device::CUDA);

        run_argmax_tie_break_test(Device::CPU);
        run_argmax_tie_break_test(Device::CUDA);

        test_argmax_invalid_dtype();
        test_argmax_invalid_shape();
        test_argmax_empty_tokens();
    } catch (const std::exception& e) {
        std::cerr << "[test_argmax] failed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "[test_argmax] all passed\n";
    return 0;
}
#include "layers/embedding.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace lite_llm;

namespace {

void check_close(float a, float b) {
    if (std::fabs(a - b) > 1e-6f) {
        throw std::runtime_error("check_close failed");
    }
}

void test_embedding_cpu() {
    std::cout << "[test_embedding_cpu] start\n";

    // weight shape: [vocab_size, hidden_size] = [4, 3]
    std::vector<float> weight_cpu = {
        0.0f,  1.0f,  2.0f,
        10.0f, 11.0f, 12.0f,
        20.0f, 21.0f, 22.0f,
        30.0f, 31.0f, 32.0f,
    };

    std::vector<int32_t> input_ids_cpu = {
        2, 0, 3
    };

    std::vector<float> expected = {
        20.0f, 21.0f, 22.0f,
        0.0f,  1.0f,  2.0f,
        30.0f, 31.0f, 32.0f,
    };

    Tensor weight({4, 3}, DType::FP32, Device::CPU);
    weight.copy_from_cpu(weight_cpu.data(), weight_cpu.size() * sizeof(float));

    Tensor input_ids({3}, DType::INT32, Device::CPU);
    input_ids.copy_from_cpu(input_ids_cpu.data(), input_ids_cpu.size() * sizeof(int32_t));

    Tensor output({3, 3}, DType::FP32, Device::CPU);
    output.zero_();

    Embedding embedding;
    embedding.load_weight(std::move(weight));
    embedding.forward(input_ids, output);

    std::vector<float> output_cpu(expected.size());
    output.copy_to_cpu(output_cpu.data(), output_cpu.size() * sizeof(float));

    for (size_t i = 0; i < expected.size(); ++i) {
        check_close(output_cpu[i], expected[i]);
    }

    std::cout << "[test_embedding_cpu] passed\n";
}

void test_embedding_cuda() {
    std::cout << "[test_embedding_cuda] start\n";

    std::vector<float> weight_cpu = {
        0.0f,  1.0f,  2.0f,
        10.0f, 11.0f, 12.0f,
        20.0f, 21.0f, 22.0f,
        30.0f, 31.0f, 32.0f,
    };

    std::vector<int32_t> input_ids_cpu = {
        2, 0, 3
    };

    std::vector<float> expected = {
        20.0f, 21.0f, 22.0f,
        0.0f,  1.0f,  2.0f,
        30.0f, 31.0f, 32.0f,
    };

    Tensor weight({4, 3}, DType::FP32, Device::CUDA);
    weight.copy_from_cpu(weight_cpu.data(), weight_cpu.size() * sizeof(float));

    Tensor input_ids({3}, DType::INT32, Device::CUDA);
    input_ids.copy_from_cpu(input_ids_cpu.data(), input_ids_cpu.size() * sizeof(int32_t));

    Tensor output({3, 3}, DType::FP32, Device::CUDA);
    output.zero_();

    Embedding embedding;
    embedding.load_weight(std::move(weight));
    embedding.forward(input_ids, output);

    std::vector<float> output_cpu(expected.size());
    output.copy_to_cpu(output_cpu.data(), output_cpu.size() * sizeof(float));

    for (size_t i = 0; i < expected.size(); ++i) {
        check_close(output_cpu[i], expected[i]);
    }

    std::cout << "[test_embedding_cuda] passed\n";
}

void test_embedding_cpu_out_of_range() {
    std::cout << "[test_embedding_cpu_out_of_range] start\n";

    std::vector<float> weight_cpu = {
        0.0f,  1.0f,
        10.0f, 11.0f,
    };

    std::vector<int32_t> input_ids_cpu = {
        0, 2
    };

    Tensor weight({2, 2}, DType::FP32, Device::CPU);
    weight.copy_from_cpu(weight_cpu.data(), weight_cpu.size() * sizeof(float));

    Tensor input_ids({2}, DType::INT32, Device::CPU);
    input_ids.copy_from_cpu(input_ids_cpu.data(), input_ids_cpu.size() * sizeof(int32_t));

    Tensor output({2, 2}, DType::FP32, Device::CPU);

    Embedding embedding;
    embedding.load_weight(std::move(weight));

    bool caught = false;

    try {
        embedding.forward(input_ids, output);
    } catch (const std::runtime_error&) {
        caught = true;
    }

    if (!caught) {
        throw std::runtime_error("Expected CPU embedding out-of-range error");
    }

    std::cout << "[test_embedding_cpu_out_of_range] passed\n";
}

void test_embedding_cuda_out_of_range() {
    std::cout << "[test_embedding_cuda_out_of_range] start\n";

    std::vector<float> weight_cpu = {
        0.0f,  1.0f,
        10.0f, 11.0f,
    };

    std::vector<int32_t> input_ids_cpu = {
        0, 2
    };

    Tensor weight({2, 2}, DType::FP32, Device::CUDA);
    weight.copy_from_cpu(weight_cpu.data(), weight_cpu.size() * sizeof(float));

    Tensor input_ids({2}, DType::INT32, Device::CUDA);
    input_ids.copy_from_cpu(input_ids_cpu.data(), input_ids_cpu.size() * sizeof(int32_t));

    Tensor output({2, 2}, DType::FP32, Device::CUDA);

    Embedding embedding;
    embedding.load_weight(std::move(weight));

    bool caught = false;

    try {
        embedding.forward(input_ids, output);
    } catch (const std::runtime_error&) {
        caught = true;
    }

    if (!caught) {
        throw std::runtime_error("Expected CUDA embedding out-of-range error");
    }

    std::cout << "[test_embedding_cuda_out_of_range] passed\n";
}

}  // namespace

int main() {
    try {
        test_embedding_cpu();
        test_embedding_cuda();
        test_embedding_cpu_out_of_range();
        test_embedding_cuda_out_of_range();
    } catch (const std::exception& e) {
        std::cerr << "[test_embedding] failed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "[test_embedding] all passed\n";
    return 0;
}
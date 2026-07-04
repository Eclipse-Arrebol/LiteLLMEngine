#include "core/device.hpp"
#include "core/dtype.hpp"
#include "core/tensor.hpp"
#include "layers/embedding.hpp"
#include "layers/linear.hpp"
#include "layers/rms_norm.hpp"

#include <iostream>
#include <stdexcept>

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error("Test failed: " + message);
    }
}

} // namespace

int main() {
    try {
        std::cout << "[test_layer_skeleton] build layers..." << std::endl;

        lite_llm::Tensor embedding_weight(
            {10, 4},
            lite_llm::DType::FP32,
            lite_llm::Device::CPU
        );

        lite_llm::Embedding embedding(std::move(embedding_weight));

        expect(embedding.initialized(), "Embedding should be initialized");
        expect(embedding.vocab_size() == 10, "Embedding vocab_size mismatch");
        expect(embedding.hidden_size() == 4, "Embedding hidden_size mismatch");
        expect(std::string(embedding.name()) == "Embedding", "Embedding name mismatch");

        lite_llm::Tensor linear_weight(
            {8, 4},
            lite_llm::DType::FP32,
            lite_llm::Device::CPU
        );

        lite_llm::Linear linear(std::move(linear_weight));

        expect(linear.initialized(), "Linear should be initialized");
        expect(linear.in_features() == 4, "Linear in_features mismatch");
        expect(linear.out_features() == 8, "Linear out_features mismatch");
        expect(!linear.has_bias(), "Linear should not have bias");
        expect(std::string(linear.name()) == "Linear", "Linear name mismatch");

        lite_llm::Tensor norm_weight(
            {4},
            lite_llm::DType::FP32,
            lite_llm::Device::CPU
        );

        lite_llm::RMSNorm norm(std::move(norm_weight), 1e-6f);

        expect(norm.initialized(), "RMSNorm should be initialized");
        expect(norm.hidden_size() == 4, "RMSNorm hidden_size mismatch");
        expect(norm.eps() == 1e-6f, "RMSNorm eps mismatch");
        expect(std::string(norm.name()) == "RMSNorm", "RMSNorm name mismatch");

        std::cout << "[test_layer_skeleton] passed" << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
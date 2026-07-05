#include "model/qwen3.hpp"
#include "weights/weight_map.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lite_llm;

namespace {

void check_close(float a, float b, float tol = 1e-4f) {
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

std::vector<float> linear_ref(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    int64_t m,
    int64_t in_features,
    int64_t out_features
) {
    std::vector<float> output(m * out_features, 0.0f);

    for (int64_t row = 0; row < m; ++row) {
        for (int64_t out_col = 0; out_col < out_features; ++out_col) {
            float sum = 0.0f;

            for (int64_t k = 0; k < in_features; ++k) {
                sum += input[row * in_features + k] *
                       weight[out_col * in_features + k];
            }

            output[row * out_features + out_col] = sum;
        }
    }

    return output;
}

std::vector<float> reference_mlp(
    const std::vector<float>& input,
    const std::vector<float>& gate_weight,
    const std::vector<float>& up_weight,
    const std::vector<float>& down_weight,
    int64_t num_tokens,
    int64_t hidden_size,
    int64_t intermediate_size
) {
    std::vector<float> gate = linear_ref(
        input,
        gate_weight,
        num_tokens,
        hidden_size,
        intermediate_size
    );

    std::vector<float> up = linear_ref(
        input,
        up_weight,
        num_tokens,
        hidden_size,
        intermediate_size
    );

    std::vector<float> act(num_tokens * intermediate_size, 0.0f);

    for (size_t i = 0; i < act.size(); ++i) {
        act[i] = silu_ref(gate[i]) * up[i];
    }

    return linear_ref(
        act,
        down_weight,
        num_tokens,
        intermediate_size,
        hidden_size
    );
}

Tensor make_tensor(
    const std::vector<int64_t>& shape,
    const std::vector<float>& data,
    Device device
) {
    Tensor tensor(shape, DType::FP32, device);
    tensor.copy_from_cpu(data.data(), data.size() * sizeof(float));
    return tensor;
}

ModelConfig make_test_config() {
    ModelConfig config;
    config.model_type = "qwen3";
    config.hidden_size = 3;
    config.intermediate_size = 4;

    return config;
}

WeightMap make_test_weights(
    const std::vector<float>& gate_weight,
    const std::vector<float>& up_weight,
    const std::vector<float>& down_weight,
    Device device
) {
    WeightMap weights;

    weights.add(
        "model.layers.0.mlp.gate_proj.weight",
        make_tensor({4, 3}, gate_weight, device)
    );

    weights.add(
        "model.layers.0.mlp.up_proj.weight",
        make_tensor({4, 3}, up_weight, device)
    );

    weights.add(
        "model.layers.0.mlp.down_proj.weight",
        make_tensor({3, 4}, down_weight, device)
    );

    return weights;
}

void run_qwen3_mlp_test(Device device) {
    const std::string device_name = device == Device::CPU ? "cpu" : "cuda";

    std::cout << "[test_qwen3_mlp_" << device_name << "] start\n";

    constexpr int64_t num_tokens = 2;
    constexpr int64_t hidden_size = 3;
    constexpr int64_t intermediate_size = 4;

    std::vector<float> input_cpu = {
        1.0f, -2.0f, 0.5f,
        0.0f,  1.0f, 2.0f,
    };

    // gate_proj.weight shape: [intermediate_size, hidden_size] = [4, 3]
    std::vector<float> gate_weight = {
         0.5f, -1.0f,  0.25f,
         1.0f,  0.0f, -0.5f,
        -0.75f, 0.5f,  1.0f,
         2.0f, -1.0f,  0.0f,
    };

    // up_proj.weight shape: [intermediate_size, hidden_size] = [4, 3]
    std::vector<float> up_weight = {
         1.0f,  2.0f, -1.0f,
        -0.5f,  0.25f, 1.5f,
         2.0f, -1.0f,  0.5f,
         0.0f,  1.0f, -2.0f,
    };

    // down_proj.weight shape: [hidden_size, intermediate_size] = [3, 4]
    std::vector<float> down_weight = {
         1.0f, -1.0f,  0.5f,  2.0f,
         0.0f,  1.5f, -0.5f,  1.0f,
        -1.0f,  0.25f, 1.0f, -0.75f,
    };

    std::vector<float> expected = reference_mlp(
        input_cpu,
        gate_weight,
        up_weight,
        down_weight,
        num_tokens,
        hidden_size,
        intermediate_size
    );

    ModelConfig config = make_test_config();

    Qwen3MLP mlp(config);

    WeightMap weights = make_test_weights(
        gate_weight,
        up_weight,
        down_weight,
        device
    );

    mlp.load_weights(weights, "model.layers.0.mlp");

    if (!weights.empty()) {
        throw std::runtime_error("Qwen3MLP did not consume all weights");
    }

    if (!mlp.initialized()) {
        throw std::runtime_error("Qwen3MLP should be initialized after load_weights");
    }

    Tensor input({num_tokens, hidden_size}, DType::FP32, device);
    input.copy_from_cpu(input_cpu.data(), input_cpu.size() * sizeof(float));

    Tensor output({num_tokens, hidden_size}, DType::FP32, device);
    output.zero_();

    mlp.forward(input, output);

    std::vector<float> output_cpu(expected.size(), 0.0f);
    output.copy_to_cpu(output_cpu.data(), output_cpu.size() * sizeof(float));

    for (size_t i = 0; i < expected.size(); ++i) {
        check_close(output_cpu[i], expected[i]);
    }

    std::cout << "[test_qwen3_mlp_" << device_name << "] passed\n";
}

void test_qwen3_mlp_input_shape_mismatch() {
    std::cout << "[test_qwen3_mlp_input_shape_mismatch] start\n";

    std::vector<float> gate_weight = {
         0.5f, -1.0f,  0.25f,
         1.0f,  0.0f, -0.5f,
        -0.75f, 0.5f,  1.0f,
         2.0f, -1.0f,  0.0f,
    };

    std::vector<float> up_weight = {
         1.0f,  2.0f, -1.0f,
        -0.5f,  0.25f, 1.5f,
         2.0f, -1.0f,  0.5f,
         0.0f,  1.0f, -2.0f,
    };

    std::vector<float> down_weight = {
         1.0f, -1.0f,  0.5f,  2.0f,
         0.0f,  1.5f, -0.5f,  1.0f,
        -1.0f,  0.25f, 1.0f, -0.75f,
    };

    ModelConfig config = make_test_config();
    Qwen3MLP mlp(config);

    WeightMap weights = make_test_weights(
        gate_weight,
        up_weight,
        down_weight,
        Device::CPU
    );

    mlp.load_weights(weights, "model.layers.0.mlp");

    std::vector<float> bad_input_cpu = {
        1.0f, 2.0f,
        3.0f, 4.0f,
    };

    Tensor bad_input({2, 2}, DType::FP32, Device::CPU);
    bad_input.copy_from_cpu(
        bad_input_cpu.data(),
        bad_input_cpu.size() * sizeof(float)
    );

    Tensor output({2, 3}, DType::FP32, Device::CPU);

    bool caught = false;

    try {
        mlp.forward(bad_input, output);
    } catch (const std::runtime_error&) {
        caught = true;
    }

    if (!caught) {
        throw std::runtime_error("Expected Qwen3MLP input shape mismatch error");
    }

    std::cout << "[test_qwen3_mlp_input_shape_mismatch] passed\n";
}

}  // namespace

int main() {
    try {
        run_qwen3_mlp_test(Device::CPU);
        run_qwen3_mlp_test(Device::CUDA);

        test_qwen3_mlp_input_shape_mismatch();
    } catch (const std::exception& e) {
        std::cerr << "[test_qwen3_mlp] failed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "[test_qwen3_mlp] all passed\n";
    return 0;
}
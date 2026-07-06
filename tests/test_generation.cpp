#include "model/qwen3.hpp"
#include "runtime/generation.hpp"
#include "weights/weight_map.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lite_llm;

namespace {

Tensor make_float_tensor(
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

    config.vocab_size = 5;
    config.hidden_size = 3;
    config.intermediate_size = 4;
    config.num_hidden_layers = 1;

    config.num_attention_heads = 2;
    config.num_key_value_heads = 1;
    config.head_dim = 2;

    config.max_position_embeddings = 16;
    config.rms_norm_eps = 1e-6f;
    config.rope_theta = 10000.0f;
    config.tie_word_embeddings = false;

    return config;
}

WeightMap make_test_weights(
    Device device
) {
    WeightMap weights;

    // token 0 -> hidden direction dim 0
    // token 1 -> hidden direction dim 1
    // token 2 -> hidden direction dim 2
    //
    // decoder layer 里所有投影权重为 0，所以 hidden 基本保持 embedding。
    // final norm 后：
    // token 0 大致变成 [sqrt(3), 0, 0]
    // token 1 大致变成 [0, sqrt(3), 0]
    // token 2 大致变成 [0, 0, sqrt(3)]
    std::vector<float> embed_weight = {
        1.0f, 0.0f, 0.0f,   // token 0
        0.0f, 1.0f, 0.0f,   // token 1
        0.0f, 0.0f, 1.0f,   // token 2
        1.0f, 1.0f, 0.0f,   // token 3
       -1.0f, 0.0f, 0.0f,   // token 4
    };

    weights.add(
        "model.embed_tokens.weight",
        make_float_tensor({5, 3}, embed_weight, device)
    );

    weights.add(
        "model.layers.0.input_layernorm.weight",
        make_float_tensor({3}, std::vector<float>{1.0f, 1.0f, 1.0f}, device)
    );

    weights.add(
        "model.layers.0.self_attn.q_proj.weight",
        make_float_tensor({4, 3}, std::vector<float>(4 * 3, 0.0f), device)
    );

    weights.add(
        "model.layers.0.self_attn.k_proj.weight",
        make_float_tensor({2, 3}, std::vector<float>(2 * 3, 0.0f), device)
    );

    weights.add(
        "model.layers.0.self_attn.v_proj.weight",
        make_float_tensor({2, 3}, std::vector<float>(2 * 3, 0.0f), device)
    );

    weights.add(
        "model.layers.0.self_attn.o_proj.weight",
        make_float_tensor({3, 4}, std::vector<float>(3 * 4, 0.0f), device)
    );

    weights.add(
        "model.layers.0.self_attn.q_norm.weight",
        make_float_tensor({2}, std::vector<float>{1.0f, 1.0f}, device)
    );

    weights.add(
        "model.layers.0.self_attn.k_norm.weight",
        make_float_tensor({2}, std::vector<float>{1.0f, 1.0f}, device)
    );

    weights.add(
        "model.layers.0.post_attention_layernorm.weight",
        make_float_tensor({3}, std::vector<float>{1.0f, 1.0f, 1.0f}, device)
    );

    weights.add(
        "model.layers.0.mlp.gate_proj.weight",
        make_float_tensor({4, 3}, std::vector<float>(4 * 3, 0.0f), device)
    );

    weights.add(
        "model.layers.0.mlp.up_proj.weight",
        make_float_tensor({4, 3}, std::vector<float>(4 * 3, 0.0f), device)
    );

    weights.add(
        "model.layers.0.mlp.down_proj.weight",
        make_float_tensor({3, 4}, std::vector<float>(3 * 4, 0.0f), device)
    );

    weights.add(
        "model.norm.weight",
        make_float_tensor({3}, std::vector<float>{1.0f, 1.0f, 1.0f}, device)
    );

    // lm_head.weight: [vocab_size, hidden_size] = [5, 3]
    //
    // 设计成：
    // last token 0 -> next token 1
    // last token 1 -> next token 2
    // last token 2 -> next token 4
    //
    // token 4 作为 eos。
    std::vector<float> lm_head_weight = {
        0.0f, 0.0f, 0.0f,    // token 0
        1.0f, 0.0f, 0.0f,    // token 1: 看 dim 0
        0.0f, 1.0f, 0.0f,    // token 2: 看 dim 1
       -1.0f, -1.0f, -1.0f,  // token 3: 永远较低
        0.0f, 0.0f, 1.0f,    // token 4: 看 dim 2
    };

    weights.add(
        "lm_head.weight",
        make_float_tensor({5, 3}, lm_head_weight, device)
    );

    return weights;
}

void check_ids_equal(
    const std::vector<int32_t>& actual,
    const std::vector<int32_t>& expected
) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error(
            "generated size mismatch: got " +
            std::to_string(actual.size()) +
            ", expected " +
            std::to_string(expected.size())
        );
    }

    for (size_t i = 0; i < expected.size(); ++i) {
        if (actual[i] != expected[i]) {
            throw std::runtime_error(
                "generated id mismatch at " + std::to_string(i) +
                ": got " + std::to_string(actual[i]) +
                ", expected " + std::to_string(expected[i])
            );
        }
    }
}

void run_generate_greedy_test(Device device) {
    const std::string device_name = device == Device::CPU ? "cpu" : "cuda";

    std::cout << "[test_generate_greedy_" << device_name << "] start\n";

    ModelConfig config = make_test_config();
    Qwen3ForCausalLM model(config);

    WeightMap weights = make_test_weights(device);
    model.load_weights(weights);

    if (!weights.empty()) {
        throw std::runtime_error("Qwen3ForCausalLM did not consume all weights");
    }

    if (!model.initialized()) {
        throw std::runtime_error("Qwen3ForCausalLM should be initialized");
    }

    GreedyGenerateOptions options;
    options.max_new_tokens = 8;
    options.eos_token_id = 4;
    options.device = device;

    std::vector<int32_t> input_ids = {
        0,
    };

    std::vector<int32_t> generated = generate_greedy(
        model,
        input_ids,
        options
    );

    // 预期：
    // 初始 input: [0]
    // step 1: 0 -> 1
    // step 2: 1 -> 2
    // step 3: 2 -> 4
    // 4 是 eos，停止。
    std::vector<int32_t> expected = {
        0, 1, 2, 4,
    };

    check_ids_equal(generated, expected);

    std::cout << "[test_generate_greedy_" << device_name << "] passed\n";
}

void test_generate_greedy_zero_new_tokens() {
    std::cout << "[test_generate_greedy_zero_new_tokens] start\n";

    ModelConfig config = make_test_config();
    Qwen3ForCausalLM model(config);

    WeightMap weights = make_test_weights(Device::CPU);
    model.load_weights(weights);

    GreedyGenerateOptions options;
    options.max_new_tokens = 0;
    options.eos_token_id = 4;
    options.device = Device::CPU;

    std::vector<int32_t> input_ids = {
        0, 1,
    };

    std::vector<int32_t> generated = generate_greedy(
        model,
        input_ids,
        options
    );

    check_ids_equal(generated, input_ids);

    std::cout << "[test_generate_greedy_zero_new_tokens] passed\n";
}

void test_generate_greedy_empty_input() {
    std::cout << "[test_generate_greedy_empty_input] start\n";

    ModelConfig config = make_test_config();
    Qwen3ForCausalLM model(config);

    WeightMap weights = make_test_weights(Device::CPU);
    model.load_weights(weights);

    GreedyGenerateOptions options;
    options.max_new_tokens = 1;
    options.eos_token_id = 4;
    options.device = Device::CPU;

    bool caught = false;

    try {
        (void)generate_greedy(model, {}, options);
    } catch (const std::runtime_error&) {
        caught = true;
    }

    if (!caught) {
        throw std::runtime_error("Expected generate_greedy empty input error");
    }

    std::cout << "[test_generate_greedy_empty_input] passed\n";
}

void test_generate_greedy_uninitialized_model() {
    std::cout << "[test_generate_greedy_uninitialized_model] start\n";

    ModelConfig config = make_test_config();
    Qwen3ForCausalLM model(config);

    GreedyGenerateOptions options;
    options.max_new_tokens = 1;
    options.eos_token_id = 4;
    options.device = Device::CPU;

    bool caught = false;

    try {
        (void)generate_greedy(model, {0}, options);
    } catch (const std::runtime_error&) {
        caught = true;
    }

    if (!caught) {
        throw std::runtime_error("Expected generate_greedy uninitialized model error");
    }

    std::cout << "[test_generate_greedy_uninitialized_model] passed\n";
}

}  // namespace

int main() {
    try {
        run_generate_greedy_test(Device::CPU);
        run_generate_greedy_test(Device::CUDA);

        test_generate_greedy_zero_new_tokens();
        test_generate_greedy_empty_input();
        test_generate_greedy_uninitialized_model();
    } catch (const std::exception& e) {
        std::cerr << "[test_generation] failed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "[test_generation] all passed\n";
    return 0;
}
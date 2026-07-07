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

WeightMap make_test_weights(Device device) {
    WeightMap weights;

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

    std::vector<float> lm_head_weight = {
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
       -1.0f, -1.0f, -1.0f,
        0.0f, 0.0f, 1.0f,
    };

    weights.add(
        "lm_head.weight",
        make_float_tensor({5, 3}, lm_head_weight, device)
    );

    return weights;
}

void check_ids_equal(
    const std::vector<int32_t>& actual,
    const std::vector<int32_t>& expected,
    const std::string& name
) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error(
            name + " size mismatch: got " +
            std::to_string(actual.size()) +
            ", expected " +
            std::to_string(expected.size())
        );
    }

    for (size_t i = 0; i < expected.size(); ++i) {
        if (actual[i] != expected[i]) {
            throw std::runtime_error(
                name + " mismatch at " + std::to_string(i) +
                ": got " + std::to_string(actual[i]) +
                ", expected " + std::to_string(expected[i])
            );
        }
    }
}

void run_generate_greedy_kv_cache_test(Device device) {
    const std::string device_name = device == Device::CPU ? "cpu" : "cuda";

    std::cout << "[test_generate_greedy_kv_cache_"
              << device_name
              << "] start\n";

    ModelConfig config = make_test_config();

    Qwen3ForCausalLM model_no_cache(config);
    WeightMap weights_no_cache = make_test_weights(device);
    model_no_cache.load_weights(weights_no_cache);

    if (!weights_no_cache.empty()) {
        throw std::runtime_error("model_no_cache did not consume all weights");
    }

    Qwen3ForCausalLM model_with_cache(config);
    WeightMap weights_with_cache = make_test_weights(device);
    model_with_cache.load_weights(weights_with_cache);

    if (!weights_with_cache.empty()) {
        throw std::runtime_error("model_with_cache did not consume all weights");
    }

    if (!model_no_cache.initialized() || !model_with_cache.initialized()) {
        throw std::runtime_error("models should be initialized");
    }

    GreedyGenerateOptions options;
    options.max_new_tokens = 8;
    options.eos_token_id = 4;
    options.device = device;
    options.verbose = false;

    std::vector<int32_t> input_ids = {
        0,
    };

    const std::vector<int32_t> generated_no_cache =
        generate_greedy(
            model_no_cache,
            input_ids,
            options
        );

    const std::vector<int32_t> generated_with_cache =
        generate_greedy_with_kv_cache(
            model_with_cache,
            input_ids,
            options
        );

    check_ids_equal(
        generated_with_cache,
        generated_no_cache,
        "generate_greedy_with_kv_cache"
    );

    // 当前 eos 不加入 generated，所以预期是 [0, 1, 2]
    check_ids_equal(
        generated_with_cache,
        std::vector<int32_t>{0, 1, 2},
        "expected generated ids"
    );

    std::cout << "[test_generate_greedy_kv_cache_"
              << device_name
              << "] passed\n";
}

void test_generate_greedy_kv_cache_zero_new_tokens() {
    std::cout << "[test_generate_greedy_kv_cache_zero_new_tokens] start\n";

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

    const std::vector<int32_t> generated =
        generate_greedy_with_kv_cache(
            model,
            input_ids,
            options
        );

    check_ids_equal(
        generated,
        input_ids,
        "zero new tokens"
    );

    std::cout << "[test_generate_greedy_kv_cache_zero_new_tokens] passed\n";
}

void test_generate_greedy_kv_cache_empty_input() {
    std::cout << "[test_generate_greedy_kv_cache_empty_input] start\n";

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
        (void)generate_greedy_with_kv_cache(
            model,
            {},
            options
        );
    } catch (const std::runtime_error&) {
        caught = true;
    }

    if (!caught) {
        throw std::runtime_error(
            "Expected generate_greedy_with_kv_cache empty input error"
        );
    }

    std::cout << "[test_generate_greedy_kv_cache_empty_input] passed\n";
}

}  // namespace

int main() {
    try {
        run_generate_greedy_kv_cache_test(Device::CPU);
        run_generate_greedy_kv_cache_test(Device::CUDA);

        test_generate_greedy_kv_cache_zero_new_tokens();
        test_generate_greedy_kv_cache_empty_input();
    } catch (const std::exception& e) {
        std::cerr << "[test_generation_kv_cache] failed: "
                  << e.what()
                  << "\n";
        return 1;
    }

    std::cout << "[test_generation_kv_cache] all passed\n";
    return 0;
}
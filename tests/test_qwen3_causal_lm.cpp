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

void check_close(float a, float b, float tol = 5e-4f) {
    if (std::fabs(a - b) > tol) {
        throw std::runtime_error(
            "check_close failed: got " + std::to_string(a) +
            ", expected " + std::to_string(b)
        );
    }
}

std::vector<float> embedding_ref(
    const std::vector<int32_t>& input_ids,
    const std::vector<float>& embed_weight,
    int64_t vocab_size,
    int64_t hidden_size
) {
    std::vector<float> output(input_ids.size() * hidden_size, 0.0f);

    for (size_t token = 0; token < input_ids.size(); ++token) {
        const int32_t token_id = input_ids[token];

        if (token_id < 0 || token_id >= vocab_size) {
            throw std::runtime_error("embedding_ref token id out of range");
        }

        for (int64_t h = 0; h < hidden_size; ++h) {
            output[static_cast<int64_t>(token) * hidden_size + h] =
                embed_weight[static_cast<int64_t>(token_id) * hidden_size + h];
        }
    }

    return output;
}

std::vector<float> rms_norm_ref(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    int64_t rows,
    int64_t hidden_size,
    float eps
) {
    std::vector<float> output(input.size(), 0.0f);

    for (int64_t row = 0; row < rows; ++row) {
        const int64_t base = row * hidden_size;

        float sum_sq = 0.0f;
        for (int64_t h = 0; h < hidden_size; ++h) {
            const float v = input[base + h];
            sum_sq += v * v;
        }

        const float inv_rms =
            1.0f / std::sqrt(sum_sq / static_cast<float>(hidden_size) + eps);

        for (int64_t h = 0; h < hidden_size; ++h) {
            output[base + h] = input[base + h] * inv_rms * weight[h];
        }
    }

    return output;
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

Tensor make_float_tensor(
    const std::vector<int64_t>& shape,
    const std::vector<float>& data,
    Device device
) {
    Tensor tensor(shape, DType::FP32, device);
    tensor.copy_from_cpu(data.data(), data.size() * sizeof(float));
    return tensor;
}

Tensor make_int_tensor(
    const std::vector<int64_t>& shape,
    const std::vector<int32_t>& data,
    Device device
) {
    Tensor tensor(shape, DType::INT32, device);
    tensor.copy_from_cpu(data.data(), data.size() * sizeof(int32_t));
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
    const std::vector<float>& embed_weight,
    const std::vector<float>& final_norm_weight,
    const std::vector<float>& lm_head_weight,
    Device device
) {
    WeightMap weights;

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
        make_float_tensor({3}, final_norm_weight, device)
    );

    weights.add(
        "lm_head.weight",
        make_float_tensor({5, 3}, lm_head_weight, device)
    );

    return weights;
}

void run_qwen3_causal_lm_test(Device device) {
    const std::string device_name = device == Device::CPU ? "cpu" : "cuda";

    std::cout << "[test_qwen3_causal_lm_" << device_name << "] start\n";

    constexpr int64_t num_tokens = 3;
    constexpr int64_t vocab_size = 5;
    constexpr int64_t hidden_size = 3;
    constexpr float rms_norm_eps = 1e-6f;

    std::vector<int32_t> input_ids_cpu = {
        2, 0, 3,
    };

    std::vector<int32_t> position_ids_cpu = {
        0, 1, 2,
    };

    std::vector<float> embed_weight = {
         0.5f, -1.0f,  1.5f,
         1.0f,  0.0f, -0.5f,
        -1.0f,  2.0f,  0.25f,
         0.0f, -0.75f, 1.25f,
         2.0f,  1.0f, -1.5f,
    };

    std::vector<float> final_norm_weight = {
        1.2f, 0.9f, 1.1f,
    };

    // lm_head.weight shape: [vocab_size, hidden_size] = [5, 3]
    std::vector<float> lm_head_weight = {
         1.0f, -0.5f,  0.25f,
        -1.0f,  1.5f,  0.5f,
         0.0f,  0.75f, -1.0f,
         2.0f, -0.25f, 1.0f,
        -0.5f,  1.0f,  1.25f,
    };

    // 当前测试里 decoder layer 权重全部为 0，所以 decoder 输出等于 embedding 输出。
    std::vector<float> embedded = embedding_ref(
        input_ids_cpu,
        embed_weight,
        vocab_size,
        hidden_size
    );

    std::vector<float> final_hidden = rms_norm_ref(
        embedded,
        final_norm_weight,
        num_tokens,
        hidden_size,
        rms_norm_eps
    );

    std::vector<float> expected_logits = linear_ref(
        final_hidden,
        lm_head_weight,
        num_tokens,
        hidden_size,
        vocab_size
    );

    ModelConfig config = make_test_config();
    Qwen3ForCausalLM model(config);

    WeightMap weights = make_test_weights(
        embed_weight,
        final_norm_weight,
        lm_head_weight,
        device
    );

    model.load_weights(weights);

    if (!weights.empty()) {
        throw std::runtime_error("Qwen3ForCausalLM did not consume all weights");
    }

    if (!model.initialized()) {
        throw std::runtime_error("Qwen3ForCausalLM should be initialized");
    }

    Tensor input_ids = make_int_tensor(
        {num_tokens},
        input_ids_cpu,
        device
    );

    Tensor position_ids = make_int_tensor(
        {num_tokens},
        position_ids_cpu,
        device
    );

    Tensor logits(
        {num_tokens, vocab_size},
        DType::FP32,
        device
    );

    logits.zero_();

    ForwardContext context;
    context.position_ids = &position_ids;
    context.seq_len = num_tokens;
    context.past_len = 0;
    context.use_cache = false;

    model.forward(input_ids, context, logits);

    std::vector<float> logits_cpu(expected_logits.size(), 0.0f);
    logits.copy_to_cpu(
        logits_cpu.data(),
        logits_cpu.size() * sizeof(float)
    );

    for (size_t i = 0; i < expected_logits.size(); ++i) {
        check_close(logits_cpu[i], expected_logits[i]);
    }

    std::cout << "[test_qwen3_causal_lm_" << device_name << "] passed\n";
}

void test_qwen3_causal_lm_logits_shape_mismatch() {
    std::cout << "[test_qwen3_causal_lm_logits_shape_mismatch] start\n";

    ModelConfig config = make_test_config();
    Qwen3ForCausalLM model(config);

    std::vector<float> embed_weight(5 * 3, 0.1f);
    std::vector<float> final_norm_weight(3, 1.0f);
    std::vector<float> lm_head_weight(5 * 3, 0.1f);

    WeightMap weights = make_test_weights(
        embed_weight,
        final_norm_weight,
        lm_head_weight,
        Device::CPU
    );

    model.load_weights(weights);

    std::vector<int32_t> input_ids_cpu = {
        0,
    };

    std::vector<int32_t> position_ids_cpu = {
        0,
    };

    Tensor input_ids = make_int_tensor(
        {1},
        input_ids_cpu,
        Device::CPU
    );

    Tensor position_ids = make_int_tensor(
        {1},
        position_ids_cpu,
        Device::CPU
    );

    Tensor bad_logits({1, 4}, DType::FP32, Device::CPU);

    ForwardContext context;
    context.position_ids = &position_ids;
    context.seq_len = 1;
    context.past_len = 0;
    context.use_cache = false;

    bool caught = false;

    try {
        model.forward(input_ids, context, bad_logits);
    } catch (const std::runtime_error&) {
        caught = true;
    }

    if (!caught) {
        throw std::runtime_error("Expected Qwen3ForCausalLM logits shape mismatch error");
    }

    std::cout << "[test_qwen3_causal_lm_logits_shape_mismatch] passed\n";
}

void test_qwen3_causal_lm_null_position_ids() {
    std::cout << "[test_qwen3_causal_lm_null_position_ids] start\n";

    ModelConfig config = make_test_config();
    Qwen3ForCausalLM model(config);

    std::vector<float> embed_weight(5 * 3, 0.1f);
    std::vector<float> final_norm_weight(3, 1.0f);
    std::vector<float> lm_head_weight(5 * 3, 0.1f);

    WeightMap weights = make_test_weights(
        embed_weight,
        final_norm_weight,
        lm_head_weight,
        Device::CPU
    );

    model.load_weights(weights);

    std::vector<int32_t> input_ids_cpu = {
        0,
    };

    Tensor input_ids = make_int_tensor(
        {1},
        input_ids_cpu,
        Device::CPU
    );

    Tensor logits({1, 5}, DType::FP32, Device::CPU);

    ForwardContext context;
    context.position_ids = nullptr;
    context.seq_len = 1;
    context.past_len = 0;
    context.use_cache = false;

    bool caught = false;

    try {
        model.forward(input_ids, context, logits);
    } catch (const std::runtime_error&) {
        caught = true;
    }

    if (!caught) {
        throw std::runtime_error("Expected Qwen3ForCausalLM null position_ids error");
    }

    std::cout << "[test_qwen3_causal_lm_null_position_ids] passed\n";
}

}  // namespace

int main() {
    try {
        run_qwen3_causal_lm_test(Device::CPU);
        run_qwen3_causal_lm_test(Device::CUDA);

        test_qwen3_causal_lm_logits_shape_mismatch();
        test_qwen3_causal_lm_null_position_ids();
    } catch (const std::exception& e) {
        std::cerr << "[test_qwen3_causal_lm] failed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "[test_qwen3_causal_lm] all passed\n";
    return 0;
}
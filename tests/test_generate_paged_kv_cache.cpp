// tests/test_generate_paged_kv_cache.cpp

#include "core/tensor.hpp"
#include "model/qwen3.hpp"
#include "runtime/generation.hpp"
#include "weights/weight_map.hpp"

#include <cassert>
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

    if (tensor.numel() != data.size()) {
        throw std::runtime_error("make_float_tensor data size mismatch");
    }

    tensor.copy_from_cpu(
        data.data(),
        data.size() * sizeof(float)
    );

    return tensor;
}

std::vector<float> make_sequential_weight(
    size_t n,
    float base,
    float step
) {
    std::vector<float> data(n);

    for (size_t i = 0; i < n; ++i) {
        data[i] = base + static_cast<float>(i) * step;
    }

    return data;
}

std::vector<float> make_constant_weight(
    size_t n,
    float value
) {
    return std::vector<float>(n, value);
}

ModelConfig make_tiny_config() {
    ModelConfig config;

    config.model_type = "qwen3";

    config.vocab_size = 16;
    config.hidden_size = 4;
    config.intermediate_size = 8;
    config.num_hidden_layers = 1;
    config.num_attention_heads = 2;
    config.num_key_value_heads = 1;

    config.head_dim = 2;
    config.max_position_embeddings = 128;

    config.rms_norm_eps = 1e-6f;
    config.rope_theta = 10000.0f;

    config.tie_word_embeddings = false;

    return config;
}

Qwen3ForCausalLM make_tiny_model(Device device) {
    ModelConfig config = make_tiny_config();

    Qwen3ForCausalLM model(config);

    WeightMap weights;

    /*
     * embedding: [vocab_size, hidden_size] = [16, 4]
     */
    weights.add(
        "model.embed_tokens.weight",
        make_float_tensor(
            {config.vocab_size, config.hidden_size},
            make_sequential_weight(
                static_cast<size_t>(config.vocab_size * config.hidden_size),
                0.01f,
                0.001f
            ),
            device
        )
    );

    /*
     * layer 0 input/post attention norm: [hidden_size]
     */
    weights.add(
        "model.layers.0.input_layernorm.weight",
        make_float_tensor(
            {config.hidden_size},
            make_constant_weight(
                static_cast<size_t>(config.hidden_size),
                1.0f
            ),
            device
        )
    );

    weights.add(
        "model.layers.0.post_attention_layernorm.weight",
        make_float_tensor(
            {config.hidden_size},
            make_constant_weight(
                static_cast<size_t>(config.hidden_size),
                1.0f
            ),
            device
        )
    );

    /*
     * attention:
     *
     * q_proj: [num_q_heads * head_dim, hidden_size] = [4, 4]
     * k_proj: [num_kv_heads * head_dim, hidden_size] = [2, 4]
     * v_proj: [num_kv_heads * head_dim, hidden_size] = [2, 4]
     * o_proj: [hidden_size, num_q_heads * head_dim] = [4, 4]
     */
    weights.add(
        "model.layers.0.self_attn.q_proj.weight",
        make_float_tensor(
            {config.num_attention_heads * config.head_dim, config.hidden_size},
            make_sequential_weight(
                static_cast<size_t>(
                    config.num_attention_heads *
                    config.head_dim *
                    config.hidden_size
                ),
                0.02f,
                0.001f
            ),
            device
        )
    );

    weights.add(
        "model.layers.0.self_attn.k_proj.weight",
        make_float_tensor(
            {config.num_key_value_heads * config.head_dim, config.hidden_size},
            make_sequential_weight(
                static_cast<size_t>(
                    config.num_key_value_heads *
                    config.head_dim *
                    config.hidden_size
                ),
                0.03f,
                0.001f
            ),
            device
        )
    );

    weights.add(
        "model.layers.0.self_attn.v_proj.weight",
        make_float_tensor(
            {config.num_key_value_heads * config.head_dim, config.hidden_size},
            make_sequential_weight(
                static_cast<size_t>(
                    config.num_key_value_heads *
                    config.head_dim *
                    config.hidden_size
                ),
                0.04f,
                0.001f
            ),
            device
        )
    );

    weights.add(
        "model.layers.0.self_attn.o_proj.weight",
        make_float_tensor(
            {config.hidden_size, config.num_attention_heads * config.head_dim},
            make_sequential_weight(
                static_cast<size_t>(
                    config.hidden_size *
                    config.num_attention_heads *
                    config.head_dim
                ),
                0.05f,
                0.001f
            ),
            device
        )
    );

    /*
     * Qwen3 attention q_norm/k_norm: [head_dim]
     */
    weights.add(
        "model.layers.0.self_attn.q_norm.weight",
        make_float_tensor(
            {config.head_dim},
            make_constant_weight(
                static_cast<size_t>(config.head_dim),
                1.0f
            ),
            device
        )
    );

    weights.add(
        "model.layers.0.self_attn.k_norm.weight",
        make_float_tensor(
            {config.head_dim},
            make_constant_weight(
                static_cast<size_t>(config.head_dim),
                1.0f
            ),
            device
        )
    );

    /*
     * MLP:
     *
     * gate_proj: [intermediate_size, hidden_size] = [8, 4]
     * up_proj:   [intermediate_size, hidden_size] = [8, 4]
     * down_proj: [hidden_size, intermediate_size] = [4, 8]
     */
    weights.add(
        "model.layers.0.mlp.gate_proj.weight",
        make_float_tensor(
            {config.intermediate_size, config.hidden_size},
            make_sequential_weight(
                static_cast<size_t>(
                    config.intermediate_size *
                    config.hidden_size
                ),
                0.01f,
                0.0005f
            ),
            device
        )
    );

    weights.add(
        "model.layers.0.mlp.up_proj.weight",
        make_float_tensor(
            {config.intermediate_size, config.hidden_size},
            make_sequential_weight(
                static_cast<size_t>(
                    config.intermediate_size *
                    config.hidden_size
                ),
                0.015f,
                0.0005f
            ),
            device
        )
    );

    weights.add(
        "model.layers.0.mlp.down_proj.weight",
        make_float_tensor(
            {config.hidden_size, config.intermediate_size},
            make_sequential_weight(
                static_cast<size_t>(
                    config.hidden_size *
                    config.intermediate_size
                ),
                0.02f,
                0.0005f
            ),
            device
        )
    );

    /*
     * final norm: [hidden_size]
     */
    weights.add(
        "model.norm.weight",
        make_float_tensor(
            {config.hidden_size},
            make_constant_weight(
                static_cast<size_t>(config.hidden_size),
                1.0f
            ),
            device
        )
    );

    /*
     * lm_head: [vocab_size, hidden_size]
     */
    weights.add(
        "lm_head.weight",
        make_float_tensor(
            {config.vocab_size, config.hidden_size},
            make_sequential_weight(
                static_cast<size_t>(config.vocab_size * config.hidden_size),
                0.025f,
                0.001f
            ),
            device
        )
    );

    model.load_weights(weights);

    if (!weights.empty()) {
        throw std::runtime_error(
            "make_tiny_model: Qwen3ForCausalLM did not consume all weights"
        );
    }

    if (!model.initialized()) {
        throw std::runtime_error(
            "make_tiny_model: model is not initialized"
        );
    }

    return model;
}

void expect_same_tokens(
    const std::vector<int32_t>& a,
    const std::vector<int32_t>& b
) {
    if (a.size() != b.size()) {
        throw std::runtime_error(
            "generated token size mismatch: " +
            std::to_string(a.size()) +
            " vs " +
            std::to_string(b.size())
        );
    }

    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            throw std::runtime_error(
                "generated token mismatch at index " +
                std::to_string(i) +
                ": kv_cache=" +
                std::to_string(a[i]) +
                ", paged_kv_cache=" +
                std::to_string(b[i])
            );
        }
    }
}

void print_tokens(
    const char* name,
    const std::vector<int32_t>& tokens
) {
    std::cerr << name << ":";

    for (int32_t token : tokens) {
        std::cerr << " " << token;
    }

    std::cerr << std::endl;
}

void run_generate_compare_test(
    Device device,
    const std::vector<int32_t>& input_ids,
    int64_t max_new_tokens
) {
    const char* device_name =
        device == Device::CUDA ? "cuda" : "cpu";

    std::cerr << "[test_generate_paged_kv_cache] device="
              << device_name
              << ", prompt_len="
              << input_ids.size()
              << ", max_new_tokens="
              << max_new_tokens
              << std::endl;

    Qwen3ForCausalLM model = make_tiny_model(device);

    GreedyGenerateOptions options;
    options.max_new_tokens = max_new_tokens;
    options.eos_token_id = -1;
    options.device = device;
    options.verbose = false;

    const std::vector<int32_t> ordinary_output =
        generate_greedy_with_kv_cache(
            model,
            input_ids,
            options
        );

    const std::vector<int32_t> paged_output =
        generate_greedy_with_paged_kv_cache(
            model,
            input_ids,
            options
        );

    print_tokens("ordinary", ordinary_output);
    print_tokens("paged   ", paged_output);

    expect_same_tokens(
        ordinary_output,
        paged_output
    );

    const size_t expected_max_size =
        input_ids.size() + static_cast<size_t>(max_new_tokens);

    if (ordinary_output.size() > expected_max_size) {
        throw std::runtime_error(
            "ordinary_output generated too many tokens"
        );
    }

    if (paged_output.size() > expected_max_size) {
        throw std::runtime_error(
            "paged_output generated too many tokens"
        );
    }
}

void test_generate_max_new_tokens_zero() {
    Device device = Device::CPU;

    Qwen3ForCausalLM model = make_tiny_model(device);

    std::vector<int32_t> input_ids = {1, 2, 3};

    GreedyGenerateOptions options;
    options.max_new_tokens = 0;
    options.eos_token_id = -1;
    options.device = device;
    options.verbose = false;

    const std::vector<int32_t> ordinary_output =
        generate_greedy_with_kv_cache(
            model,
            input_ids,
            options
        );

    const std::vector<int32_t> paged_output =
        generate_greedy_with_paged_kv_cache(
            model,
            input_ids,
            options
        );

    expect_same_tokens(
        ordinary_output,
        input_ids
    );

    expect_same_tokens(
        paged_output,
        input_ids
    );
}

}  // namespace

int main() {
    try {
        test_generate_max_new_tokens_zero();

        run_generate_compare_test(
            Device::CPU,
            std::vector<int32_t>{1, 2, 3},
            4
        );

        run_generate_compare_test(
            Device::CPU,
            std::vector<int32_t>{4},
            3
        );

        run_generate_compare_test(
            Device::CUDA,
            std::vector<int32_t>{1, 2, 3},
            4
        );

        run_generate_compare_test(
            Device::CUDA,
            std::vector<int32_t>{4},
            3
        );

    } catch (const std::exception& e) {
        std::cerr << "[test_generate_paged_kv_cache] failed: "
                  << e.what()
                  << std::endl;
        return 1;
    }

    std::cout << "test_generate_paged_kv_cache passed"
              << std::endl;

    return 0;
}
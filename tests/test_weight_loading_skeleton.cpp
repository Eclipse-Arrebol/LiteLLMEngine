#include "core/device.hpp"
#include "core/dtype.hpp"
#include "core/tensor.hpp"
#include "model/model_config.hpp"
#include "model/qwen3.hpp"
#include "weights/weight_map.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error("Test failed: " + message);
    }
}

lite_llm::ModelConfig make_test_config() {
    lite_llm::ModelConfig config;

    config.model_type = "qwen3";

    config.vocab_size = 100;
    config.hidden_size = 16;
    config.intermediate_size = 32;
    config.num_hidden_layers = 2;
    config.num_attention_heads = 4;
    config.num_key_value_heads = 2;
    config.head_dim = 4;
    config.max_position_embeddings = 128;

    config.rms_norm_eps = 1e-6f;
    config.rope_theta = 1000000.0f;

    config.tie_word_embeddings = false;

    return config;
}

lite_llm::Tensor make_weight(std::vector<int64_t> shape) {
    lite_llm::Tensor tensor(
        std::move(shape),
        lite_llm::DType::FP32,
        lite_llm::Device::CPU
    );

    std::vector<float> data(tensor.numel(), 0.1f);

    tensor.copy_from_cpu(
        data.data(),
        data.size() * sizeof(float)
    );

    return tensor;
}

void add_layer_weights(
    lite_llm::WeightMap& weights,
    int layer_id,
    const lite_llm::ModelConfig& config
) {
    std::string prefix = "model.layers." + std::to_string(layer_id);

    int64_t hidden_size = config.hidden_size;
    int64_t intermediate_size = config.intermediate_size;
    int64_t q_out = config.num_attention_heads * config.head_dim;
    int64_t kv_out = config.num_key_value_heads * config.head_dim;

    weights.add(
        prefix + ".input_layernorm.weight",
        make_weight({hidden_size})
    );

    weights.add(
        prefix + ".self_attn.q_proj.weight",
        make_weight({q_out, hidden_size})
    );

    weights.add(
        prefix + ".self_attn.k_proj.weight",
        make_weight({kv_out, hidden_size})
    );

    weights.add(
        prefix + ".self_attn.v_proj.weight",
        make_weight({kv_out, hidden_size})
    );

    weights.add(
        prefix + ".self_attn.o_proj.weight",
        make_weight({hidden_size, q_out})
    );

    weights.add(
        prefix + ".post_attention_layernorm.weight",
        make_weight({hidden_size})
    );

    weights.add(
        prefix + ".mlp.gate_proj.weight",
        make_weight({intermediate_size, hidden_size})
    );

    weights.add(
        prefix + ".mlp.up_proj.weight",
        make_weight({intermediate_size, hidden_size})
    );

    weights.add(
        prefix + ".mlp.down_proj.weight",
        make_weight({hidden_size, intermediate_size})
    );
}

lite_llm::WeightMap make_fake_qwen3_weights(const lite_llm::ModelConfig& config) {
    lite_llm::WeightMap weights;

    weights.add(
        "model.embed_tokens.weight",
        make_weight({config.vocab_size, config.hidden_size})
    );

    for (int i = 0; i < config.num_hidden_layers; ++i) {
        add_layer_weights(weights, i, config);
    }

    weights.add(
        "model.norm.weight",
        make_weight({config.hidden_size})
    );

    weights.add(
        "lm_head.weight",
        make_weight({config.vocab_size, config.hidden_size})
    );

    return weights;
}

} // namespace

int main() {
    try {
        std::cout << "[test_weight_loading_skeleton] build model..." << std::endl;

        lite_llm::ModelConfig config = make_test_config();
        lite_llm::Qwen3ForCausalLM model(config);

        expect(!model.initialized(), "Model should not be initialized before load_weights");

        lite_llm::WeightMap weights = make_fake_qwen3_weights(config);

        expect(!weights.empty(), "Fake weight map should not be empty");

        model.load_weights(weights);

        expect(model.initialized(), "Model should be initialized after load_weights");
        expect(weights.empty(), "All weights should be consumed after load_weights");

        std::cout << "[test_weight_loading_skeleton] passed" << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
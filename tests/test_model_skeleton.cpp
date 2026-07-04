#include "model/model_config.hpp"
#include "model/qwen3.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

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

} // namespace

int main() {
    try {
        std::cout << "[test_model_skeleton] build Qwen3 skeleton..." << std::endl;

        lite_llm::ModelConfig config = make_test_config();

        lite_llm::Qwen3MLP mlp(config);

        expect(std::string(mlp.name()) == "Qwen3MLP", "MLP name mismatch");
        expect(mlp.hidden_size() == 16, "MLP hidden_size mismatch");
        expect(mlp.intermediate_size() == 32, "MLP intermediate_size mismatch");
        expect(!mlp.initialized(), "MLP should not be initialized before weights are loaded");

        lite_llm::Qwen3Attention attention(config);

        expect(std::string(attention.name()) == "Qwen3Attention", "Attention name mismatch");
        expect(attention.hidden_size() == 16, "Attention hidden_size mismatch");
        expect(attention.num_attention_heads() == 4, "Attention num_attention_heads mismatch");
        expect(attention.num_key_value_heads() == 2, "Attention num_key_value_heads mismatch");
        expect(attention.head_dim() == 4, "Attention head_dim mismatch");
        expect(!attention.initialized(), "Attention should not be initialized before weights are loaded");

        lite_llm::Qwen3DecoderLayer layer(config);

        expect(std::string(layer.name()) == "Qwen3DecoderLayer", "DecoderLayer name mismatch");
        expect(!layer.initialized(), "DecoderLayer should not be initialized before weights are loaded");

        lite_llm::Qwen3Model model(config);

        expect(std::string(model.name()) == "Qwen3Model", "Model name mismatch");
        expect(model.num_layers() == 2, "Model layer count mismatch");
        expect(model.config().hidden_size == 16, "Model config hidden_size mismatch");
        expect(model.config().num_hidden_layers == 2, "Model config layer mismatch");
        expect(!model.initialized(), "Model should not be initialized before weights are loaded");

        lite_llm::Qwen3ForCausalLM causal_lm(config);

        expect(std::string(causal_lm.name()) == "Qwen3ForCausalLM", "CausalLM name mismatch");
        expect(causal_lm.config().vocab_size == 100, "CausalLM vocab_size mismatch");
        expect(causal_lm.model().num_layers() == 2, "CausalLM model layer count mismatch");
        expect(!causal_lm.initialized(), "CausalLM should not be initialized before weights are loaded");

        std::cout << "[test_model_skeleton] passed" << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
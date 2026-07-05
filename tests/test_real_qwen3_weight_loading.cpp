#include "model/model_config.hpp"
#include "model/qwen3.hpp"
#include "weights/weight_loader.hpp"
#include "weights/weight_map.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

/*
想要换dir就要重新设计环境变量，默认是/root/rivermind-data/Qwen_Qwen3-0.6B
LITELLM_TEST_QWEN3_MODEL_DIR
*/
namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error("Test failed: " + message);
    }
}

std::string get_env_or_default(const char* name, const std::string& default_value) {
    const char* value = std::getenv(name);

    if (value == nullptr || std::string(value).empty()) {
        return default_value;
    }

    return std::string(value);
}

} // namespace

int main() {
    try {
        std::string model_dir = get_env_or_default(
            "LITELLM_TEST_QWEN3_MODEL_DIR",
            "/root/rivermind-data/Qwen_Qwen3-0.6B"
        );

        std::filesystem::path model_path(model_dir);
        std::filesystem::path config_path = model_path / "config.json";
        std::filesystem::path weights_dir = model_path / "converted_weights";
        std::filesystem::path index_path = weights_dir / "weights_index.json";

        if (!std::filesystem::exists(config_path)) {
            std::cout
                << "[test_real_qwen3_weight_loading] skipped: config not found: "
                << config_path
                << std::endl;
            return 0;
        }

        if (!std::filesystem::exists(index_path)) {
            std::cout
                << "[test_real_qwen3_weight_loading] skipped: weights index not found: "
                << index_path
                << std::endl;
            return 0;
        }

        std::cout << "[test_real_qwen3_weight_loading] model_dir: "
                  << model_path
                  << std::endl;

        std::cout << "[test_real_qwen3_weight_loading] load config..." << std::endl;

        lite_llm::ModelConfig config =
            lite_llm::load_model_config(config_path.string());

        std::cout << "[test_real_qwen3_weight_loading] load weight map..." << std::endl;

        lite_llm::WeightLoaderOptions options;
        options.device = lite_llm::Device::CPU;

        lite_llm::WeightMap weights =
            lite_llm::load_weight_map_from_directory(
                weights_dir.string(),
                options
            );

        std::cout << "[test_real_qwen3_weight_loading] weights count: "
                  << weights.size()
                  << std::endl;

        expect(weights.size() == 311, "Qwen3-0.6B converted weight count should be 311");

        expect(
            weights.contains("model.layers.0.input_layernorm.weight"),
            "missing model.layers.0.input_layernorm.weight"
        );

        expect(
            weights.contains("model.layers.0.post_attention_layernorm.weight"),
            "missing model.layers.0.post_attention_layernorm.weight"
        );

        expect(
            weights.contains("model.layers.0.self_attn.q_norm.weight"),
            "missing model.layers.0.self_attn.q_norm.weight"
        );

        expect(
            weights.contains("model.layers.0.self_attn.k_norm.weight"),
            "missing model.layers.0.self_attn.k_norm.weight"
        );

        std::cout << "[test_real_qwen3_weight_loading] build model..." << std::endl;

        lite_llm::Qwen3ForCausalLM model(config);

        expect(!model.initialized(), "model should not be initialized before load_weights");

        std::cout << "[test_real_qwen3_weight_loading] inject weights..." << std::endl;

        model.load_weights(weights);

        expect(model.initialized(), "model should be initialized after load_weights");
        expect(weights.empty(), "all real Qwen3 weights should be consumed");

        std::cout << "[test_real_qwen3_weight_loading] passed" << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
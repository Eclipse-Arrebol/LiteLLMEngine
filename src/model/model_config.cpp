#include "model/model_config.hpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <iostream>
#include <stdexcept>

namespace lite_llm {

namespace {

template <typename T>
T get_required(const nlohmann::json& json, const std::string& key) {
    if (!json.contains(key)) {
        throw std::runtime_error("Missing required config field: " + key);
    }

    return json.at(key).get<T>();
}

template <typename T>
T get_optional(const nlohmann::json& json, const std::string& key, T default_value) {
    if (!json.contains(key)) {
        return default_value;
    }

    return json.at(key).get<T>();
}

} // namespace

ModelConfig load_model_config(const std::string& config_path) {
    std::ifstream file(config_path);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open config file: " + config_path);
    }

    nlohmann::json json;
    file >> json;

    ModelConfig config;

    config.model_type = get_required<std::string>(json, "model_type");

    config.vocab_size = get_required<int>(json, "vocab_size");
    config.hidden_size = get_required<int>(json, "hidden_size");
    config.intermediate_size = get_required<int>(json, "intermediate_size");
    config.num_hidden_layers = get_required<int>(json, "num_hidden_layers");
    config.num_attention_heads = get_required<int>(json, "num_attention_heads");

    config.num_key_value_heads = get_optional<int>(
        json,
        "num_key_value_heads",
        config.num_attention_heads
    );

    config.head_dim = get_optional<int>(
        json,
        "head_dim",
        config.hidden_size / config.num_attention_heads
    );

    config.max_position_embeddings = get_optional<int>(
        json,
        "max_position_embeddings",
        32768
    );

    config.rms_norm_eps = get_optional<float>(
        json,
        "rms_norm_eps",
        1e-6f
    );

    config.rope_theta = get_optional<float>(
        json,
        "rope_theta",
        10000.0f
    );

    config.tie_word_embeddings = get_optional<bool>(
        json,
        "tie_word_embeddings",
        false
    );

    return config;
}

void print_model_config(const ModelConfig& config) {
    std::cout << "Model config:\n";
    std::cout << "  Model type:            " << config.model_type << "\n";
    std::cout << "  Vocab size:            " << config.vocab_size << "\n";
    std::cout << "  Hidden size:           " << config.hidden_size << "\n";
    std::cout << "  Intermediate size:     " << config.intermediate_size << "\n";
    std::cout << "  Num layers:            " << config.num_hidden_layers << "\n";
    std::cout << "  Num attention heads:   " << config.num_attention_heads << "\n";
    std::cout << "  Num KV heads:          " << config.num_key_value_heads << "\n";
    std::cout << "  Head dim:              " << config.head_dim << "\n";
    std::cout << "  Max position:          " << config.max_position_embeddings << "\n";
    std::cout << "  RMS norm eps:          " << config.rms_norm_eps << "\n";
    std::cout << "  RoPE theta:            " << config.rope_theta << "\n";
    std::cout << "  Tie word embeddings:   " << std::boolalpha
              << config.tie_word_embeddings << "\n";
}

} // namespace lite_llm
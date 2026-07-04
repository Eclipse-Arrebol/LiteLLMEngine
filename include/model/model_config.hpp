#pragma once

#include <string>

namespace lite_llm {

struct ModelConfig {
    std::string model_type;

    int vocab_size = 0;
    int hidden_size = 0;
    int intermediate_size = 0;
    int num_hidden_layers = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;

    int head_dim = 0;
    int max_position_embeddings = 0;

    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000.0f;

    bool tie_word_embeddings = false;
};

ModelConfig load_model_config(const std::string& config_path);

void print_model_config(const ModelConfig& config);

} // namespace lite_llm
#include "model/qwen3.hpp"

#include <stdexcept>

namespace lite_llm {

namespace {

void check_qwen3_config(const ModelConfig& config) {
    if (config.vocab_size <= 0) {
        throw std::runtime_error("Qwen3 config vocab_size must be positive");
    }

    if (config.hidden_size <= 0) {
        throw std::runtime_error("Qwen3 config hidden_size must be positive");
    }

    if (config.intermediate_size <= 0) {
        throw std::runtime_error("Qwen3 config intermediate_size must be positive");
    }

    if (config.num_hidden_layers <= 0) {
        throw std::runtime_error("Qwen3 config num_hidden_layers must be positive");
    }

    if (config.num_attention_heads <= 0) {
        throw std::runtime_error("Qwen3 config num_attention_heads must be positive");
    }

    if (config.num_key_value_heads <= 0) {
        throw std::runtime_error("Qwen3 config num_key_value_heads must be positive");
    }

    if (config.head_dim <= 0) {
        throw std::runtime_error("Qwen3 config head_dim must be positive");
    }

    if (config.hidden_size != config.num_attention_heads * config.head_dim) {
        throw std::runtime_error(
            "Qwen3 config hidden_size must equal num_attention_heads * head_dim"
        );
    }

    if (config.num_attention_heads % config.num_key_value_heads != 0) {
        throw std::runtime_error(
            "Qwen3 config num_attention_heads must be divisible by num_key_value_heads"
        );
    }

    if (config.rms_norm_eps <= 0.0f) {
        throw std::runtime_error("Qwen3 config rms_norm_eps must be positive");
    }

    if (config.rope_theta <= 0.0f) {
        throw std::runtime_error("Qwen3 config rope_theta must be positive");
    }
}

} // namespace

Qwen3MLP::Qwen3MLP(const ModelConfig& config)
    : hidden_size_(config.hidden_size),
      intermediate_size_(config.intermediate_size) {
    if (hidden_size_ <= 0 || intermediate_size_ <= 0) {
        throw std::runtime_error("Qwen3MLP got invalid config");
    }
}

void Qwen3MLP::forward(const Tensor& hidden_states, Tensor& output) const {
    if (!initialized()) {
        throw std::runtime_error("Qwen3MLP::forward called before weights are initialized");
    }

    (void)hidden_states;
    (void)output;

    throw std::runtime_error("Qwen3MLP::forward not implemented yet");
}

Qwen3Attention::Qwen3Attention(const ModelConfig& config)
    : q_proj_(),
      k_proj_(),
      v_proj_(),
      o_proj_(),
      rotary_(config.head_dim, config.rope_theta),
      hidden_size_(config.hidden_size),
      num_attention_heads_(config.num_attention_heads),
      num_key_value_heads_(config.num_key_value_heads),
      head_dim_(config.head_dim) {
    if (hidden_size_ <= 0 ||
        num_attention_heads_ <= 0 ||
        num_key_value_heads_ <= 0 ||
        head_dim_ <= 0) {
        throw std::runtime_error("Qwen3Attention got invalid config");
    }

    if (hidden_size_ != num_attention_heads_ * head_dim_) {
        throw std::runtime_error("Qwen3Attention hidden_size mismatch");
    }

    if (num_attention_heads_ % num_key_value_heads_ != 0) {
        throw std::runtime_error("Qwen3Attention invalid kv head config");
    }
}

void Qwen3Attention::forward(
    const Tensor& hidden_states,
    const ForwardContext& context,
    Tensor& output
) const {
    if (!initialized()) {
        throw std::runtime_error("Qwen3Attention::forward called before weights are initialized");
    }

    (void)hidden_states;
    (void)context;
    (void)output;

    throw std::runtime_error("Qwen3Attention::forward not implemented yet");
}

Qwen3DecoderLayer::Qwen3DecoderLayer(const ModelConfig& config)
    : self_attn_(config),
      mlp_(config),
      input_layernorm_(config.rms_norm_eps),
      post_attention_layernorm_(config.rms_norm_eps) {
}

void Qwen3DecoderLayer::forward(
    const Tensor& hidden_states,
    const ForwardContext& context,
    Tensor& output
) const {
    if (!initialized()) {
        throw std::runtime_error("Qwen3DecoderLayer::forward called before weights are initialized");
    }

    (void)hidden_states;
    (void)context;
    (void)output;

    throw std::runtime_error("Qwen3DecoderLayer::forward not implemented yet");
}

Qwen3Model::Qwen3Model(const ModelConfig& config)
    : config_(config),
      embed_tokens_(),
      norm_(config.rms_norm_eps) {
    check_qwen3_config(config_);

    layers_.reserve(static_cast<size_t>(config_.num_hidden_layers));

    for (int i = 0; i < config_.num_hidden_layers; ++i) {
        layers_.emplace_back(config_);
    }
}

bool Qwen3Model::initialized() const {
    if (!embed_tokens_.initialized()) {
        return false;
    }

    if (!norm_.initialized()) {
        return false;
    }

    for (const auto& layer : layers_) {
        if (!layer.initialized()) {
            return false;
        }
    }

    return true;
}

void Qwen3Model::forward(
    const Tensor& input_ids,
    const ForwardContext& context,
    Tensor& hidden_states
) const {
    if (!initialized()) {
        throw std::runtime_error("Qwen3Model::forward called before weights are initialized");
    }

    (void)input_ids;
    (void)context;
    (void)hidden_states;

    throw std::runtime_error("Qwen3Model::forward not implemented yet");
}

Qwen3ForCausalLM::Qwen3ForCausalLM(const ModelConfig& config)
    : config_(config),
      model_(config),
      lm_head_() {
    check_qwen3_config(config_);
}

void Qwen3ForCausalLM::forward(
    const Tensor& input_ids,
    const ForwardContext& context,
    Tensor& logits
) const {
    if (!initialized()) {
        throw std::runtime_error("Qwen3ForCausalLM::forward called before weights are initialized");
    }

    (void)input_ids;
    (void)context;
    (void)logits;

    throw std::runtime_error("Qwen3ForCausalLM::forward not implemented yet");
}

} // namespace lite_llm
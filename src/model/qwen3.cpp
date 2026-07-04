#include "model/qwen3.hpp"

#include <stdexcept>
#include <string>

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

void Qwen3MLP::load_weights(WeightMap& weights, const std::string& prefix) {
    gate_proj_.load_weight(weights.take(prefix + ".gate_proj.weight"));
    up_proj_.load_weight(weights.take(prefix + ".up_proj.weight"));
    down_proj_.load_weight(weights.take(prefix + ".down_proj.weight"));

    if (gate_proj_.in_features() != hidden_size_ ||
        gate_proj_.out_features() != intermediate_size_) {
        throw std::runtime_error("Qwen3MLP gate_proj weight shape mismatch");
    }

    if (up_proj_.in_features() != hidden_size_ ||
        up_proj_.out_features() != intermediate_size_) {
        throw std::runtime_error("Qwen3MLP up_proj weight shape mismatch");
    }

    if (down_proj_.in_features() != intermediate_size_ ||
        down_proj_.out_features() != hidden_size_) {
        throw std::runtime_error("Qwen3MLP down_proj weight shape mismatch");
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

void Qwen3Attention::load_weights(WeightMap& weights, const std::string& prefix) {
    q_proj_.load_weight(weights.take(prefix + ".q_proj.weight"));
    k_proj_.load_weight(weights.take(prefix + ".k_proj.weight"));
    v_proj_.load_weight(weights.take(prefix + ".v_proj.weight"));
    o_proj_.load_weight(weights.take(prefix + ".o_proj.weight"));

    int64_t q_out = num_attention_heads_ * head_dim_;
    int64_t kv_out = num_key_value_heads_ * head_dim_;

    if (q_proj_.in_features() != hidden_size_ ||
        q_proj_.out_features() != q_out) {
        throw std::runtime_error("Qwen3Attention q_proj weight shape mismatch");
    }

    if (k_proj_.in_features() != hidden_size_ ||
        k_proj_.out_features() != kv_out) {
        throw std::runtime_error("Qwen3Attention k_proj weight shape mismatch");
    }

    if (v_proj_.in_features() != hidden_size_ ||
        v_proj_.out_features() != kv_out) {
        throw std::runtime_error("Qwen3Attention v_proj weight shape mismatch");
    }

    if (o_proj_.in_features() != q_out ||
        o_proj_.out_features() != hidden_size_) {
        throw std::runtime_error("Qwen3Attention o_proj weight shape mismatch");
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

void Qwen3DecoderLayer::load_weights(WeightMap& weights, const std::string& prefix) {
    input_layernorm_.load_weight(
        weights.take(prefix + ".input_layernorm.weight")
    );

    self_attn_.load_weights(
        weights,
        prefix + ".self_attn"
    );

    post_attention_layernorm_.load_weight(
        weights.take(prefix + ".post_attention_layernorm.weight")
    );

    mlp_.load_weights(
        weights,
        prefix + ".mlp"
    );
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

void Qwen3Model::load_weights(WeightMap& weights) {
    embed_tokens_.load_weight(
        weights.take("model.embed_tokens.weight")
    );

    for (int i = 0; i < config_.num_hidden_layers; ++i) {
        layers_[static_cast<size_t>(i)].load_weights(
            weights,
            "model.layers." + std::to_string(i)
        );
    }

    norm_.load_weight(
        weights.take("model.norm.weight")
    );

    if (embed_tokens_.vocab_size() != config_.vocab_size ||
        embed_tokens_.hidden_size() != config_.hidden_size) {
        throw std::runtime_error("Qwen3Model embed_tokens weight shape mismatch");
    }

    if (norm_.hidden_size() != config_.hidden_size) {
        throw std::runtime_error("Qwen3Model norm weight shape mismatch");
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

void Qwen3ForCausalLM::load_weights(WeightMap& weights) {
    model_.load_weights(weights);

    if (config_.tie_word_embeddings) {
        throw std::runtime_error("tie_word_embeddings=true is not supported yet");
    }

    lm_head_.load_weight(
        weights.take("lm_head.weight")
    );

    if (lm_head_.in_features() != config_.hidden_size ||
        lm_head_.out_features() != config_.vocab_size) {
        throw std::runtime_error("Qwen3ForCausalLM lm_head weight shape mismatch");
    }
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
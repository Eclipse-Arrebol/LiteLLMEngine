#include "model/qwen3.hpp"
#include "ops/silu_and_mul.hpp"
#include "ops/copy.hpp"
#include "ops/attention.hpp"

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

    int64_t q_proj_out = static_cast<int64_t>(config.num_attention_heads) *
                     static_cast<int64_t>(config.head_dim);

    int64_t kv_proj_out = static_cast<int64_t>(config.num_key_value_heads) *
                        static_cast<int64_t>(config.head_dim);

    if (q_proj_out <= 0) {
        throw std::runtime_error("Qwen3 config q_proj_out must be positive");
    }

    if (kv_proj_out <= 0) {
        throw std::runtime_error("Qwen3 config kv_proj_out must be positive");
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

void Qwen3MLP::forward(const Tensor& input, Tensor& output) const {
    if (!initialized()) {
        throw std::runtime_error("Qwen3MLP is not initialized");
    }

    if (input.dtype() != DType::FP32) {
        throw std::runtime_error("Qwen3MLP input must be FP32");
    }

    if (output.dtype() != DType::FP32) {
        throw std::runtime_error("Qwen3MLP output must be FP32");
    }

    if (input.shape().size() != 2) {
        throw std::runtime_error("Qwen3MLP input must be 2D: [num_tokens, hidden_size]");
    }

    if (output.shape().size() != 2) {
        throw std::runtime_error("Qwen3MLP output must be 2D: [num_tokens, hidden_size]");
    }

    const int64_t num_tokens = input.shape()[0];

    if (input.shape()[1] != hidden_size_) {
        throw std::runtime_error("Qwen3MLP input hidden_size mismatch");
    }

    if (output.shape()[0] != num_tokens || output.shape()[1] != hidden_size_) {
        throw std::runtime_error("Qwen3MLP output shape mismatch");
    }

    if (input.device() != output.device()) {
        throw std::runtime_error("Qwen3MLP input and output must be on same device");
    }

    Tensor gate({num_tokens, intermediate_size_}, DType::FP32, input.device());
    Tensor up({num_tokens, intermediate_size_}, DType::FP32, input.device());
    Tensor act({num_tokens, intermediate_size_}, DType::FP32, input.device());

    gate_proj_.forward(input, gate);
    up_proj_.forward(input, up);

    silu_and_mul(gate, up, act);

    down_proj_.forward(act, output);
}

Qwen3Attention::Qwen3Attention(const ModelConfig& config)
    : q_proj_(),
      k_proj_(),
      v_proj_(),
      o_proj_(),
      q_norm_(config.rms_norm_eps),
      k_norm_(config.rms_norm_eps),
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


    if (num_attention_heads_ % num_key_value_heads_ != 0) {
        throw std::runtime_error("Qwen3Attention invalid kv head config");
    }
}

void Qwen3Attention::load_weights(WeightMap& weights, const std::string& prefix) {
    q_proj_.load_weight(weights.take(prefix + ".q_proj.weight"));
    k_proj_.load_weight(weights.take(prefix + ".k_proj.weight"));
    v_proj_.load_weight(weights.take(prefix + ".v_proj.weight"));
    o_proj_.load_weight(weights.take(prefix + ".o_proj.weight"));

    q_norm_.load_weight(weights.take(prefix + ".q_norm.weight"));
    k_norm_.load_weight(weights.take(prefix + ".k_norm.weight"));

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

    if (q_norm_.hidden_size() != head_dim_) {
        throw std::runtime_error("Qwen3Attention q_norm weight shape mismatch");
    }

    if (k_norm_.hidden_size() != head_dim_) {
        throw std::runtime_error("Qwen3Attention k_norm weight shape mismatch");
    }
}

void Qwen3Attention::forward(
    const Tensor& hidden_states,
    const ForwardContext& context,
    Tensor& output
) const {
    if (!initialized()) {
        throw std::runtime_error(
            "Qwen3Attention::forward called before weights are initialized"
        );
    }

    if (context.position_ids == nullptr) {
        throw std::runtime_error("Qwen3Attention ForwardContext.position_ids is null");
    }

    if (context.use_cache) {
        throw std::runtime_error("Qwen3Attention KV cache is not supported yet");
    }

    if (context.past_len != 0) {
        throw std::runtime_error("Qwen3Attention past_len is not supported yet");
    }

    const Tensor& position_ids = *context.position_ids;

    if (hidden_states.dtype() != DType::FP32) {
        throw std::runtime_error("Qwen3Attention hidden_states must be FP32");
    }

    if (position_ids.dtype() != DType::INT32) {
        throw std::runtime_error("Qwen3Attention position_ids must be INT32");
    }

    if (output.dtype() != DType::FP32) {
        throw std::runtime_error("Qwen3Attention output must be FP32");
    }

    if (hidden_states.shape().size() != 2) {
        throw std::runtime_error(
            "Qwen3Attention hidden_states must be 2D: [num_tokens, hidden_size]"
        );
    }

    if (position_ids.shape().size() != 1) {
        throw std::runtime_error(
            "Qwen3Attention position_ids must be 1D: [num_tokens]"
        );
    }

    if (output.shape().size() != 2) {
        throw std::runtime_error(
            "Qwen3Attention output must be 2D: [num_tokens, hidden_size]"
        );
    }

    const int64_t num_tokens = hidden_states.shape()[0];

    if (context.seq_len != 0 && context.seq_len != num_tokens) {
        throw std::runtime_error("Qwen3Attention context.seq_len mismatch");
    }

    if (hidden_states.shape()[1] != hidden_size_) {
        throw std::runtime_error("Qwen3Attention hidden_size mismatch");
    }

    if (position_ids.shape()[0] != num_tokens) {
        throw std::runtime_error("Qwen3Attention position_ids shape mismatch");
    }

    if (output.shape()[0] != num_tokens || output.shape()[1] != hidden_size_) {
        throw std::runtime_error("Qwen3Attention output shape mismatch");
    }

    if (hidden_states.device() != position_ids.device() ||
        hidden_states.device() != output.device()) {
        throw std::runtime_error(
            "Qwen3Attention hidden_states, position_ids and output must be on same device"
        );
    }

    const Device device = hidden_states.device();

    const int64_t q_size = num_attention_heads_ * head_dim_;
    const int64_t kv_size = num_key_value_heads_ * head_dim_;

    Tensor q_flat({num_tokens, q_size}, DType::FP32, device);
    Tensor k_flat({num_tokens, kv_size}, DType::FP32, device);
    Tensor v_flat({num_tokens, kv_size}, DType::FP32, device);

    q_proj_.forward(hidden_states, q_flat);
    k_proj_.forward(hidden_states, k_flat);
    v_proj_.forward(hidden_states, v_flat);

    Tensor q_2d(
        {num_tokens * num_attention_heads_, head_dim_},
        DType::FP32,
        device
    );

    Tensor k_2d(
        {num_tokens * num_key_value_heads_, head_dim_},
        DType::FP32,
        device
    );

    tensor_copy(q_flat, q_2d);
    tensor_copy(k_flat, k_2d);

    Tensor q_normed_2d(
        {num_tokens * num_attention_heads_, head_dim_},
        DType::FP32,
        device
    );

    Tensor k_normed_2d(
        {num_tokens * num_key_value_heads_, head_dim_},
        DType::FP32,
        device
    );

    q_norm_.forward(q_2d, q_normed_2d);
    k_norm_.forward(k_2d, k_normed_2d);

    Tensor q_3d(
        {num_tokens, num_attention_heads_, head_dim_},
        DType::FP32,
        device
    );

    Tensor k_3d(
        {num_tokens, num_key_value_heads_, head_dim_},
        DType::FP32,
        device
    );

    Tensor v_3d(
        {num_tokens, num_key_value_heads_, head_dim_},
        DType::FP32,
        device
    );

    tensor_copy(q_normed_2d, q_3d);
    tensor_copy(k_normed_2d, k_3d);
    tensor_copy(v_flat, v_3d);

    Tensor q_rot(
        {num_tokens, num_attention_heads_, head_dim_},
        DType::FP32,
        device
    );

    Tensor k_rot(
        {num_tokens, num_key_value_heads_, head_dim_},
        DType::FP32,
        device
    );

    rotary_.apply(q_3d, k_3d, position_ids, q_rot, k_rot);

    Tensor attn_out_3d(
        {num_tokens, num_attention_heads_, head_dim_},
        DType::FP32,
        device
    );

    flash_attention(q_rot, k_rot, v_3d, attn_out_3d);

    Tensor attn_out_flat(
        {num_tokens, q_size},
        DType::FP32,
        device
    );

    tensor_copy(attn_out_3d, attn_out_flat);

    o_proj_.forward(attn_out_flat, output);
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

    if (weights.contains("lm_head.weight")) {
        lm_head_.load_weight(
            weights.take("lm_head.weight")
        );
    } else if (config_.tie_word_embeddings) {
        throw std::runtime_error(
            "tie_word_embeddings=true without lm_head.weight is not supported yet"
        );
    } else {
        throw std::runtime_error("Weight not found: lm_head.weight");
    }

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
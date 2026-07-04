#pragma once

#include "core/tensor.hpp"
#include "layers/base.hpp"
#include "layers/embedding.hpp"
#include "layers/linear.hpp"
#include "layers/rms_norm.hpp"
#include "layers/rotary.hpp"
#include "model/model_config.hpp"
#include "runtime/forward_context.hpp"
#include "weights/weight_map.hpp"

#include <cstdint>
#include <vector>
#include <string>

namespace lite_llm {

class Qwen3MLP : public Layer {
public:
    Qwen3MLP() = default;

    explicit Qwen3MLP(const ModelConfig& config);

    Qwen3MLP(const Qwen3MLP&) = delete;
    Qwen3MLP& operator=(const Qwen3MLP&) = delete;

    Qwen3MLP(Qwen3MLP&&) noexcept = default;
    Qwen3MLP& operator=(Qwen3MLP&&) noexcept = default;

    const char* name() const override {
        return "Qwen3MLP";
    }

    bool initialized() const override {
        return gate_proj_.initialized() &&
               up_proj_.initialized() &&
               down_proj_.initialized();
    }

    void forward(const Tensor& hidden_states, Tensor& output) const;

    void load_weights(WeightMap& weights, const std::string& prefix);

    int64_t hidden_size() const {
        return hidden_size_;
    }

    int64_t intermediate_size() const {
        return intermediate_size_;
    }

private:
    Linear gate_proj_;
    Linear up_proj_;
    Linear down_proj_;

    int64_t hidden_size_ = 0;
    int64_t intermediate_size_ = 0;
};

class Qwen3Attention : public Layer {
public:
    Qwen3Attention() = default;

    explicit Qwen3Attention(const ModelConfig& config);

    Qwen3Attention(const Qwen3Attention&) = delete;
    Qwen3Attention& operator=(const Qwen3Attention&) = delete;

    Qwen3Attention(Qwen3Attention&&) noexcept = default;
    Qwen3Attention& operator=(Qwen3Attention&&) noexcept = default;

    const char* name() const override {
        return "Qwen3Attention";
    }

    bool initialized() const override {
        return q_proj_.initialized() &&
               k_proj_.initialized() &&
               v_proj_.initialized() &&
               o_proj_.initialized();
    }

    void forward(
        const Tensor& hidden_states,
        const ForwardContext& context,
        Tensor& output
    ) const;

    void load_weights(WeightMap& weights, const std::string& prefix);

    int64_t hidden_size() const {
        return hidden_size_;
    }

    int64_t num_attention_heads() const {
        return num_attention_heads_;
    }

    int64_t num_key_value_heads() const {
        return num_key_value_heads_;
    }

    int64_t head_dim() const {
        return head_dim_;
    }

private:
    Linear q_proj_;
    Linear k_proj_;
    Linear v_proj_;
    Linear o_proj_;

    RotaryEmbedding rotary_;

    int64_t hidden_size_ = 0;
    int64_t num_attention_heads_ = 0;
    int64_t num_key_value_heads_ = 0;
    int64_t head_dim_ = 0;
};

class Qwen3DecoderLayer : public Layer {
public:
    Qwen3DecoderLayer() = default;

    explicit Qwen3DecoderLayer(const ModelConfig& config);

    Qwen3DecoderLayer(const Qwen3DecoderLayer&) = delete;
    Qwen3DecoderLayer& operator=(const Qwen3DecoderLayer&) = delete;

    Qwen3DecoderLayer(Qwen3DecoderLayer&&) noexcept = default;
    Qwen3DecoderLayer& operator=(Qwen3DecoderLayer&&) noexcept = default;

    const char* name() const override {
        return "Qwen3DecoderLayer";
    }

    bool initialized() const override {
        return self_attn_.initialized() &&
               mlp_.initialized() &&
               input_layernorm_.initialized() &&
               post_attention_layernorm_.initialized();
    }

    void forward(
        const Tensor& hidden_states,
        const ForwardContext& context,
        Tensor& output
    ) const;

    void load_weights(WeightMap& weights, const std::string& prefix);

private:
    Qwen3Attention self_attn_;
    Qwen3MLP mlp_;

    RMSNorm input_layernorm_;
    RMSNorm post_attention_layernorm_;
};

class Qwen3Model : public Layer {
public:
    Qwen3Model() = default;

    explicit Qwen3Model(const ModelConfig& config);

    Qwen3Model(const Qwen3Model&) = delete;
    Qwen3Model& operator=(const Qwen3Model&) = delete;

    Qwen3Model(Qwen3Model&&) noexcept = default;
    Qwen3Model& operator=(Qwen3Model&&) noexcept = default;

    const char* name() const override {
        return "Qwen3Model";
    }

    bool initialized() const override;

    void forward(
        const Tensor& input_ids,
        const ForwardContext& context,
        Tensor& hidden_states
    ) const;

    void load_weights(WeightMap& weights);

    int64_t num_layers() const {
        return static_cast<int64_t>(layers_.size());
    }

    const ModelConfig& config() const {
        return config_;
    }

private:
    ModelConfig config_;

    Embedding embed_tokens_;
    std::vector<Qwen3DecoderLayer> layers_;
    RMSNorm norm_;
};

class Qwen3ForCausalLM : public Layer {
public:
    Qwen3ForCausalLM() = default;

    explicit Qwen3ForCausalLM(const ModelConfig& config);

    Qwen3ForCausalLM(const Qwen3ForCausalLM&) = delete;
    Qwen3ForCausalLM& operator=(const Qwen3ForCausalLM&) = delete;

    Qwen3ForCausalLM(Qwen3ForCausalLM&&) noexcept = default;
    Qwen3ForCausalLM& operator=(Qwen3ForCausalLM&&) noexcept = default;

    const char* name() const override {
        return "Qwen3ForCausalLM";
    }

    bool initialized() const override {
        return model_.initialized() && lm_head_.initialized();
    }

    void forward(
        const Tensor& input_ids,
        const ForwardContext& context,
        Tensor& logits
    ) const;

    void load_weights(WeightMap& weights);
    
    const ModelConfig& config() const {
        return config_;
    }

    const Qwen3Model& model() const {
        return model_;
    }

private:
    ModelConfig config_;

    Qwen3Model model_;
    Linear lm_head_;
};

} // namespace lite_llm
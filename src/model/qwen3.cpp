#include "model/qwen3.hpp"
#include "ops/silu_and_mul.hpp"
#include "ops/copy.hpp"
#include "ops/attention.hpp"
#include "ops/add.hpp"
#include "ops/paged_attention.hpp"
#include "engine/kv_cache.hpp"
#include "engine/paged_kv_cache.hpp"


#include <stdexcept>
#include <string>
#include <iostream>

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
        throw std::runtime_error(
            "Qwen3Attention ForwardContext.position_ids is null"
        );
    }

    if (context.use_paged_kv_cache && !context.use_cache) {
        throw std::runtime_error(
            "Qwen3Attention use_paged_kv_cache=true requires use_cache=true"
        );
    }

    if (context.use_cache) {
        if (context.layer_idx < 0) {
            throw std::runtime_error(
                "Qwen3Attention use_cache=true but layer_idx is invalid"
            );
        }

        if (context.use_paged_kv_cache) {
            if (context.paged_kv_cache == nullptr) {
                throw std::runtime_error(
                    "Qwen3Attention use_paged_kv_cache=true but paged_kv_cache is null"
                );
            }

            if (context.block_table_manager == nullptr) {
                throw std::runtime_error(
                    "Qwen3Attention use_paged_kv_cache=true but block_table_manager is null"
                );
            }

            if (context.table_idx < 0) {
                throw std::runtime_error(
                    "Qwen3Attention use_paged_kv_cache=true but table_idx is invalid"
                );
            }
        } else {
            if (context.kv_cache == nullptr) {
                throw std::runtime_error(
                    "Qwen3Attention use_cache=true but kv_cache is null"
                );
            }
        }
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

    if (context.past_len != 0) {
        if (!context.use_cache) {
            throw std::runtime_error(
                "Qwen3Attention past_len > 0 requires use_cache=true"
            );
        }

        if (num_tokens != 1) {
            throw std::runtime_error(
                "Qwen3Attention decode with past_len > 0 only supports num_tokens=1"
            );
        }
    }

    if (context.seq_len != 0 && context.seq_len != num_tokens) {
        throw std::runtime_error("Qwen3Attention context.seq_len mismatch");
    }

    if (hidden_states.shape()[1] != hidden_size_) {
        throw std::runtime_error("Qwen3Attention hidden_size mismatch");
    }

    if (position_ids.shape()[0] != num_tokens) {
        throw std::runtime_error(
            "Qwen3Attention position_ids shape mismatch"
        );
    }

    if (output.shape()[0] != num_tokens ||
        output.shape()[1] != hidden_size_) {
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

    Tensor q_flat(
        {num_tokens, q_size},
        DType::FP32,
        device
    );

    Tensor k_flat(
        {num_tokens, kv_size},
        DType::FP32,
        device
    );

    Tensor v_flat(
        {num_tokens, kv_size},
        DType::FP32,
        device
    );

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

    rotary_.apply(
        q_3d,
        k_3d,
        position_ids,
        q_rot,
        k_rot
    );

    /*
     * Cache write:
     *
     * prefill:
     *   past_len = 0
     *   num_tokens = prompt_len
     *   write logical token [0, prompt_len)
     *
     * decode:
     *   past_len = cached token count
     *   num_tokens = 1
     *   write logical token past_len
     */
    if (context.use_cache) {
        if (context.use_paged_kv_cache) {
            context.paged_kv_cache->update_layer(
                context.layer_idx,
                *context.block_table_manager,
                context.table_idx,
                context.past_len,
                k_rot,
                v_3d
            );
        } else {
            context.kv_cache->update_layer(
                context.layer_idx,
                k_rot,
                v_3d
            );
        }
    }

    Tensor attn_out_3d(
        {num_tokens, num_attention_heads_, head_dim_},
        DType::FP32,
        device
    );

    /*
     * Attention:
     *
     * prefill:
     *   use current q/k/v directly.
     *
     * decode:
     *   use q_rot + cached K/V.
     */
    if (context.use_cache && context.past_len > 0) {
        static int printed = 0;
        if (printed < 10) {
            std::cerr << "[attention] use kv cache decode"
                      << ", layer_idx=" << context.layer_idx
                      << ", past_len=" << context.past_len
                      << ", num_tokens=" << num_tokens
                      << ", paged=" << context.use_paged_kv_cache
                      << std::endl;
            ++printed;
        }

        const int64_t kv_seq_len = context.past_len + num_tokens;

        if (context.use_paged_kv_cache) {
            if (device == Device::CUDA) {
                flash_attention_paged_kv_cache_cuda(
                    q_rot,
                    *context.paged_kv_cache,
                    *context.block_table_manager,
                    context.table_idx,
                    context.layer_idx,
                    kv_seq_len,
                    attn_out_3d
                );
            } else if (device == Device::CPU) {
                paged_attention_decode_cpu(
                    q_rot,
                    *context.paged_kv_cache,
                    *context.block_table_manager,
                    context.table_idx,
                    context.layer_idx,
                    kv_seq_len,
                    attn_out_3d
                );
            } else {
                throw std::runtime_error(
                    "Qwen3Attention unsupported device for paged kv cache decode"
                );
            }
        } else {
            const LayerKVCache& layer_cache =
                context.kv_cache->layer(context.layer_idx);

            flash_attention_kv_cache(
                q_rot,
                layer_cache.key,
                layer_cache.value,
                kv_seq_len,
                attn_out_3d
            );
        }
    } else {
        flash_attention(
            q_rot,
            k_rot,
            v_3d,
            attn_out_3d
        );
    }

    Tensor attn_out_flat(
        {num_tokens, q_size},
        DType::FP32,
        device
    );

    tensor_copy(attn_out_3d, attn_out_flat);

    o_proj_.forward(attn_out_flat, output);
}

void Qwen3Attention::forward_decode_batch(
    const Tensor& hidden_states,
    const BatchDecodeForwardContext& context,
    Tensor& output
) const {
    if (!initialized()) {
        throw std::runtime_error(
            "Qwen3Attention::forward_decode_batch called before weights are initialized"
        );
    }

    if (context.position_ids == nullptr) {
        throw std::runtime_error(
            "Qwen3Attention batch decode position_ids is null"
        );
    }

    if (!context.use_paged_kv_cache ||
        context.paged_kv_cache == nullptr ||
        context.block_table_manager == nullptr ||
        context.table_indices == nullptr ||
        context.past_lens == nullptr ||
        context.kv_seq_lens == nullptr) {
        throw std::runtime_error(
            "Qwen3Attention batch decode requires paged KV cache metadata"
        );
    }

    if (context.layer_idx < 0) {
        throw std::runtime_error(
            "Qwen3Attention batch decode layer_idx is invalid"
        );
    }

    const Tensor& position_ids = *context.position_ids;

    if (hidden_states.dtype() != DType::FP32 ||
        output.dtype() != DType::FP32) {
        throw std::runtime_error(
            "Qwen3Attention batch decode tensors must be FP32"
        );
    }

    if (position_ids.dtype() != DType::INT32) {
        throw std::runtime_error(
            "Qwen3Attention batch decode position_ids must be INT32"
        );
    }

    if (hidden_states.shape().size() != 2 ||
        output.shape().size() != 2) {
        throw std::runtime_error(
            "Qwen3Attention batch decode hidden/output must be 2D"
        );
    }

    if (position_ids.shape().size() != 1) {
        throw std::runtime_error(
            "Qwen3Attention batch decode position_ids must be 1D"
        );
    }

    const int64_t batch_size = hidden_states.shape()[0];

    if (context.batch_size != batch_size) {
        throw std::runtime_error(
            "Qwen3Attention batch decode batch_size mismatch"
        );
    }

    if (batch_size <= 0) {
        throw std::runtime_error(
            "Qwen3Attention batch decode batch_size must be positive"
        );
    }

    if (hidden_states.shape()[1] != hidden_size_ ||
        output.shape()[0] != batch_size ||
        output.shape()[1] != hidden_size_) {
        throw std::runtime_error(
            "Qwen3Attention batch decode hidden/output shape mismatch"
        );
    }

    if (position_ids.shape()[0] != batch_size) {
        throw std::runtime_error(
            "Qwen3Attention batch decode position_ids shape mismatch"
        );
    }

    if (context.table_indices->size() != static_cast<size_t>(batch_size) ||
        context.past_lens->size() != static_cast<size_t>(batch_size) ||
        context.kv_seq_lens->size() != static_cast<size_t>(batch_size)) {
        throw std::runtime_error(
            "Qwen3Attention batch decode metadata size mismatch"
        );
    }

    for (int64_t i = 0; i < batch_size; ++i) {
        const int64_t past_len =
            (*context.past_lens)[static_cast<size_t>(i)];
        const int64_t kv_seq_len =
            (*context.kv_seq_lens)[static_cast<size_t>(i)];

        if (past_len < 0 || kv_seq_len != past_len + 1) {
            throw std::runtime_error(
                "Qwen3Attention batch decode invalid kv_seq_len"
            );
        }
    }

    if (hidden_states.device() != position_ids.device() ||
        hidden_states.device() != output.device()) {
        throw std::runtime_error(
            "Qwen3Attention batch decode tensors must be on same device"
        );
    }

    const Device device = hidden_states.device();

    const int64_t q_size = num_attention_heads_ * head_dim_;
    const int64_t kv_size = num_key_value_heads_ * head_dim_;

    Tensor q_flat({batch_size, q_size}, DType::FP32, device);
    Tensor k_flat({batch_size, kv_size}, DType::FP32, device);
    Tensor v_flat({batch_size, kv_size}, DType::FP32, device);

    q_proj_.forward(hidden_states, q_flat);
    k_proj_.forward(hidden_states, k_flat);
    v_proj_.forward(hidden_states, v_flat);

    Tensor q_2d(
        {batch_size * num_attention_heads_, head_dim_},
        DType::FP32,
        device
    );

    Tensor k_2d(
        {batch_size * num_key_value_heads_, head_dim_},
        DType::FP32,
        device
    );

    tensor_copy(q_flat, q_2d);
    tensor_copy(k_flat, k_2d);

    Tensor q_normed_2d(
        {batch_size * num_attention_heads_, head_dim_},
        DType::FP32,
        device
    );

    Tensor k_normed_2d(
        {batch_size * num_key_value_heads_, head_dim_},
        DType::FP32,
        device
    );

    q_norm_.forward(q_2d, q_normed_2d);
    k_norm_.forward(k_2d, k_normed_2d);

    Tensor q_3d(
        {batch_size, num_attention_heads_, head_dim_},
        DType::FP32,
        device
    );

    Tensor k_3d(
        {batch_size, num_key_value_heads_, head_dim_},
        DType::FP32,
        device
    );

    Tensor v_3d(
        {batch_size, num_key_value_heads_, head_dim_},
        DType::FP32,
        device
    );

    tensor_copy(q_normed_2d, q_3d);
    tensor_copy(k_normed_2d, k_3d);
    tensor_copy(v_flat, v_3d);

    Tensor q_rot(
        {batch_size, num_attention_heads_, head_dim_},
        DType::FP32,
        device
    );

    Tensor k_rot(
        {batch_size, num_key_value_heads_, head_dim_},
        DType::FP32,
        device
    );

    rotary_.apply(
        q_3d,
        k_3d,
        position_ids,
        q_rot,
        k_rot
    );

    const size_t kv_row_bytes =
        static_cast<size_t>(num_key_value_heads_ * head_dim_) *
        sizeof(float);

    for (int64_t row = 0; row < batch_size; ++row) {
        Tensor k_row(
            {1, num_key_value_heads_, head_dim_},
            DType::FP32,
            device
        );

        Tensor v_row(
            {1, num_key_value_heads_, head_dim_},
            DType::FP32,
            device
        );

        k_row.copy_from_tensor(
            k_rot,
            0,
            static_cast<size_t>(row) * kv_row_bytes,
            kv_row_bytes
        );

        v_row.copy_from_tensor(
            v_3d,
            0,
            static_cast<size_t>(row) * kv_row_bytes,
            kv_row_bytes
        );

        context.paged_kv_cache->update_layer(
            context.layer_idx,
            *context.block_table_manager,
            (*context.table_indices)[static_cast<size_t>(row)],
            (*context.past_lens)[static_cast<size_t>(row)],
            k_row,
            v_row
        );
    }

    Tensor attn_out_3d(
        {batch_size, num_attention_heads_, head_dim_},
        DType::FP32,
        device
    );

    paged_attention_decode_batch_gather(
        q_rot,
        *context.paged_kv_cache,
        *context.block_table_manager,
        *context.table_indices,
        context.layer_idx,
        *context.kv_seq_lens,
        attn_out_3d
    );

    Tensor attn_out_flat(
        {batch_size, q_size},
        DType::FP32,
        device
    );

    tensor_copy(attn_out_3d, attn_out_flat);

    o_proj_.forward(attn_out_flat, output);
}

Qwen3DecoderLayer::Qwen3DecoderLayer(
    const ModelConfig& config
)
    : Qwen3DecoderLayer(config, -1) {
}

Qwen3DecoderLayer::Qwen3DecoderLayer(const ModelConfig& config,int64_t layer_idx)
    : layer_idx_(layer_idx),
      hidden_size_(config.hidden_size),
      self_attn_(config),
      mlp_(config),
      input_layernorm_(config.rms_norm_eps),
      post_attention_layernorm_(config.rms_norm_eps) {
    if (hidden_size_ <= 0) {
        throw std::runtime_error("Qwen3DecoderLayer hidden_size must be positive");
    }
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
        throw std::runtime_error(
            "Qwen3DecoderLayer::forward called before weights are initialized"
        );
    }

    if (context.use_cache && layer_idx_ < 0) {
        throw std::runtime_error(
            "Qwen3DecoderLayer use_cache=true but layer_idx is invalid"
        );
    }

    if (hidden_states.dtype() != DType::FP32) {
        throw std::runtime_error("Qwen3DecoderLayer hidden_states must be FP32");
    }

    if (output.dtype() != DType::FP32) {
        throw std::runtime_error("Qwen3DecoderLayer output must be FP32");
    }

    if (hidden_states.shape().size() != 2) {
        throw std::runtime_error(
            "Qwen3DecoderLayer hidden_states must be 2D: [num_tokens, hidden_size]"
        );
    }

    if (output.shape() != hidden_states.shape()) {
        throw std::runtime_error("Qwen3DecoderLayer output shape mismatch");
    }

    if (hidden_states.shape()[1] != hidden_size_) {
        throw std::runtime_error("Qwen3DecoderLayer hidden_size mismatch");
    }

    if (hidden_states.device() != output.device()) {
        throw std::runtime_error(
            "Qwen3DecoderLayer hidden_states and output must be on same device"
        );
    }

    const int64_t num_tokens = hidden_states.shape()[0];
    const Device device = hidden_states.device();

    ForwardContext layer_context = context;
    layer_context.layer_idx = layer_idx_;

    Tensor normed_1(
        {num_tokens, hidden_size_},
        DType::FP32,
        device
    );

    Tensor attn_out(
        {num_tokens, hidden_size_},
        DType::FP32,
        device
    );

    Tensor residual_after_attn(
        {num_tokens, hidden_size_},
        DType::FP32,
        device
    );

    input_layernorm_.forward(hidden_states, normed_1);
    self_attn_.forward(normed_1, layer_context, attn_out);

    tensor_add(hidden_states, attn_out, residual_after_attn);

    Tensor normed_2(
        {num_tokens, hidden_size_},
        DType::FP32,
        device
    );

    Tensor mlp_out(
        {num_tokens, hidden_size_},
        DType::FP32,
        device
    );

    post_attention_layernorm_.forward(residual_after_attn, normed_2);
    mlp_.forward(normed_2, mlp_out);

    tensor_add(residual_after_attn, mlp_out, output);
}

void Qwen3DecoderLayer::forward_decode_batch(
    const Tensor& hidden_states,
    const BatchDecodeForwardContext& context,
    Tensor& output
) const {
    if (!initialized()) {
        throw std::runtime_error(
            "Qwen3DecoderLayer::forward_decode_batch called before weights are initialized"
        );
    }

    if (layer_idx_ < 0) {
        throw std::runtime_error(
            "Qwen3DecoderLayer batch decode layer_idx is invalid"
        );
    }

    if (hidden_states.dtype() != DType::FP32 ||
        output.dtype() != DType::FP32) {
        throw std::runtime_error(
            "Qwen3DecoderLayer batch decode tensors must be FP32"
        );
    }

    if (hidden_states.shape().size() != 2 ||
        output.shape() != hidden_states.shape()) {
        throw std::runtime_error(
            "Qwen3DecoderLayer batch decode hidden/output shape mismatch"
        );
    }

    if (hidden_states.shape()[1] != hidden_size_) {
        throw std::runtime_error(
            "Qwen3DecoderLayer batch decode hidden_size mismatch"
        );
    }

    if (hidden_states.device() != output.device()) {
        throw std::runtime_error(
            "Qwen3DecoderLayer batch decode tensors must be on same device"
        );
    }

    const int64_t batch_size = hidden_states.shape()[0];
    const Device device = hidden_states.device();

    BatchDecodeForwardContext layer_context = context;
    layer_context.layer_idx = layer_idx_;

    Tensor normed_1(
        {batch_size, hidden_size_},
        DType::FP32,
        device
    );

    Tensor attn_out(
        {batch_size, hidden_size_},
        DType::FP32,
        device
    );

    Tensor residual_after_attn(
        {batch_size, hidden_size_},
        DType::FP32,
        device
    );

    input_layernorm_.forward(hidden_states, normed_1);
    self_attn_.forward_decode_batch(normed_1, layer_context, attn_out);

    tensor_add(hidden_states, attn_out, residual_after_attn);

    Tensor normed_2(
        {batch_size, hidden_size_},
        DType::FP32,
        device
    );

    Tensor mlp_out(
        {batch_size, hidden_size_},
        DType::FP32,
        device
    );

    post_attention_layernorm_.forward(residual_after_attn, normed_2);
    mlp_.forward(normed_2, mlp_out);

    tensor_add(residual_after_attn, mlp_out, output);
}

Qwen3Model::Qwen3Model(const ModelConfig& config)
    : config_(config),
      embed_tokens_(),
      norm_(config.rms_norm_eps) {
    check_qwen3_config(config_);

    layers_.reserve(static_cast<size_t>(config_.num_hidden_layers));

    for (int i = 0; i < config_.num_hidden_layers; ++i) {
        layers_.emplace_back(config_,i);
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
        throw std::runtime_error(
            "Qwen3Model::forward called before weights are initialized"
        );
    }

    if (context.position_ids == nullptr) {
        throw std::runtime_error(
            "Qwen3Model ForwardContext.position_ids is null"
        );
    }

    if (context.use_paged_kv_cache && !context.use_cache) {
        throw std::runtime_error(
            "Qwen3Model use_paged_kv_cache=true requires use_cache=true"
        );
    }

    if (context.use_cache) {
        if (context.use_paged_kv_cache) {
            if (context.paged_kv_cache == nullptr) {
                throw std::runtime_error(
                    "Qwen3Model use_paged_kv_cache=true but paged_kv_cache is null"
                );
            }

            if (context.block_table_manager == nullptr) {
                throw std::runtime_error(
                    "Qwen3Model use_paged_kv_cache=true but block_table_manager is null"
                );
            }

            if (context.table_idx < 0) {
                throw std::runtime_error(
                    "Qwen3Model use_paged_kv_cache=true but table_idx is invalid"
                );
            }
        } else {
            if (context.kv_cache == nullptr) {
                throw std::runtime_error(
                    "Qwen3Model use_cache=true but kv_cache is null"
                );
            }
        }
    }

    if (input_ids.dtype() != DType::INT32) {
        throw std::runtime_error("Qwen3Model input_ids must be INT32");
    }

    if (hidden_states.dtype() != DType::FP32) {
        throw std::runtime_error("Qwen3Model hidden_states must be FP32");
    }

    if (input_ids.shape().size() != 1) {
        throw std::runtime_error(
            "Qwen3Model input_ids must be 1D: [num_tokens]"
        );
    }

    if (hidden_states.shape().size() != 2) {
        throw std::runtime_error(
            "Qwen3Model hidden_states must be 2D: [num_tokens, hidden_size]"
        );
    }

    const Tensor& position_ids = *context.position_ids;

    if (position_ids.dtype() != DType::INT32) {
        throw std::runtime_error("Qwen3Model position_ids must be INT32");
    }

    if (position_ids.shape().size() != 1) {
        throw std::runtime_error(
            "Qwen3Model position_ids must be 1D: [num_tokens]"
        );
    }

    const int64_t num_tokens = input_ids.shape()[0];
    const int64_t hidden_size = config_.hidden_size;

    if (context.seq_len != 0 && context.seq_len != num_tokens) {
        throw std::runtime_error("Qwen3Model context.seq_len mismatch");
    }

    if (position_ids.shape()[0] != num_tokens) {
        throw std::runtime_error(
            "Qwen3Model position_ids shape mismatch"
        );
    }

    if (hidden_states.shape()[0] != num_tokens ||
        hidden_states.shape()[1] != hidden_size) {
        throw std::runtime_error(
            "Qwen3Model hidden_states shape mismatch"
        );
    }

    if (input_ids.device() != position_ids.device() ||
        input_ids.device() != hidden_states.device()) {
        throw std::runtime_error(
            "Qwen3Model input_ids, position_ids and hidden_states must be on same device"
        );
    }

    const Device device = input_ids.device();

    Tensor hidden_a(
        {num_tokens, hidden_size},
        DType::FP32,
        device
    );

    Tensor hidden_b(
        {num_tokens, hidden_size},
        DType::FP32,
        device
    );

    embed_tokens_.forward(input_ids, hidden_a);

    const Tensor* current = &hidden_a;
    Tensor* next = &hidden_b;

    for (size_t i = 0; i < layers_.size(); ++i) {
        ForwardContext layer_context = context;
        layer_context.layer_idx = static_cast<int64_t>(i);

        layers_[i].forward(
            *current,
            layer_context,
            *next
        );

        if (current == &hidden_a) {
            current = &hidden_b;
            next = &hidden_a;
        } else {
            current = &hidden_a;
            next = &hidden_b;
        }
    }

    /*
     * 普通 KVCache:
     *   Qwen3Model::forward 内部 advance。
     *
     * PagedKVCache:
     *   外层 PagedKVCacheSession::advance 管理 current_len。
     *   这里不能 advance。
     */
    if (context.use_cache && !context.use_paged_kv_cache) {
        context.kv_cache->advance(num_tokens);
    }

    norm_.forward(*current, hidden_states);
}

void Qwen3Model::forward_decode_batch(
    const Tensor& input_ids,
    const BatchDecodeForwardContext& context,
    Tensor& hidden_states
) const {
    if (!initialized()) {
        throw std::runtime_error(
            "Qwen3Model::forward_decode_batch called before weights are initialized"
        );
    }

    if (context.position_ids == nullptr ||
        !context.use_paged_kv_cache ||
        context.paged_kv_cache == nullptr ||
        context.block_table_manager == nullptr ||
        context.table_indices == nullptr ||
        context.past_lens == nullptr ||
        context.kv_seq_lens == nullptr) {
        throw std::runtime_error(
            "Qwen3Model batch decode requires paged KV cache metadata"
        );
    }

    if (input_ids.dtype() != DType::INT32) {
        throw std::runtime_error(
            "Qwen3Model batch decode input_ids must be INT32"
        );
    }

    if (hidden_states.dtype() != DType::FP32) {
        throw std::runtime_error(
            "Qwen3Model batch decode hidden_states must be FP32"
        );
    }

    if (input_ids.shape().size() != 1) {
        throw std::runtime_error(
            "Qwen3Model batch decode input_ids must be 1D"
        );
    }

    if (hidden_states.shape().size() != 2) {
        throw std::runtime_error(
            "Qwen3Model batch decode hidden_states must be 2D"
        );
    }

    const Tensor& position_ids = *context.position_ids;

    if (position_ids.dtype() != DType::INT32 ||
        position_ids.shape().size() != 1) {
        throw std::runtime_error(
            "Qwen3Model batch decode position_ids must be INT32 1D"
        );
    }

    const int64_t batch_size = input_ids.shape()[0];
    const int64_t hidden_size = config_.hidden_size;

    if (context.batch_size != batch_size ||
        batch_size <= 0) {
        throw std::runtime_error(
            "Qwen3Model batch decode batch_size mismatch"
        );
    }

    if (position_ids.shape()[0] != batch_size ||
        hidden_states.shape()[0] != batch_size ||
        hidden_states.shape()[1] != hidden_size) {
        throw std::runtime_error(
            "Qwen3Model batch decode tensor shape mismatch"
        );
    }

    if (input_ids.device() != position_ids.device() ||
        input_ids.device() != hidden_states.device()) {
        throw std::runtime_error(
            "Qwen3Model batch decode tensors must be on same device"
        );
    }

    const Device device = input_ids.device();

    Tensor hidden_a(
        {batch_size, hidden_size},
        DType::FP32,
        device
    );

    Tensor hidden_b(
        {batch_size, hidden_size},
        DType::FP32,
        device
    );

    embed_tokens_.forward(input_ids, hidden_a);

    const Tensor* current = &hidden_a;
    Tensor* next = &hidden_b;

    for (size_t i = 0; i < layers_.size(); ++i) {
        BatchDecodeForwardContext layer_context = context;
        layer_context.layer_idx = static_cast<int64_t>(i);

        layers_[i].forward_decode_batch(
            *current,
            layer_context,
            *next
        );

        if (current == &hidden_a) {
            current = &hidden_b;
            next = &hidden_a;
        } else {
            current = &hidden_a;
            next = &hidden_b;
        }
    }

    norm_.forward(*current, hidden_states);
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
        throw std::runtime_error(
            "Qwen3ForCausalLM::forward called before weights are initialized"
        );
    }

    if (input_ids.dtype() != DType::INT32) {
        throw std::runtime_error("Qwen3ForCausalLM input_ids must be INT32");
    }

    if (logits.dtype() != DType::FP32) {
        throw std::runtime_error("Qwen3ForCausalLM logits must be FP32");
    }

    if (input_ids.shape().size() != 1) {
        throw std::runtime_error(
            "Qwen3ForCausalLM input_ids must be 1D: [num_tokens]"
        );
    }

    if (logits.shape().size() != 2) {
        throw std::runtime_error(
            "Qwen3ForCausalLM logits must be 2D: [num_tokens, vocab_size]"
        );
    }

    const int64_t num_tokens = input_ids.shape()[0];

    if (context.seq_len != 0 && context.seq_len != num_tokens) {
        throw std::runtime_error("Qwen3ForCausalLM context.seq_len mismatch");
    }

    if (logits.shape()[0] != num_tokens ||
        logits.shape()[1] != config_.vocab_size) {
        throw std::runtime_error("Qwen3ForCausalLM logits shape mismatch");
    }

    if (input_ids.device() != logits.device()) {
        throw std::runtime_error(
            "Qwen3ForCausalLM input_ids and logits must be on same device"
        );
    }

    if (context.position_ids == nullptr) {
        throw std::runtime_error(
            "Qwen3ForCausalLM ForwardContext.position_ids is null"
        );
    }

    if (context.position_ids->device() != input_ids.device()) {
        throw std::runtime_error(
            "Qwen3ForCausalLM position_ids device mismatch"
        );
    }

    const Device device = input_ids.device();

    Tensor hidden_states(
        {num_tokens, config_.hidden_size},
        DType::FP32,
        device
    );

    model_.forward(input_ids, context, hidden_states);

    lm_head_.forward(hidden_states, logits);
}

void Qwen3ForCausalLM::forward_decode_batch(
    const Tensor& input_ids,
    const BatchDecodeForwardContext& context,
    Tensor& logits
) const {
    if (!initialized()) {
        throw std::runtime_error(
            "Qwen3ForCausalLM::forward_decode_batch called before weights are initialized"
        );
    }

    if (input_ids.dtype() != DType::INT32) {
        throw std::runtime_error(
            "Qwen3ForCausalLM batch decode input_ids must be INT32"
        );
    }

    if (logits.dtype() != DType::FP32) {
        throw std::runtime_error(
            "Qwen3ForCausalLM batch decode logits must be FP32"
        );
    }

    if (input_ids.shape().size() != 1 ||
        logits.shape().size() != 2) {
        throw std::runtime_error(
            "Qwen3ForCausalLM batch decode tensor rank mismatch"
        );
    }

    const int64_t batch_size = input_ids.shape()[0];

    if (context.batch_size != batch_size ||
        batch_size <= 0) {
        throw std::runtime_error(
            "Qwen3ForCausalLM batch decode batch_size mismatch"
        );
    }

    if (logits.shape()[0] != batch_size ||
        logits.shape()[1] != config_.vocab_size) {
        throw std::runtime_error(
            "Qwen3ForCausalLM batch decode logits shape mismatch"
        );
    }

    if (context.position_ids == nullptr) {
        throw std::runtime_error(
            "Qwen3ForCausalLM batch decode position_ids is null"
        );
    }

    if (input_ids.device() != logits.device() ||
        context.position_ids->device() != input_ids.device()) {
        throw std::runtime_error(
            "Qwen3ForCausalLM batch decode tensors must be on same device"
        );
    }

    const Device device = input_ids.device();

    Tensor hidden_states(
        {batch_size, config_.hidden_size},
        DType::FP32,
        device
    );

    model_.forward_decode_batch(
        input_ids,
        context,
        hidden_states
    );

    lm_head_.forward(hidden_states, logits);
}

} // namespace lite_llm

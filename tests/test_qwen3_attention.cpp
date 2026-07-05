#include "model/qwen3.hpp"
#include "weights/weight_map.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lite_llm;

namespace {

void check_close(float a, float b, float tol = 1e-4f) {
    if (std::fabs(a - b) > tol) {
        throw std::runtime_error(
            "check_close failed: got " + std::to_string(a) +
            ", expected " + std::to_string(b)
        );
    }
}

int64_t offset3(
    int64_t token,
    int64_t head,
    int64_t dim,
    int64_t num_heads,
    int64_t head_dim
) {
    return (token * num_heads + head) * head_dim + dim;
}

std::vector<float> linear_ref(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    int64_t m,
    int64_t in_features,
    int64_t out_features
) {
    std::vector<float> output(m * out_features, 0.0f);

    for (int64_t row = 0; row < m; ++row) {
        for (int64_t out_col = 0; out_col < out_features; ++out_col) {
            float sum = 0.0f;

            for (int64_t k = 0; k < in_features; ++k) {
                sum += input[row * in_features + k] *
                       weight[out_col * in_features + k];
            }

            output[row * out_features + out_col] = sum;
        }
    }

    return output;
}

std::vector<float> rms_norm_ref(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    int64_t rows,
    int64_t hidden_size,
    float eps
) {
    std::vector<float> output(input.size(), 0.0f);

    for (int64_t row = 0; row < rows; ++row) {
        const int64_t base = row * hidden_size;

        float sum_sq = 0.0f;
        for (int64_t d = 0; d < hidden_size; ++d) {
            const float v = input[base + d];
            sum_sq += v * v;
        }

        const float inv_rms =
            1.0f / std::sqrt(sum_sq / static_cast<float>(hidden_size) + eps);

        for (int64_t d = 0; d < hidden_size; ++d) {
            output[base + d] = input[base + d] * inv_rms * weight[d];
        }
    }

    return output;
}

std::vector<float> rotary_ref(
    const std::vector<float>& input,
    const std::vector<int32_t>& position_ids,
    int64_t num_tokens,
    int64_t num_heads,
    int64_t head_dim,
    float rope_theta
) {
    std::vector<float> output(input.size(), 0.0f);

    const int64_t half_dim = head_dim / 2;

    for (int64_t token = 0; token < num_tokens; ++token) {
        const int32_t pos = position_ids[token];

        for (int64_t head = 0; head < num_heads; ++head) {
            const int64_t base = (token * num_heads + head) * head_dim;

            for (int64_t i = 0; i < half_dim; ++i) {
                const float inv_freq = std::pow(
                    rope_theta,
                    -static_cast<float>(2 * i) / static_cast<float>(head_dim)
                );

                const float angle = static_cast<float>(pos) * inv_freq;
                const float c = std::cos(angle);
                const float s = std::sin(angle);

                const float x1 = input[base + i];
                const float x2 = input[base + i + half_dim];

                output[base + i] = x1 * c - x2 * s;
                output[base + i + half_dim] = x2 * c + x1 * s;
            }
        }
    }

    return output;
}

std::vector<float> attention_ref(
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    int64_t num_tokens,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim
) {
    std::vector<float> output(
        num_tokens * num_q_heads * head_dim,
        0.0f
    );

    const int64_t group_size = num_q_heads / num_kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::vector<float> scores(num_tokens, 0.0f);

    for (int64_t token = 0; token < num_tokens; ++token) {
        for (int64_t q_head = 0; q_head < num_q_heads; ++q_head) {
            const int64_t kv_head = q_head / group_size;

            float max_score = -std::numeric_limits<float>::infinity();

            for (int64_t key_token = 0; key_token <= token; ++key_token) {
                float dot = 0.0f;

                for (int64_t d = 0; d < head_dim; ++d) {
                    dot += q[offset3(token, q_head, d, num_q_heads, head_dim)] *
                           k[offset3(key_token, kv_head, d, num_kv_heads, head_dim)];
                }

                const float score = dot * scale;
                scores[key_token] = score;
                max_score = std::max(max_score, score);
            }

            float sum_exp = 0.0f;

            for (int64_t key_token = 0; key_token <= token; ++key_token) {
                const float e = std::exp(scores[key_token] - max_score);
                scores[key_token] = e;
                sum_exp += e;
            }

            for (int64_t d = 0; d < head_dim; ++d) {
                float out = 0.0f;

                for (int64_t key_token = 0; key_token <= token; ++key_token) {
                    const float prob = scores[key_token] / sum_exp;

                    out += prob *
                           v[offset3(key_token, kv_head, d, num_kv_heads, head_dim)];
                }

                output[offset3(token, q_head, d, num_q_heads, head_dim)] = out;
            }
        }
    }

    return output;
}

std::vector<float> reference_qwen3_attention(
    const std::vector<float>& hidden_states,
    const std::vector<int32_t>& position_ids,
    const std::vector<float>& q_weight,
    const std::vector<float>& k_weight,
    const std::vector<float>& v_weight,
    const std::vector<float>& o_weight,
    const std::vector<float>& q_norm_weight,
    const std::vector<float>& k_norm_weight,
    int64_t num_tokens,
    int64_t hidden_size,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float rms_norm_eps,
    float rope_theta
) {
    const int64_t q_size = num_q_heads * head_dim;
    const int64_t kv_size = num_kv_heads * head_dim;

    std::vector<float> q_flat = linear_ref(
        hidden_states,
        q_weight,
        num_tokens,
        hidden_size,
        q_size
    );

    std::vector<float> k_flat = linear_ref(
        hidden_states,
        k_weight,
        num_tokens,
        hidden_size,
        kv_size
    );

    std::vector<float> v_flat = linear_ref(
        hidden_states,
        v_weight,
        num_tokens,
        hidden_size,
        kv_size
    );

    std::vector<float> q_normed = rms_norm_ref(
        q_flat,
        q_norm_weight,
        num_tokens * num_q_heads,
        head_dim,
        rms_norm_eps
    );

    std::vector<float> k_normed = rms_norm_ref(
        k_flat,
        k_norm_weight,
        num_tokens * num_kv_heads,
        head_dim,
        rms_norm_eps
    );

    std::vector<float> q_rot = rotary_ref(
        q_normed,
        position_ids,
        num_tokens,
        num_q_heads,
        head_dim,
        rope_theta
    );

    std::vector<float> k_rot = rotary_ref(
        k_normed,
        position_ids,
        num_tokens,
        num_kv_heads,
        head_dim,
        rope_theta
    );

    std::vector<float> attn_out = attention_ref(
        q_rot,
        k_rot,
        v_flat,
        num_tokens,
        num_q_heads,
        num_kv_heads,
        head_dim
    );

    return linear_ref(
        attn_out,
        o_weight,
        num_tokens,
        q_size,
        hidden_size
    );
}

Tensor make_float_tensor(
    const std::vector<int64_t>& shape,
    const std::vector<float>& data,
    Device device
) {
    Tensor tensor(shape, DType::FP32, device);
    tensor.copy_from_cpu(data.data(), data.size() * sizeof(float));
    return tensor;
}

Tensor make_int_tensor(
    const std::vector<int64_t>& shape,
    const std::vector<int32_t>& data,
    Device device
) {
    Tensor tensor(shape, DType::INT32, device);
    tensor.copy_from_cpu(data.data(), data.size() * sizeof(int32_t));
    return tensor;
}

ModelConfig make_test_config() {
    ModelConfig config;
    config.model_type = "qwen3";
    config.hidden_size = 3;
    config.num_attention_heads = 2;
    config.num_key_value_heads = 1;
    config.head_dim = 2;
    config.rms_norm_eps = 1e-6f;
    config.rope_theta = 10000.0f;
    return config;
}

WeightMap make_test_weights(
    const std::vector<float>& q_weight,
    const std::vector<float>& k_weight,
    const std::vector<float>& v_weight,
    const std::vector<float>& o_weight,
    const std::vector<float>& q_norm_weight,
    const std::vector<float>& k_norm_weight,
    Device device
) {
    WeightMap weights;

    weights.add(
        "model.layers.0.self_attn.q_proj.weight",
        make_float_tensor({4, 3}, q_weight, device)
    );

    weights.add(
        "model.layers.0.self_attn.k_proj.weight",
        make_float_tensor({2, 3}, k_weight, device)
    );

    weights.add(
        "model.layers.0.self_attn.v_proj.weight",
        make_float_tensor({2, 3}, v_weight, device)
    );

    weights.add(
        "model.layers.0.self_attn.o_proj.weight",
        make_float_tensor({3, 4}, o_weight, device)
    );

    weights.add(
        "model.layers.0.self_attn.q_norm.weight",
        make_float_tensor({2}, q_norm_weight, device)
    );

    weights.add(
        "model.layers.0.self_attn.k_norm.weight",
        make_float_tensor({2}, k_norm_weight, device)
    );

    return weights;
}

void run_qwen3_attention_test(Device device) {
    const std::string device_name = device == Device::CPU ? "cpu" : "cuda";

    std::cout << "[test_qwen3_attention_" << device_name << "] start\n";

    constexpr int64_t num_tokens = 3;
    constexpr int64_t hidden_size = 3;
    constexpr int64_t num_q_heads = 2;
    constexpr int64_t num_kv_heads = 1;
    constexpr int64_t head_dim = 2;
    constexpr float rms_norm_eps = 1e-6f;
    constexpr float rope_theta = 10000.0f;

    std::vector<float> hidden_states_cpu = {
        1.0f, -2.0f, 0.5f,
        0.0f,  1.0f, 2.0f,
       -1.0f,  0.5f, 1.5f,
    };

    std::vector<int32_t> position_ids_cpu = {
        0, 1, 2,
    };

    // q_proj.weight: [num_q_heads * head_dim, hidden_size] = [4, 3]
    std::vector<float> q_weight = {
         0.5f, -1.0f,  0.25f,
         1.0f,  0.0f, -0.5f,
        -0.75f, 0.5f,  1.0f,
         2.0f, -1.0f,  0.0f,
    };

    // k_proj.weight: [num_kv_heads * head_dim, hidden_size] = [2, 3]
    std::vector<float> k_weight = {
         1.0f,  0.5f, -1.0f,
        -0.5f,  1.5f,  0.25f,
    };

    // v_proj.weight: [num_kv_heads * head_dim, hidden_size] = [2, 3]
    std::vector<float> v_weight = {
         0.25f, -1.0f,  2.0f,
         1.5f,   0.0f, -0.5f,
    };

    // o_proj.weight: [hidden_size, num_q_heads * head_dim] = [3, 4]
    std::vector<float> o_weight = {
         1.0f, -0.5f,  0.25f,  2.0f,
        -1.0f,  1.5f,  0.5f,  -0.25f,
         0.0f,  0.75f, -1.0f,  1.0f,
    };

    std::vector<float> q_norm_weight = {
        1.0f, 1.5f,
    };

    std::vector<float> k_norm_weight = {
        0.75f, 1.25f,
    };

    std::vector<float> expected = reference_qwen3_attention(
        hidden_states_cpu,
        position_ids_cpu,
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        q_norm_weight,
        k_norm_weight,
        num_tokens,
        hidden_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rms_norm_eps,
        rope_theta
    );

    ModelConfig config = make_test_config();
    Qwen3Attention attention(config);

    WeightMap weights = make_test_weights(
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        q_norm_weight,
        k_norm_weight,
        device
    );

    attention.load_weights(weights, "model.layers.0.self_attn");

    if (!weights.empty()) {
        throw std::runtime_error("Qwen3Attention did not consume all weights");
    }

    if (!attention.initialized()) {
        throw std::runtime_error("Qwen3Attention should be initialized");
    }

    Tensor hidden_states = make_float_tensor(
        {num_tokens, hidden_size},
        hidden_states_cpu,
        device
    );

    Tensor position_ids = make_int_tensor(
        {num_tokens},
        position_ids_cpu,
        device
    );

    Tensor output(
        {num_tokens, hidden_size},
        DType::FP32,
        device
    );

    output.zero_();

    ForwardContext context;
    context.position_ids = &position_ids;
    context.seq_len = num_tokens;
    context.past_len = 0;
    context.use_cache = false;

    attention.forward(hidden_states, context, output);

    std::vector<float> output_cpu(expected.size(), 0.0f);
    output.copy_to_cpu(output_cpu.data(), output_cpu.size() * sizeof(float));

    for (size_t i = 0; i < expected.size(); ++i) {
        check_close(output_cpu[i], expected[i]);
    }

    std::cout << "[test_qwen3_attention_" << device_name << "] passed\n";
}

void test_qwen3_attention_null_position_ids() {
    std::cout << "[test_qwen3_attention_null_position_ids] start\n";

    ModelConfig config = make_test_config();
    Qwen3Attention attention(config);

    std::vector<float> q_weight(4 * 3, 0.1f);
    std::vector<float> k_weight(2 * 3, 0.1f);
    std::vector<float> v_weight(2 * 3, 0.1f);
    std::vector<float> o_weight(3 * 4, 0.1f);
    std::vector<float> q_norm_weight(2, 1.0f);
    std::vector<float> k_norm_weight(2, 1.0f);

    WeightMap weights = make_test_weights(
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        q_norm_weight,
        k_norm_weight,
        Device::CPU
    );

    attention.load_weights(weights, "model.layers.0.self_attn");

    Tensor hidden_states({1, 3}, DType::FP32, Device::CPU);
    Tensor output({1, 3}, DType::FP32, Device::CPU);

    ForwardContext context;
    context.position_ids = nullptr;
    context.seq_len = 1;

    bool caught = false;

    try {
        attention.forward(hidden_states, context, output);
    } catch (const std::runtime_error&) {
        caught = true;
    }

    if (!caught) {
        throw std::runtime_error("Expected null position_ids error");
    }

    std::cout << "[test_qwen3_attention_null_position_ids] passed\n";
}

void test_qwen3_attention_seq_len_mismatch() {
    std::cout << "[test_qwen3_attention_seq_len_mismatch] start\n";

    ModelConfig config = make_test_config();
    Qwen3Attention attention(config);

    std::vector<float> q_weight(4 * 3, 0.1f);
    std::vector<float> k_weight(2 * 3, 0.1f);
    std::vector<float> v_weight(2 * 3, 0.1f);
    std::vector<float> o_weight(3 * 4, 0.1f);
    std::vector<float> q_norm_weight(2, 1.0f);
    std::vector<float> k_norm_weight(2, 1.0f);

    WeightMap weights = make_test_weights(
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        q_norm_weight,
        k_norm_weight,
        Device::CPU
    );

    attention.load_weights(weights, "model.layers.0.self_attn");

    std::vector<float> hidden_states_cpu = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
    };

    std::vector<int32_t> position_ids_cpu = {
        0, 1,
    };

    Tensor hidden_states = make_float_tensor(
        {2, 3},
        hidden_states_cpu,
        Device::CPU
    );

    Tensor position_ids = make_int_tensor(
        {2},
        position_ids_cpu,
        Device::CPU
    );

    Tensor output({2, 3}, DType::FP32, Device::CPU);

    ForwardContext context;
    context.position_ids = &position_ids;
    context.seq_len = 3;
    context.past_len = 0;
    context.use_cache = false;

    bool caught = false;

    try {
        attention.forward(hidden_states, context, output);
    } catch (const std::runtime_error&) {
        caught = true;
    }

    if (!caught) {
        throw std::runtime_error("Expected seq_len mismatch error");
    }

    std::cout << "[test_qwen3_attention_seq_len_mismatch] passed\n";
}

void test_qwen3_attention_cache_not_supported() {
    std::cout << "[test_qwen3_attention_cache_not_supported] start\n";

    ModelConfig config = make_test_config();
    Qwen3Attention attention(config);

    std::vector<float> q_weight(4 * 3, 0.1f);
    std::vector<float> k_weight(2 * 3, 0.1f);
    std::vector<float> v_weight(2 * 3, 0.1f);
    std::vector<float> o_weight(3 * 4, 0.1f);
    std::vector<float> q_norm_weight(2, 1.0f);
    std::vector<float> k_norm_weight(2, 1.0f);

    WeightMap weights = make_test_weights(
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        q_norm_weight,
        k_norm_weight,
        Device::CPU
    );

    attention.load_weights(weights, "model.layers.0.self_attn");

    std::vector<float> hidden_states_cpu = {
        1.0f, 2.0f, 3.0f,
    };

    std::vector<int32_t> position_ids_cpu = {
        0,
    };

    Tensor hidden_states = make_float_tensor(
        {1, 3},
        hidden_states_cpu,
        Device::CPU
    );

    Tensor position_ids = make_int_tensor(
        {1},
        position_ids_cpu,
        Device::CPU
    );

    Tensor output({1, 3}, DType::FP32, Device::CPU);

    ForwardContext context;
    context.position_ids = &position_ids;
    context.seq_len = 1;
    context.past_len = 0;
    context.use_cache = true;

    bool caught = false;

    try {
        attention.forward(hidden_states, context, output);
    } catch (const std::runtime_error&) {
        caught = true;
    }

    if (!caught) {
        throw std::runtime_error("Expected cache not supported error");
    }

    std::cout << "[test_qwen3_attention_cache_not_supported] passed\n";
}

}  // namespace

int main() {
    try {
        run_qwen3_attention_test(Device::CPU);
        run_qwen3_attention_test(Device::CUDA);

        test_qwen3_attention_null_position_ids();
        test_qwen3_attention_seq_len_mismatch();
        test_qwen3_attention_cache_not_supported();
    } catch (const std::exception& e) {
        std::cerr << "[test_qwen3_attention] failed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "[test_qwen3_attention] all passed\n";
    return 0;
}

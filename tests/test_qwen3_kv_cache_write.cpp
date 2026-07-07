#include "engine/kv_cache.hpp"

#include "core/device.hpp"
#include "core/dtype.hpp"
#include "core/tensor.hpp"

#include "model/qwen3.hpp"

#include "weights/weight_map.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lite_llm;

namespace {

void check_close(float a, float b, float tol = 1e-5f) {
    if (std::fabs(a - b) > tol) {
        throw std::runtime_error(
            "check_close failed: got " + std::to_string(a) +
            ", expected " + std::to_string(b)
        );
    }
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

Tensor make_int32_tensor(
    const std::vector<int64_t>& shape,
    const std::vector<int32_t>& data,
    Device device
) {
    Tensor tensor(shape, DType::INT32, device);
    tensor.copy_from_cpu(data.data(), data.size() * sizeof(int32_t));
    return tensor;
}

std::vector<float> tensor_to_cpu_float(const Tensor& tensor) {
    if (tensor.dtype() != DType::FP32) {
        throw std::runtime_error("tensor_to_cpu_float requires FP32 tensor");
    }

    std::vector<float> data(tensor.numel(), 0.0f);
    tensor.copy_to_cpu(data.data(), data.size() * sizeof(float));
    return data;
}

ModelConfig make_test_config() {
    ModelConfig config;
    config.model_type = "qwen3";

    config.vocab_size = 5;
    config.hidden_size = 3;
    config.intermediate_size = 4;
    config.num_hidden_layers = 1;

    config.num_attention_heads = 2;
    config.num_key_value_heads = 1;
    config.head_dim = 2;

    config.max_position_embeddings = 16;
    config.rms_norm_eps = 1e-6f;
    config.rope_theta = 10000.0f;
    config.tie_word_embeddings = false;

    return config;
}

WeightMap make_test_weights(Device device) {
    WeightMap weights;

    std::vector<float> embed_weight = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f,
        1.0f, 1.0f, 0.0f,
       -1.0f, 0.0f, 0.0f,
    };

    weights.add(
        "model.embed_tokens.weight",
        make_float_tensor({5, 3}, embed_weight, device)
    );

    weights.add(
        "model.layers.0.input_layernorm.weight",
        make_float_tensor({3}, std::vector<float>{1.0f, 1.0f, 1.0f}, device)
    );

    weights.add(
        "model.layers.0.self_attn.q_proj.weight",
        make_float_tensor({4, 3}, std::vector<float>(4 * 3, 0.0f), device)
    );

    weights.add(
        "model.layers.0.self_attn.k_proj.weight",
        make_float_tensor({2, 3}, std::vector<float>(2 * 3, 0.0f), device)
    );

    weights.add(
        "model.layers.0.self_attn.v_proj.weight",
        make_float_tensor({2, 3}, std::vector<float>(2 * 3, 0.0f), device)
    );

    weights.add(
        "model.layers.0.self_attn.o_proj.weight",
        make_float_tensor({3, 4}, std::vector<float>(3 * 4, 0.0f), device)
    );

    weights.add(
        "model.layers.0.self_attn.q_norm.weight",
        make_float_tensor({2}, std::vector<float>{1.0f, 1.0f}, device)
    );

    weights.add(
        "model.layers.0.self_attn.k_norm.weight",
        make_float_tensor({2}, std::vector<float>{1.0f, 1.0f}, device)
    );

    weights.add(
        "model.layers.0.post_attention_layernorm.weight",
        make_float_tensor({3}, std::vector<float>{1.0f, 1.0f, 1.0f}, device)
    );

    weights.add(
        "model.layers.0.mlp.gate_proj.weight",
        make_float_tensor({4, 3}, std::vector<float>(4 * 3, 0.0f), device)
    );

    weights.add(
        "model.layers.0.mlp.up_proj.weight",
        make_float_tensor({4, 3}, std::vector<float>(4 * 3, 0.0f), device)
    );

    weights.add(
        "model.layers.0.mlp.down_proj.weight",
        make_float_tensor({3, 4}, std::vector<float>(3 * 4, 0.0f), device)
    );

    weights.add(
        "model.norm.weight",
        make_float_tensor({3}, std::vector<float>{1.0f, 1.0f, 1.0f}, device)
    );

    std::vector<float> lm_head_weight = {
        0.0f, 0.0f, 0.0f,
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
       -1.0f, -1.0f, -1.0f,
        0.0f, 0.0f, 1.0f,
    };

    weights.add(
        "lm_head.weight",
        make_float_tensor({5, 3}, lm_head_weight, device)
    );

    return weights;
}

void check_logits_equal(
    const Tensor& a,
    const Tensor& b
) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("logits shape mismatch");
    }

    std::vector<float> a_cpu = tensor_to_cpu_float(a);
    std::vector<float> b_cpu = tensor_to_cpu_float(b);

    if (a_cpu.size() != b_cpu.size()) {
        throw std::runtime_error("logits numel mismatch");
    }

    for (size_t i = 0; i < a_cpu.size(); ++i) {
        check_close(a_cpu[i], b_cpu[i]);
    }
}

void run_qwen3_kv_cache_write_test(Device device) {
    const std::string device_name =
        device == Device::CPU ? "cpu" : "cuda";

    std::cout << "[test_qwen3_kv_cache_write_" << device_name << "] start\n";

    ModelConfig config = make_test_config();

    Qwen3ForCausalLM model(config);

    WeightMap weights = make_test_weights(device);
    model.load_weights(weights);

    if (!weights.empty()) {
        throw std::runtime_error("Qwen3ForCausalLM did not consume all weights");
    }

    if (!model.initialized()) {
        throw std::runtime_error("Qwen3ForCausalLM should be initialized");
    }

    std::vector<int32_t> input_ids_cpu = {
        0, 1,
    };

    std::vector<int32_t> position_ids_cpu = {
        0, 1,
    };

    const int64_t seq_len = static_cast<int64_t>(input_ids_cpu.size());

    Tensor input_ids = make_int32_tensor(
        {seq_len},
        input_ids_cpu,
        device
    );

    Tensor position_ids = make_int32_tensor(
        {seq_len},
        position_ids_cpu,
        device
    );

    Tensor logits_no_cache(
        {seq_len, config.vocab_size},
        DType::FP32,
        device
    );

    ForwardContext no_cache_context;
    no_cache_context.position_ids = &position_ids;
    no_cache_context.seq_len = seq_len;
    no_cache_context.past_len = 0;
    no_cache_context.use_cache = false;

    model.forward(
        input_ids,
        no_cache_context,
        logits_no_cache
    );

    ModelKVCache kv_cache(
        config,
        device,
        DType::FP32,
        0
    );

    Tensor logits_with_cache(
        {seq_len, config.vocab_size},
        DType::FP32,
        device
    );

    ForwardContext cache_context;
    cache_context.position_ids = &position_ids;
    cache_context.seq_len = seq_len;
    cache_context.past_len = 0;
    cache_context.use_cache = true;
    cache_context.kv_cache = &kv_cache;

    model.forward(
        input_ids,
        cache_context,
        logits_with_cache
    );

    // 当前阶段只是写 cache，没有改变 attention 计算，所以 logits 应该一致。
    check_logits_equal(logits_no_cache, logits_with_cache);

    // 如果你还没有在 Qwen3Model::forward 末尾调用 kv_cache.advance(seq_len)，
    // 那么 ModelKVCache 的全局 current_len 仍然应该是 0。
    if (kv_cache.current_len() != 0) {
        throw std::runtime_error(
            "kv_cache.current_len should remain 0 before advance is integrated"
        );
    }

    // 但 layer 0 的 cache 应该已经被 Attention::update_layer 写入。
    if (kv_cache.layer(0).current_len != seq_len) {
        throw std::runtime_error(
            "kv_cache.layer(0).current_len should equal seq_len after update_layer"
        );
    }

    if (kv_cache.capacity() < seq_len) {
        throw std::runtime_error("kv_cache capacity should be at least seq_len");
    }

    const std::vector<int64_t> expected_kv_shape = {
        kv_cache.capacity(),
        config.num_key_value_heads,
        config.head_dim
    };

    if (kv_cache.layer(0).key.shape() != expected_kv_shape) {
        throw std::runtime_error("kv_cache layer 0 key shape mismatch");
    }

    if (kv_cache.layer(0).value.shape() != expected_kv_shape) {
        throw std::runtime_error("kv_cache layer 0 value shape mismatch");
    }

    std::cout << "[test_qwen3_kv_cache_write_" << device_name << "] passed\n";
}

}  // namespace

int main() {
    try {
        run_qwen3_kv_cache_write_test(Device::CPU);
        run_qwen3_kv_cache_write_test(Device::CUDA);
    } catch (const std::exception& e) {
        std::cerr << "[test_qwen3_kv_cache_write] failed: "
                  << e.what()
                  << "\n";
        return 1;
    }

    std::cout << "[test_qwen3_kv_cache_write] all passed\n";
    return 0;
}
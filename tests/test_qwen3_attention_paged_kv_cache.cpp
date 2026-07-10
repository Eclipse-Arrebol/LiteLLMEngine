// tests/test_qwen3_attention_paged_kv_cache.cpp

#include "core/tensor.hpp"
#include "engine/block_table_manager.hpp"
#include "engine/kv_cache.hpp"
#include "engine/paged_kv_cache.hpp"
#include "model/qwen3.hpp"
#include "runtime/forward_context.hpp"
#include "weights/weight_map.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lite_llm;

namespace {

void check_close(float a, float b, float tol = 1e-4f) {
    if (std::fabs(a - b) > tol) {
        throw std::runtime_error(
            "check_close failed: got " +
            std::to_string(a) +
            ", expected " +
            std::to_string(b) +
            ", diff=" +
            std::to_string(std::fabs(a - b))
        );
    }
}

void expect_tensor_close(
    const Tensor& a,
    const Tensor& b,
    float tol = 1e-4f
) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("expect_tensor_close shape mismatch");
    }

    if (a.dtype() != DType::FP32 || b.dtype() != DType::FP32) {
        throw std::runtime_error("expect_tensor_close expects FP32 tensors");
    }

    std::vector<float> a_cpu(a.numel());
    std::vector<float> b_cpu(b.numel());

    a.copy_to_cpu(a_cpu.data(), a_cpu.size() * sizeof(float));
    b.copy_to_cpu(b_cpu.data(), b_cpu.size() * sizeof(float));

    for (size_t i = 0; i < a_cpu.size(); ++i) {
        try {
            check_close(a_cpu[i], b_cpu[i], tol);
        } catch (const std::runtime_error& e) {
            throw std::runtime_error(
                "tensor mismatch at index " +
                std::to_string(i) +
                ": " +
                e.what()
            );
        }
    }
}

Tensor make_float_tensor(
    const std::vector<int64_t>& shape,
    const std::vector<float>& data,
    Device device
) {
    Tensor tensor(shape, DType::FP32, device);

    if (tensor.numel() != data.size()) {
        throw std::runtime_error("make_float_tensor data size mismatch");
    }

    tensor.copy_from_cpu(data.data(), data.size() * sizeof(float));
    return tensor;
}

Tensor make_int_tensor(
    const std::vector<int64_t>& shape,
    const std::vector<int32_t>& data,
    Device device
) {
    Tensor tensor(shape, DType::INT32, device);

    if (tensor.numel() != data.size()) {
        throw std::runtime_error("make_int_tensor data size mismatch");
    }

    tensor.copy_from_cpu(data.data(), data.size() * sizeof(int32_t));
    return tensor;
}

ModelConfig make_test_config() {
    ModelConfig config;

    config.model_type = "qwen3";

    config.vocab_size = 0;
    config.hidden_size = 3;
    config.intermediate_size = 0;

    config.num_hidden_layers = 1;
    config.num_attention_heads = 2;
    config.num_key_value_heads = 1;

    config.head_dim = 2;
    config.max_position_embeddings = 1024;

    config.rms_norm_eps = 1e-6f;
    config.rope_theta = 10000.0f;

    config.tie_word_embeddings = false;

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

Qwen3Attention make_initialized_attention(Device device) {
    ModelConfig config = make_test_config();

    Qwen3Attention attention(config);

    // q_proj.weight: [num_q_heads * head_dim, hidden_size] = [4, 3]
    std::vector<float> q_weight = {
        0.5f,  -1.0f, 0.25f,
        1.0f,   0.0f, -0.5f,
       -0.75f,  0.5f, 1.0f,
        2.0f,  -1.0f, 0.0f,
    };

    // k_proj.weight: [num_kv_heads * head_dim, hidden_size] = [2, 3]
    std::vector<float> k_weight = {
        1.0f,  0.5f, -1.0f,
       -0.5f,  1.5f,  0.25f,
    };

    // v_proj.weight: [num_kv_heads * head_dim, hidden_size] = [2, 3]
    std::vector<float> v_weight = {
        0.25f, -1.0f, 2.0f,
        1.5f,   0.0f, -0.5f,
    };

    // o_proj.weight: [hidden_size, num_q_heads * head_dim] = [3, 4]
    std::vector<float> o_weight = {
        1.0f, -0.5f,  0.25f,  2.0f,
       -1.0f,  1.5f,  0.5f,  -0.25f,
        0.0f,  0.75f, -1.0f,  1.0f,
    };

    std::vector<float> q_norm_weight = {
        1.0f,
        1.5f,
    };

    std::vector<float> k_norm_weight = {
        0.75f,
        1.25f,
    };

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
        throw std::runtime_error(
            "Qwen3Attention did not consume all weights"
        );
    }

    if (!attention.initialized()) {
        throw std::runtime_error(
            "Qwen3Attention should be initialized"
        );
    }

    return attention;
}

void run_qwen3_attention_paged_kv_cache_decode_test(Device device) {
    const std::string device_name =
        device == Device::CPU ? "cpu" : "cuda";

    std::cout << "[test_qwen3_attention_paged_kv_cache_"
              << device_name
              << "] start"
              << std::endl;

    constexpr int64_t num_layers = 1;
    constexpr int64_t layer_idx = 0;

    constexpr int64_t hidden_size = 3;
    constexpr int64_t num_q_heads = 2;
    constexpr int64_t num_kv_heads = 1;
    constexpr int64_t head_dim = 2;

    constexpr int64_t prefill_len = 2;
    constexpr int64_t decode_len = 1;
    constexpr int64_t total_kv_len = prefill_len + decode_len;

    constexpr int64_t capacity = 4;
    constexpr int64_t page_size = 2;

    Qwen3Attention attention = make_initialized_attention(device);

    std::vector<float> hidden_prefill_cpu = {
        1.0f, -2.0f, 0.5f,
        0.0f,  1.0f, 2.0f,
    };

    std::vector<float> hidden_decode_cpu = {
       -1.0f, 0.5f, 1.5f,
    };

    std::vector<int32_t> position_prefill_cpu = {
        0,
        1,
    };

    std::vector<int32_t> position_decode_cpu = {
        2,
    };

    Tensor hidden_prefill = make_float_tensor(
        {prefill_len, hidden_size},
        hidden_prefill_cpu,
        device
    );

    Tensor hidden_decode = make_float_tensor(
        {decode_len, hidden_size},
        hidden_decode_cpu,
        device
    );

    Tensor position_prefill = make_int_tensor(
        {prefill_len},
        position_prefill_cpu,
        device
    );

    Tensor position_decode = make_int_tensor(
        {decode_len},
        position_decode_cpu,
        device
    );

    Tensor ordinary_prefill_output(
        {prefill_len, hidden_size},
        DType::FP32,
        device
    );

    Tensor ordinary_decode_output(
        {decode_len, hidden_size},
        DType::FP32,
        device
    );

    Tensor paged_prefill_output(
        {prefill_len, hidden_size},
        DType::FP32,
        device
    );

    Tensor paged_decode_output(
        {decode_len, hidden_size},
        DType::FP32,
        device
    );

    ordinary_prefill_output.zero_();
    ordinary_decode_output.zero_();
    paged_prefill_output.zero_();
    paged_decode_output.zero_();

    /*
     * 普通连续 KV cache 路径。
     */
    ModelKVCache kv_cache(
        make_test_config(),
        device,
        DType::FP32,
        capacity
    );

    {
        ForwardContext context;

        context.position_ids = &position_prefill;
        context.seq_len = prefill_len;
        context.past_len = 0;

        context.use_cache = true;
        context.kv_cache = &kv_cache;
        context.layer_idx = layer_idx;

        context.use_paged_kv_cache = false;

        attention.forward(
            hidden_prefill,
            context,
            ordinary_prefill_output
        );

        kv_cache.advance(prefill_len);
    }

    {
        ForwardContext context;

        context.position_ids = &position_decode;
        context.seq_len = decode_len;
        context.past_len = prefill_len;

        context.use_cache = true;
        context.kv_cache = &kv_cache;
        context.layer_idx = layer_idx;

        context.use_paged_kv_cache = false;

        attention.forward(
            hidden_decode,
            context,
            ordinary_decode_output
        );

        kv_cache.advance(decode_len);
    }

    /*
     * Paged KV cache 路径。
     *
     * 这里特意设置：
     *   page_size = 2
     *   prefill_len = 2
     *   total_kv_len = 3
     *
     * prefill 阶段只需要 1 个 block。
     * decode 前 ensure 到 3 个 token，会跨 page 分配第 2 个 block。
     */
    ModelPagedKVCache paged_kv_cache(
        num_layers,
        capacity,
        page_size,
        num_kv_heads,
        head_dim,
        DType::FP32,
        device
    );

    BlockTableManager block_table_manager(
        paged_kv_cache.num_blocks()
    );

    const int64_t table_idx =
        block_table_manager.allocate_table();

    {
        block_table_manager.ensure_blocks(
            table_idx,
            prefill_len,
            page_size
        );

        assert(block_table_manager.table(table_idx).size() == 1);

        ForwardContext context;

        context.position_ids = &position_prefill;
        context.seq_len = prefill_len;
        context.past_len = 0;

        context.use_cache = true;
        context.kv_cache = nullptr;
        context.layer_idx = layer_idx;

        context.use_paged_kv_cache = true;
        context.paged_kv_cache = &paged_kv_cache;
        context.block_table_manager = &block_table_manager;
        context.table_idx = table_idx;

        attention.forward(
            hidden_prefill,
            context,
            paged_prefill_output
        );
    }

    {
        block_table_manager.ensure_blocks(
            table_idx,
            total_kv_len,
            page_size
        );

        assert(block_table_manager.table(table_idx).size() == 2);

        ForwardContext context;

        context.position_ids = &position_decode;
        context.seq_len = decode_len;
        context.past_len = prefill_len;

        context.use_cache = true;
        context.kv_cache = nullptr;
        context.layer_idx = layer_idx;

        context.use_paged_kv_cache = true;
        context.paged_kv_cache = &paged_kv_cache;
        context.block_table_manager = &block_table_manager;
        context.table_idx = table_idx;

        attention.forward(
            hidden_decode,
            context,
            paged_decode_output
        );
    }

    /*
     * 核心检查：
     *
     * 普通连续 KV cache decode 输出
     * 应该等于
     * PagedKVCache decode 输出。
     */
    const float tol =
        device == Device::CUDA ? 1e-3f : 1e-4f;

    expect_tensor_close(
        ordinary_decode_output,
        paged_decode_output,
        tol
    );

    block_table_manager.free_table(table_idx);

    assert(block_table_manager.num_used_tables() == 0);
    assert(block_table_manager.num_used_blocks() == 0);

    std::cout << "[test_qwen3_attention_paged_kv_cache_"
              << device_name
              << "] passed"
              << std::endl;
}

void test_invalid_paged_context_should_throw() {
    std::cout << "[test_invalid_paged_context_should_throw] start"
              << std::endl;

    constexpr Device device = Device::CPU;

    constexpr int64_t hidden_size = 3;
    constexpr int64_t num_layers = 1;
    constexpr int64_t capacity = 4;
    constexpr int64_t page_size = 2;
    constexpr int64_t num_kv_heads = 1;
    constexpr int64_t head_dim = 2;

    Qwen3Attention attention = make_initialized_attention(device);

    Tensor hidden_states(
        {1, hidden_size},
        DType::FP32,
        device
    );

    Tensor position_ids = make_int_tensor(
        {1},
        std::vector<int32_t>{0},
        device
    );

    Tensor output(
        {1, hidden_size},
        DType::FP32,
        device
    );

    ModelPagedKVCache paged_kv_cache(
        num_layers,
        capacity,
        page_size,
        num_kv_heads,
        head_dim,
        DType::FP32,
        device
    );

    BlockTableManager block_table_manager(
        paged_kv_cache.num_blocks()
    );

    const int64_t table_idx =
        block_table_manager.allocate_table();

    block_table_manager.ensure_blocks(
        table_idx,
        1,
        page_size
    );

    {
        ForwardContext context;

        context.position_ids = &position_ids;
        context.seq_len = 1;
        context.past_len = 0;

        context.use_cache = false;
        context.use_paged_kv_cache = true;

        bool caught = false;

        try {
            attention.forward(
                hidden_states,
                context,
                output
            );
        } catch (const std::runtime_error&) {
            caught = true;
        }

        if (!caught) {
            throw std::runtime_error(
                "Expected use_paged_kv_cache without use_cache error"
            );
        }
    }

    {
        ForwardContext context;

        context.position_ids = &position_ids;
        context.seq_len = 1;
        context.past_len = 0;

        context.use_cache = true;
        context.layer_idx = 0;

        context.use_paged_kv_cache = true;
        context.paged_kv_cache = nullptr;
        context.block_table_manager = &block_table_manager;
        context.table_idx = table_idx;

        bool caught = false;

        try {
            attention.forward(
                hidden_states,
                context,
                output
            );
        } catch (const std::runtime_error&) {
            caught = true;
        }

        if (!caught) {
            throw std::runtime_error(
                "Expected null paged_kv_cache error"
            );
        }
    }

    {
        ForwardContext context;

        context.position_ids = &position_ids;
        context.seq_len = 1;
        context.past_len = 0;

        context.use_cache = true;
        context.layer_idx = 0;

        context.use_paged_kv_cache = true;
        context.paged_kv_cache = &paged_kv_cache;
        context.block_table_manager = nullptr;
        context.table_idx = table_idx;

        bool caught = false;

        try {
            attention.forward(
                hidden_states,
                context,
                output
            );
        } catch (const std::runtime_error&) {
            caught = true;
        }

        if (!caught) {
            throw std::runtime_error(
                "Expected null block_table_manager error"
            );
        }
    }

    {
        ForwardContext context;

        context.position_ids = &position_ids;
        context.seq_len = 1;
        context.past_len = 0;

        context.use_cache = true;
        context.layer_idx = 0;

        context.use_paged_kv_cache = true;
        context.paged_kv_cache = &paged_kv_cache;
        context.block_table_manager = &block_table_manager;
        context.table_idx = -1;

        bool caught = false;

        try {
            attention.forward(
                hidden_states,
                context,
                output
            );
        } catch (const std::runtime_error&) {
            caught = true;
        }

        if (!caught) {
            throw std::runtime_error(
                "Expected invalid table_idx error"
            );
        }
    }

    block_table_manager.free_table(table_idx);

    std::cout << "[test_invalid_paged_context_should_throw] passed"
              << std::endl;
}

}  // namespace

int main() {
    try {
        run_qwen3_attention_paged_kv_cache_decode_test(Device::CPU);
        run_qwen3_attention_paged_kv_cache_decode_test(Device::CUDA);

        test_invalid_paged_context_should_throw();
    } catch (const std::exception& e) {
        std::cerr << "[test_qwen3_attention_paged_kv_cache] failed: "
                  << e.what()
                  << std::endl;
        return 1;
    }

    std::cout << "[test_qwen3_attention_paged_kv_cache] all passed"
              << std::endl;

    return 0;
}
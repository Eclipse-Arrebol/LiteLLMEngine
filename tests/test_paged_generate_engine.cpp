// tests/test_paged_generate_engine.cpp

#include "core/device.hpp"
#include "core/dtype.hpp"
#include "core/tensor.hpp"
#include "engine/paged_generate_engine.hpp"
#include "model/qwen3.hpp"
#include "runtime/generation.hpp"
#include "weights/weight_map.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lite_llm;

static Tensor make_float_tensor(
    const std::vector<int64_t>& shape,
    const std::vector<float>& values
) {
    Tensor tensor(
        shape,
        DType::FP32,
        Device::CPU
    );

    assert(tensor.numel() == values.size());

    tensor.copy_from_cpu(
        values.data(),
        values.size() * sizeof(float)
    );

    return tensor;
}

static Tensor make_constant_weight(
    const std::vector<int64_t>& shape,
    float value
) {
    int64_t numel = 1;

    for (int64_t dim : shape) {
        numel *= dim;
    }

    std::vector<float> values(
        static_cast<size_t>(numel),
        value
    );

    return make_float_tensor(
        shape,
        values
    );
}

static Tensor make_sequential_weight(
    const std::vector<int64_t>& shape,
    float scale = 0.01f,
    float bias = 0.0f
) {
    int64_t numel = 1;

    for (int64_t dim : shape) {
        numel *= dim;
    }

    std::vector<float> values(
        static_cast<size_t>(numel)
    );

    for (int64_t i = 0; i < numel; ++i) {
        const int64_t pattern = (i % 17) - 8;
        values[static_cast<size_t>(i)] =
            bias + scale * static_cast<float>(pattern);
    }

    return make_float_tensor(
        shape,
        values
    );
}

static ModelConfig make_tiny_config() {
    ModelConfig config;

    config.model_type = "qwen3";

    config.vocab_size = 16;
    config.hidden_size = 4;
    config.intermediate_size = 8;

    config.num_hidden_layers = 1;
    config.num_attention_heads = 2;
    config.num_key_value_heads = 1;
    config.head_dim = 2;

    config.max_position_embeddings = 128;

    config.rms_norm_eps = 1e-6f;
    config.rope_theta = 10000.0f;

    config.tie_word_embeddings = false;

    return config;
}

static Qwen3ForCausalLM make_tiny_model() {
    const ModelConfig config = make_tiny_config();

    Qwen3ForCausalLM model(config);

    WeightMap weights;

    weights.add(
        "model.embed_tokens.weight",
        make_sequential_weight(
            {config.vocab_size, config.hidden_size},
            0.02f,
            0.01f
        )
    );

    weights.add(
        "model.layers.0.input_layernorm.weight",
        make_constant_weight(
            {config.hidden_size},
            1.0f
        )
    );

    weights.add(
        "model.layers.0.post_attention_layernorm.weight",
        make_constant_weight(
            {config.hidden_size},
            1.0f
        )
    );

    weights.add(
        "model.layers.0.self_attn.q_proj.weight",
        make_sequential_weight(
            {config.num_attention_heads * config.head_dim, config.hidden_size},
            0.015f,
            0.0f
        )
    );

    weights.add(
        "model.layers.0.self_attn.k_proj.weight",
        make_sequential_weight(
            {config.num_key_value_heads * config.head_dim, config.hidden_size},
            0.013f,
            0.0f
        )
    );

    weights.add(
        "model.layers.0.self_attn.v_proj.weight",
        make_sequential_weight(
            {config.num_key_value_heads * config.head_dim, config.hidden_size},
            0.017f,
            0.0f
        )
    );

    weights.add(
        "model.layers.0.self_attn.o_proj.weight",
        make_sequential_weight(
            {config.hidden_size, config.num_attention_heads * config.head_dim},
            0.011f,
            0.0f
        )
    );

    weights.add(
        "model.layers.0.self_attn.q_norm.weight",
        make_constant_weight(
            {config.head_dim},
            1.0f
        )
    );

    weights.add(
        "model.layers.0.self_attn.k_norm.weight",
        make_constant_weight(
            {config.head_dim},
            1.0f
        )
    );

    weights.add(
        "model.layers.0.mlp.gate_proj.weight",
        make_sequential_weight(
            {config.intermediate_size, config.hidden_size},
            0.012f,
            0.0f
        )
    );

    weights.add(
        "model.layers.0.mlp.up_proj.weight",
        make_sequential_weight(
            {config.intermediate_size, config.hidden_size},
            0.014f,
            0.0f
        )
    );

    weights.add(
        "model.layers.0.mlp.down_proj.weight",
        make_sequential_weight(
            {config.hidden_size, config.intermediate_size},
            0.016f,
            0.0f
        )
    );

    weights.add(
        "model.norm.weight",
        make_constant_weight(
            {config.hidden_size},
            1.0f
        )
    );

    weights.add(
        "lm_head.weight",
        make_sequential_weight(
            {config.vocab_size, config.hidden_size},
            0.018f,
            0.0f
        )
    );

    model.load_weights(weights);

    if (!model.initialized()) {
        throw std::runtime_error(
            "make_tiny_model failed: model is not initialized"
        );
    }

    return model;
}

static GreedyGenerateOptions make_options(
    Device device,
    int64_t max_new_tokens,
    int32_t eos_token_id = -1
) {
    GreedyGenerateOptions options;

    options.device = device;
    options.max_new_tokens = max_new_tokens;
    options.eos_token_id = eos_token_id;
    options.verbose = false;

    return options;
}

static void expect_same_vector(
    const std::vector<int32_t>& lhs,
    const std::vector<int32_t>& rhs,
    const std::string& name
) {
    if (lhs != rhs) {
        std::cerr << "[vector mismatch] " << name << std::endl;

        std::cerr << "lhs:";
        for (int32_t x : lhs) {
            std::cerr << " " << x;
        }
        std::cerr << std::endl;

        std::cerr << "rhs:";
        for (int32_t x : rhs) {
            std::cerr << " " << x;
        }
        std::cerr << std::endl;

        throw std::runtime_error(
            "vector mismatch: " + name
        );
    }
}

static std::vector<int32_t> generate_with_paged_engine(
    const Qwen3ForCausalLM& model,
    const std::vector<int32_t>& prompt_ids,
    const GreedyGenerateOptions& options,
    int64_t page_size = 2
) {
    if (options.max_new_tokens == 0) {
        return prompt_ids;
    }

    const int64_t max_total_tokens =
        static_cast<int64_t>(prompt_ids.size()) +
        options.max_new_tokens +
        16;

    PagedGenerateEngine engine(
        model,
        options,
        max_total_tokens,
        page_size
    );

    const int64_t request_id =
        engine.add_request(
            prompt_ids,
            options.max_new_tokens,
            options.eos_token_id
        );

    std::vector<int32_t> output =
        engine.generate_until_finished(request_id);

    engine.release_request(request_id);

    return output;
}

static void test_single_request_matches_contiguous_kv(
    Device device
) {
    std::cout << "[test_single_request_matches_contiguous_kv] device="
              << (device == Device::CPU ? "cpu" : "cuda")
              << std::endl;

    Qwen3ForCausalLM model = make_tiny_model();

    const std::vector<int32_t> prompt_ids = {1, 2, 3};
    const int64_t max_new_tokens = 4;

    GreedyGenerateOptions options =
        make_options(
            device,
            max_new_tokens,
            -1
        );

    const std::vector<int32_t> contiguous_output =
        generate_greedy_with_kv_cache(
            model,
            prompt_ids,
            options
        );

    const std::vector<int32_t> paged_output =
        generate_with_paged_engine(
            model,
            prompt_ids,
            options,
            2
        );

    expect_same_vector(
        contiguous_output,
        paged_output,
        "single request paged vs contiguous"
    );

    assert(
        static_cast<int64_t>(paged_output.size()) ==
        static_cast<int64_t>(prompt_ids.size()) + max_new_tokens
    );
}

static void test_max_new_tokens_one(
    Device device
) {
    std::cout << "[test_max_new_tokens_one] device="
              << (device == Device::CPU ? "cpu" : "cuda")
              << std::endl;

    Qwen3ForCausalLM model = make_tiny_model();

    const std::vector<int32_t> prompt_ids = {4, 5};
    const int64_t max_new_tokens = 1;

    GreedyGenerateOptions options =
        make_options(
            device,
            max_new_tokens,
            -1
        );

    const std::vector<int32_t> output =
        generate_with_paged_engine(
            model,
            prompt_ids,
            options,
            2
        );

    assert(output.size() == prompt_ids.size() + 1);

    for (size_t i = 0; i < prompt_ids.size(); ++i) {
        assert(output[i] == prompt_ids[i]);
    }
}

static void test_two_requests_interleaved(
    Device device
) {
    std::cout << "[test_two_requests_interleaved] device="
              << (device == Device::CPU ? "cpu" : "cuda")
              << std::endl;

    Qwen3ForCausalLM model = make_tiny_model();

    const std::vector<int32_t> prompt0 = {1, 2, 3};
    const std::vector<int32_t> prompt1 = {4, 5};

    const int64_t max_new_tokens = 3;

    GreedyGenerateOptions options =
        make_options(
            device,
            max_new_tokens,
            -1
        );

    const std::vector<int32_t> expected0 =
        generate_with_paged_engine(
            model,
            prompt0,
            options,
            2
        );

    const std::vector<int32_t> expected1 =
        generate_with_paged_engine(
            model,
            prompt1,
            options,
            2
        );

    PagedGenerateEngine engine(
        model,
        options,
        64,
        2
    );

    const int64_t req0 =
        engine.add_request(
            prompt0,
            max_new_tokens,
            -1
        );

    const int64_t req1 =
        engine.add_request(
            prompt1,
            max_new_tokens,
            -1
        );

    engine.prefill(req0);
    engine.prefill(req1);

    assert(
        engine.request(req0).table_idx !=
        engine.request(req1).table_idx
    );

    assert(
        engine.paged_kv_manager()
              .block_table_manager()
              .num_used_tables() == 2
    );

    while (!engine.finished(req0) ||
           !engine.finished(req1)) {
        if (!engine.finished(req0)) {
            engine.decode_one_step(req0);
        }

        if (!engine.finished(req1)) {
            engine.decode_one_step(req1);
        }
    }

    const std::vector<int32_t> output0 =
        engine.request(req0).input_ids;

    const std::vector<int32_t> output1 =
        engine.request(req1).input_ids;

    expect_same_vector(
        expected0,
        output0,
        "interleaved request 0"
    );

    expect_same_vector(
        expected1,
        output1,
        "interleaved request 1"
    );

    assert(engine.request(req0).generated_ids.size() == 3);
    assert(engine.request(req1).generated_ids.size() == 3);

    engine.release_request(req0);

    assert(engine.request(req0).table_idx == -1);
    assert(engine.request(req1).table_idx >= 0);

    engine.release_request(req1);

    assert(engine.request(req1).table_idx == -1);

    assert(
        engine.paged_kv_manager()
              .block_table_manager()
              .num_used_tables() == 0
    );

    assert(
        engine.paged_kv_manager()
              .block_table_manager()
              .num_used_blocks() == 0
    );
}

static void test_multi_turn_conversation(
    Device device
) {
    std::cout << "[test_multi_turn_conversation] device="
              << (device == Device::CPU ? "cpu" : "cuda")
              << std::endl;

    Qwen3ForCausalLM model = make_tiny_model();

    GreedyGenerateOptions options =
        make_options(
            device,
            2,
            -1
        );

    PagedGenerateEngine engine(
        model,
        options,
        128,
        2
    );

    const std::vector<int32_t> first_user_tokens = {1, 2, 3};

    const int64_t request_id =
        engine.add_request(
            first_user_tokens,
            2,
            -1
        );

    const std::vector<int32_t> round1_output =
        engine.generate_until_finished(request_id);

    const GenerationRequest& req_after_round1 =
        engine.request(request_id);

    assert(req_after_round1.status == RequestStatus::Finished);
    assert(req_after_round1.generated_ids.size() == 2);
    assert(round1_output.size() == first_user_tokens.size() + 2);

    const int64_t cached_len_after_round1 =
        req_after_round1.cached_len;

    assert(cached_len_after_round1 > 0);
    assert(cached_len_after_round1 <= req_after_round1.device_len());

    const std::vector<int32_t> second_user_tokens = {6, 7};

    engine.append_input_tokens(
        request_id,
        second_user_tokens,
        3,
        -1
    );

    const GenerationRequest& req_after_append =
        engine.request(request_id);

    assert(req_after_append.status != RequestStatus::Finished);

    /*
     * 新一轮开始后 generated_ids 必须清空。
     * 否则上一轮生成数量会影响这一轮 max_new_tokens 判断。
     */
    assert(req_after_append.generated_ids.empty());

    /*
     * input_ids 保留历史上下文，并追加新的 user tokens。
     */
    assert(
        req_after_append.input_ids.size() ==
        round1_output.size() + second_user_tokens.size()
    );

    /*
     * cached_len 不应该因为 append user tokens 立刻变化。
     * 新增 token 要等下一次 prefill 才写入 KV cache。
     */
    assert(req_after_append.cached_len == cached_len_after_round1);

    const std::vector<int32_t> round2_output =
        engine.generate_until_finished(request_id);

    const GenerationRequest& req_after_round2 =
        engine.request(request_id);

    assert(req_after_round2.status == RequestStatus::Finished);
    assert(req_after_round2.generated_ids.size() == 3);

    assert(
        round2_output.size() ==
        round1_output.size() +
        second_user_tokens.size() +
        3
    );

    for (size_t i = 0; i < round1_output.size(); ++i) {
        assert(round2_output[i] == round1_output[i]);
    }

    for (size_t i = 0; i < second_user_tokens.size(); ++i) {
        assert(
            round2_output[round1_output.size() + i] ==
            second_user_tokens[i]
        );
    }

    engine.release_request(request_id);

    assert(engine.request(request_id).table_idx == -1);
}

static void test_eos_is_appended_by_request_manager() {
    std::cout << "[test_eos_is_appended_by_request_manager]"
              << std::endl;

    RequestManager manager;

    const int64_t request_id =
        manager.add_request(
            {1, 2, 3},
            8,
            99
        );

    manager.mark_forward_done(request_id);

    manager.append_token(
        request_id,
        99
    );

    const GenerationRequest& req =
        manager.request(request_id);

    assert(req.status == RequestStatus::Finished);

    /*
     * EOS 必须进入 input_ids。
     */
    assert(!req.input_ids.empty());
    assert(req.input_ids.back() == 99);

    /*
     * EOS 也应该算作本轮 generated token。
     */
    assert(!req.generated_ids.empty());
    assert(req.generated_ids.back() == 99);
}

int main() {
    try {
        test_eos_is_appended_by_request_manager();

        test_single_request_matches_contiguous_kv(Device::CPU);
        test_max_new_tokens_one(Device::CPU);
        test_two_requests_interleaved(Device::CPU);
        test_multi_turn_conversation(Device::CPU);

        test_single_request_matches_contiguous_kv(Device::CUDA);
        test_max_new_tokens_one(Device::CUDA);
        test_two_requests_interleaved(Device::CUDA);
        test_multi_turn_conversation(Device::CUDA);

        std::cout << "test_paged_generate_engine passed"
                  << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "test_paged_generate_engine failed: "
                  << e.what()
                  << std::endl;

        return 1;
    }
}
#include "runtime/generation.hpp"

#include "core/tensor.hpp"
#include "ops/argmax.hpp"
#include "engine/kv_cache.hpp"
#include "core/tensor_memory_tracker.hpp"

#include <stdexcept>
#include <vector>
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

namespace lite_llm {

namespace {
double now_ms_for_generation() {
    using Clock = std::chrono::high_resolution_clock;
    auto now = Clock::now().time_since_epoch();
    return std::chrono::duration<double, std::milli>(now).count();
}

void sync_generation_device(Device device) {
    if (device == Device::CUDA) {
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaDeviceSynchronize failed: ") +
                cudaGetErrorString(err)
            );
        }
    }
}
std::vector<int32_t> make_position_ids(int64_t seq_len) {
    if (seq_len < 0) {
        throw std::runtime_error("seq_len must be non-negative");
    }

    std::vector<int32_t> position_ids(static_cast<size_t>(seq_len));

    for (int64_t i = 0; i < seq_len; ++i) {
        position_ids[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    }

    return position_ids;
}

Tensor make_int32_tensor(
    const std::vector<int32_t>& data,
    Device device
) {
    Tensor tensor(
        {static_cast<int64_t>(data.size())},
        DType::INT32,
        device
    );

    if (!data.empty()) {
        tensor.copy_from_cpu(
            data.data(),
            data.size() * sizeof(int32_t)
        );
    }

    return tensor;
}

}  // namespace

std::vector<int32_t> generate_greedy(
    const Qwen3ForCausalLM& model,
    const std::vector<int32_t>& input_ids,
    const GreedyGenerateOptions& options
) {
    if (!model.initialized()) {
        throw std::runtime_error("generate_greedy model is not initialized");
    }

    if (input_ids.empty()) {
        throw std::runtime_error("generate_greedy input_ids must not be empty");
    }

    if (options.max_new_tokens < 0) {
        throw std::runtime_error("generate_greedy max_new_tokens must be non-negative");
    }

    const ModelConfig& config = model.config();

    if (config.vocab_size <= 0) {
        throw std::runtime_error("generate_greedy vocab_size must be positive");
    }

    std::vector<int32_t> generated = input_ids;

    for (int64_t step = 0; step < options.max_new_tokens; ++step) {
        const int64_t seq_len = static_cast<int64_t>(generated.size());

        std::vector<int32_t> position_ids_cpu = make_position_ids(seq_len);

        Tensor input_ids_tensor = make_int32_tensor(
            generated,
            options.device
        );

        Tensor position_ids_tensor = make_int32_tensor(
            position_ids_cpu,
            options.device
        );

        Tensor logits(
            {seq_len, config.vocab_size},
            DType::FP32,
            options.device
        );

        ForwardContext context;
        context.position_ids = &position_ids_tensor;
        context.seq_len = seq_len;
        context.past_len = 0;
        context.use_cache = false;

        model.forward(input_ids_tensor, context, logits);


        const int32_t next_token_id = argmax_last_token(logits);

        if (options.verbose) {
            std::cerr << "[generate] step "
                    << (step + 1)
                    << "/"
                    << options.max_new_tokens
                    << ", seq_len="
                    << seq_len
                    << ", next_token_id="
                    << next_token_id
                    << std::endl;
        }

        if (options.eos_token_id >= 0 &&
            next_token_id == options.eos_token_id) {
            if (options.verbose) {
                std::cerr << "[generate] hit eos_token_id="
                        << next_token_id
                        << std::endl;
            }
            break;
        }

        generated.push_back(next_token_id);
    }

    return generated;
}

std::vector<int32_t> generate_greedy_with_kv_cache(
    const Qwen3ForCausalLM& model,
    const std::vector<int32_t>& input_ids,
    const GreedyGenerateOptions& options
) {
    if (!model.initialized()) {
        throw std::runtime_error(
            "generate_greedy_with_kv_cache model is not initialized"
        );
    }

    if (input_ids.empty()) {
        throw std::runtime_error(
            "generate_greedy_with_kv_cache input_ids must not be empty"
        );
    }

    if (options.max_new_tokens < 0) {
        throw std::runtime_error(
            "generate_greedy_with_kv_cache max_new_tokens must be non-negative"
        );
    }

    const ModelConfig& config = model.config();

    if (config.vocab_size <= 0) {
        throw std::runtime_error(
            "generate_greedy_with_kv_cache vocab_size must be positive"
        );
    }

    std::vector<int32_t> generated = input_ids;

    if (options.max_new_tokens == 0) {
        return generated;
    }

    const int64_t initial_capacity =
    static_cast<int64_t>(input_ids.size()) + options.max_new_tokens;

    ModelKVCache kv_cache(
        config,
        options.device,
        DType::FP32,
        initial_capacity
    );

    // -------------------------
    // 1. prefill: 输入完整 prompt
    // -------------------------
    {
        const int64_t seq_len = static_cast<int64_t>(input_ids.size());

        std::vector<int32_t> position_ids_cpu =
            make_position_ids(seq_len);

        Tensor input_ids_tensor = make_int32_tensor(
            input_ids,
            options.device
        );

        Tensor position_ids_tensor = make_int32_tensor(
            position_ids_cpu,
            options.device
        );

        Tensor logits(
            {seq_len, config.vocab_size},
            DType::FP32,
            options.device
        );

        ForwardContext context;
        context.position_ids = &position_ids_tensor;
        context.seq_len = seq_len;
        context.past_len = kv_cache.current_len();  // 这里应该是 0
        context.use_cache = true;
        context.kv_cache = &kv_cache;

        
        const TensorMemorySnapshot mem_before_prefill = tensor_memory_snapshot();
        model.forward(input_ids_tensor, context, logits);
        
        const TensorMemorySnapshot mem_after_prefill =
            tensor_memory_snapshot();

        if (options.verbose) {
            print_tensor_memory_delta(
                "generate_kv prefill forward",
                mem_before_prefill,
                mem_after_prefill
            );
        }
        
        


        // 如果 Qwen3Model::forward 已经正确调用 advance(seq_len)，
        // 那么这里 kv_cache.current_len() 应该等于 prompt_len。
        if (kv_cache.current_len() != seq_len) {
            throw std::runtime_error(
                "generate_greedy_with_kv_cache prefill did not advance kv_cache"
            );
        }

        const int32_t next_token_id = argmax_last_token(logits);

        if (options.verbose) {
            std::cerr << "[generate_kv] prefill"
                      << ", seq_len=" << seq_len
                      << ", next_token_id=" << next_token_id
                      << std::endl;
        }

        if (options.eos_token_id >= 0 &&
            next_token_id == options.eos_token_id) {
            if (options.verbose) {
                std::cerr << "[generate_kv] hit eos_token_id="
                          << next_token_id
                          << std::endl;
            }
            return generated;
        }

        generated.push_back(next_token_id);
    }

    // -------------------------
    // 2. decode: 每次只输入上一个 token
    // -------------------------
    for (int64_t step = 1; step < options.max_new_tokens; ++step) {
        const int32_t last_token_id = generated.back();

        const int64_t past_len = kv_cache.current_len();

        Tensor input_ids_tensor = make_int32_tensor(
            std::vector<int32_t>{last_token_id},
            options.device
        );

        Tensor position_ids_tensor = make_int32_tensor(
            std::vector<int32_t>{static_cast<int32_t>(past_len)},
            options.device
        );

        Tensor logits(
            {1, config.vocab_size},
            DType::FP32,
            options.device
        );

        ForwardContext context;
        context.position_ids = &position_ids_tensor;
        context.seq_len = 1;
        context.past_len = past_len;
        context.use_cache = true;
        context.kv_cache = &kv_cache;

        const TensorMemorySnapshot mem_before_decode =
            tensor_memory_snapshot();

        model.forward(input_ids_tensor, context, logits);

        const TensorMemorySnapshot mem_after_decode =
            tensor_memory_snapshot();

        if (options.verbose) {
            print_tensor_memory_delta(
                "generate_kv decode forward",
                mem_before_decode,
                mem_after_decode
            );
        }



        if (kv_cache.current_len() != past_len + 1) {
            throw std::runtime_error(
                "generate_greedy_with_kv_cache decode did not advance kv_cache"
            );
        }

        const int32_t next_token_id = argmax_last_token(logits);

        if (options.verbose) {
            std::cerr << "[generate_kv] step "
                      << (step + 1)
                      << "/"
                      << options.max_new_tokens
                      << ", past_len="
                      << past_len
                      << ", next_token_id="
                      << next_token_id
                      << std::endl;
        }

        if (options.eos_token_id >= 0 &&
            next_token_id == options.eos_token_id) {
            if (options.verbose) {
                std::cerr << "[generate_kv] hit eos_token_id="
                          << next_token_id
                          << std::endl;
            }
            break;
        }

        generated.push_back(next_token_id);
    }

    return generated;
}

}  // namespace lite_llm
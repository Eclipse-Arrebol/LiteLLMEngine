#include "runtime/generation.hpp"

#include "core/tensor.hpp"
#include "ops/argmax.hpp"

#include <stdexcept>
#include <vector>
#include <iostream>

namespace lite_llm {

namespace {

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

}  // namespace lite_llm
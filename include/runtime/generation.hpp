#pragma once

#include "core/device.hpp"
#include "model/qwen3.hpp"

#include <cstdint>
#include <vector>

namespace lite_llm {

struct GreedyGenerateOptions {
    int64_t max_new_tokens = 16;

    // 小于 0 表示不检查 eos
    int32_t eos_token_id = -1;

    bool verbose = false;

    Device device = Device::CUDA;
};

std::vector<int32_t> generate_greedy(
    const Qwen3ForCausalLM& model,
    const std::vector<int32_t>& input_ids,
    const GreedyGenerateOptions& options
);

std::vector<int32_t> generate_greedy_with_kv_cache(
    const Qwen3ForCausalLM& model,
    const std::vector<int32_t>& input_ids,
    const GreedyGenerateOptions& options
);

}  // namespace lite_llm
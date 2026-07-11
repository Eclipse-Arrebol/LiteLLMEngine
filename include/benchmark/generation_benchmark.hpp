#pragma once

#include "core/device.hpp"
#include "model/qwen3.hpp"

#include <cstdint>
#include <vector>

namespace lite_llm {

struct GenerationBenchmarkOptions {
    int32_t num_requests = 32;
    int64_t max_new_tokens = 32;
    int32_t warmup_requests = 1;
    int32_t eos_token_id = -1;

    bool verbose = false;

    // 普通连续 KV cache
    bool use_kv_cache = false;

    // Paged KV cache
    bool use_paged_kv_cache = false;

    // 是否在同一个 PagedGenerateEngine 里跑多个 request
    bool interleaved = false;

    int64_t page_size = 16;

    Device device = Device::CUDA;
};

struct GenerationBenchmarkResult {
    int32_t num_requests = 0;
    int64_t prompt_tokens = 0;
    int64_t max_new_tokens = 0;
    int64_t total_new_tokens = 0;

    double total_time_ms = 0.0;
    double tok_per_sec = 0.0;
    double ms_per_token = 0.0;

    bool use_kv_cache = false;
    bool use_paged_kv_cache = false;
    bool interleaved = false;

    int64_t page_size = 16;
};

GenerationBenchmarkResult benchmark_generate_greedy(
    const Qwen3ForCausalLM& model,
    const std::vector<int32_t>& input_ids,
    const GenerationBenchmarkOptions& options
);

void print_generation_benchmark_result(
    const GenerationBenchmarkResult& result
);

}  // namespace lite_llm
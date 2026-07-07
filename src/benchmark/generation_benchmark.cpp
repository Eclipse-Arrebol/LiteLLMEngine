#include "benchmark/generation_benchmark.hpp"
#include "runtime/generation.hpp"

#include <chrono>
#include <iostream>
#include <stdexcept>


#include <cuda_runtime.h>


namespace lite_llm {
namespace {

void sync_device_if_needed(Device device) {

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

double now_ms() {
    using Clock = std::chrono::high_resolution_clock;
    auto now = Clock::now().time_since_epoch();
    return std::chrono::duration<double, std::milli>(now).count();
}

}  // namespace

GenerationBenchmarkResult benchmark_generate_greedy(
    const Qwen3ForCausalLM& model,
    const std::vector<int32_t>& input_ids,
    const GenerationBenchmarkOptions& options
) {
    if (!model.initialized()) {
        throw std::runtime_error("benchmark_generate_greedy: model is not initialized");
    }

    if (input_ids.empty()) {
        throw std::runtime_error("benchmark_generate_greedy: input_ids is empty");
    }

    if (options.num_requests <= 0) {
        throw std::runtime_error("benchmark_generate_greedy: num_requests must be positive");
    }

    if (options.max_new_tokens < 0) {
        throw std::runtime_error("benchmark_generate_greedy: max_new_tokens must be non-negative");
    }

    GreedyGenerateOptions gen_options;
    gen_options.max_new_tokens = options.max_new_tokens;
    gen_options.eos_token_id = options.eos_token_id;
    gen_options.verbose = options.verbose;
    gen_options.device = options.device;

    for (int32_t i = 0; i < options.warmup_requests; ++i) {
        (void)generate_greedy(model, input_ids, gen_options);
    }

    sync_device_if_needed(options.device);

    const double start_ms = now_ms();

    int64_t total_new_tokens = 0;

    for (int32_t i = 0; i < options.num_requests; ++i) {
        std::vector<int32_t> output_ids =
            generate_greedy(model, input_ids, gen_options);

        if (output_ids.size() < input_ids.size()) {
            throw std::runtime_error("benchmark_generate_greedy: invalid output length");
        }

        total_new_tokens +=
            static_cast<int64_t>(output_ids.size() - input_ids.size());
    }

    sync_device_if_needed(options.device);

    const double end_ms = now_ms();
    const double elapsed_ms = end_ms - start_ms;
    const double elapsed_s = elapsed_ms / 1000.0;

    GenerationBenchmarkResult result;
    result.num_requests = options.num_requests;
    result.prompt_tokens = static_cast<int64_t>(input_ids.size());
    result.max_new_tokens = options.max_new_tokens;
    result.total_new_tokens = total_new_tokens;
    result.total_time_ms = elapsed_ms;

    if (total_new_tokens > 0 && elapsed_s > 0.0) {
        result.tok_per_sec =
            static_cast<double>(total_new_tokens) / elapsed_s;
        result.ms_per_token =
            elapsed_ms / static_cast<double>(total_new_tokens);
    }

    return result;
}

void print_generation_benchmark_result(
    const GenerationBenchmarkResult& result
) {
    std::cout << "\nGeneration benchmark result\n";
    std::cout << "  Requests:         " << result.num_requests << "\n";
    std::cout << "  Prompt tokens:    " << result.prompt_tokens << "\n";
    std::cout << "  Max new tokens:   " << result.max_new_tokens << "\n";
    std::cout << "  Total new tokens: " << result.total_new_tokens << "\n";
    std::cout << "  Total time:       " << result.total_time_ms << " ms\n";
    std::cout << "  Throughput:       " << result.tok_per_sec << " tok/s\n";
    std::cout << "  Latency/token:    " << result.ms_per_token << " ms/token\n";
}

}  // namespace lite_llm
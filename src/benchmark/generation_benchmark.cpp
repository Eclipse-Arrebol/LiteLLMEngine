#include "benchmark/generation_benchmark.hpp"

#include "engine/paged_generate_engine.hpp"
#include "runtime/generation.hpp"

#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

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

const char* benchmark_mode_name(
    bool use_kv_cache,
    bool use_paged_kv_cache,
    bool interleaved,
    bool batch_decode
) {
    if (use_paged_kv_cache) {
        if (batch_decode) {
            return "Paged KV cache batch decode";
        }

        if (interleaved) {
            return "Paged KV cache interleaved";
        }

        return "Paged KV cache";
    }

    if (use_kv_cache) {
        return "KV cache";
    }

    return "No KV cache";
}

std::vector<int32_t> generate_greedy_with_paged_engine_once(
    const Qwen3ForCausalLM& model,
    const std::vector<int32_t>& input_ids,
    const GreedyGenerateOptions& gen_options,
    int64_t page_size
) {
    if (gen_options.max_new_tokens == 0) {
        return input_ids;
    }

    const int64_t max_total_tokens =
        static_cast<int64_t>(input_ids.size()) +
        gen_options.max_new_tokens +
        16;

    PagedGenerateEngine engine(
        model,
        gen_options,
        max_total_tokens,
        page_size
    );

    const int64_t request_id =
        engine.add_request(
            input_ids,
            gen_options.max_new_tokens,
            gen_options.eos_token_id
        );

    std::vector<int32_t> output_ids =
        engine.generate_until_finished(request_id);

    engine.release_request(request_id);

    return output_ids;
}

std::vector<int32_t> run_one_generation(
    const Qwen3ForCausalLM& model,
    const std::vector<int32_t>& input_ids,
    const GenerationBenchmarkOptions& options,
    const GreedyGenerateOptions& gen_options
) {
    if (options.use_paged_kv_cache) {
        return generate_greedy_with_paged_engine_once(
            model,
            input_ids,
            gen_options,
            options.page_size
        );
    }

    if (options.use_kv_cache) {
        return generate_greedy_with_kv_cache(
            model,
            input_ids,
            gen_options
        );
    }

    return generate_greedy(
        model,
        input_ids,
        gen_options
    );
}

int64_t run_paged_interleaved_generation(
    const Qwen3ForCausalLM& model,
    const std::vector<int32_t>& input_ids,
    const GenerationBenchmarkOptions& options,
    const GreedyGenerateOptions& gen_options
) {
    if (!options.use_paged_kv_cache) {
        throw std::runtime_error(
            "run_paged_interleaved_generation requires paged KV cache"
        );
    }

    if (options.num_requests <= 0) {
        throw std::runtime_error(
            "run_paged_interleaved_generation num_requests must be positive"
        );
    }

    if (gen_options.max_new_tokens == 0) {
        return 0;
    }

    const int64_t tokens_per_request =
        static_cast<int64_t>(input_ids.size()) +
        gen_options.max_new_tokens +
        16;

    const int64_t max_total_tokens =
        tokens_per_request *
        static_cast<int64_t>(options.num_requests);

    PagedGenerateEngine engine(
        model,
        gen_options,
        max_total_tokens,
        options.page_size
    );

    std::vector<int64_t> request_ids;
    request_ids.reserve(
        static_cast<size_t>(options.num_requests)
    );

    for (int32_t i = 0; i < options.num_requests; ++i) {
        const int64_t request_id =
            engine.add_request(
                input_ids,
                gen_options.max_new_tokens,
                gen_options.eos_token_id
            );

        request_ids.push_back(request_id);
    }

    /*
     * 先把所有 request 的 prompt 写进 KV cache。
     * 当前 prefill 内部还会采样第一个 token，并 append 到 request。
     */
    for (int64_t request_id : request_ids) {
        if (!engine.finished(request_id)) {
            engine.prefill(request_id);
        }
    }

    /*
     * round-robin decode。
     * 这是真正多个 request 共用一个 PagedGenerateEngine。
     */
    bool all_finished = false;

    while (!all_finished) {
        all_finished = true;

        for (int64_t request_id : request_ids) {
            if (!engine.finished(request_id)) {
                all_finished = false;

                engine.decode_one_step(request_id);
            }
        }
    }

    int64_t total_new_tokens = 0;

    for (int64_t request_id : request_ids) {
        const GenerationRequest& req =
            engine.request(request_id);

        if (req.input_ids.size() < input_ids.size()) {
            throw std::runtime_error(
                "run_paged_interleaved_generation invalid output length"
            );
        }

        total_new_tokens +=
            static_cast<int64_t>(
                req.input_ids.size() - input_ids.size()
            );
    }

    for (int64_t request_id : request_ids) {
        engine.release_request(request_id);
    }

    return total_new_tokens;
}

int64_t run_paged_batch_decode_generation(
    const Qwen3ForCausalLM& model,
    const std::vector<int32_t>& input_ids,
    const GenerationBenchmarkOptions& options,
    const GreedyGenerateOptions& gen_options
) {
    if (!options.use_paged_kv_cache) {
        throw std::runtime_error(
            "run_paged_batch_decode_generation requires paged KV cache"
        );
    }

    if (options.num_requests <= 0) {
        throw std::runtime_error(
            "run_paged_batch_decode_generation num_requests must be positive"
        );
    }

    if (gen_options.max_new_tokens == 0) {
        return 0;
    }

    const int64_t tokens_per_request =
        static_cast<int64_t>(input_ids.size()) +
        gen_options.max_new_tokens +
        16;

    const int64_t max_total_tokens =
        tokens_per_request *
        static_cast<int64_t>(options.num_requests);

    PagedGenerateEngine engine(
        model,
        gen_options,
        max_total_tokens,
        options.page_size
    );

    std::vector<int64_t> request_ids;
    request_ids.reserve(
        static_cast<size_t>(options.num_requests)
    );

    for (int32_t i = 0; i < options.num_requests; ++i) {
        const int64_t request_id =
            engine.add_request(
                input_ids,
                gen_options.max_new_tokens,
                gen_options.eos_token_id
            );

        request_ids.push_back(request_id);
    }

    for (int64_t request_id : request_ids) {
        if (!engine.finished(request_id)) {
            engine.prefill(request_id);
        }
    }

    bool all_finished = false;

    while (!all_finished) {
        all_finished = true;

        std::vector<int64_t> active_request_ids;
        active_request_ids.reserve(request_ids.size());

        for (int64_t request_id : request_ids) {
            if (!engine.finished(request_id)) {
                all_finished = false;
                active_request_ids.push_back(request_id);
            }
        }

        if (!active_request_ids.empty()) {
            engine.decode_batch(active_request_ids);
        }
    }

    int64_t total_new_tokens = 0;

    for (int64_t request_id : request_ids) {
        const GenerationRequest& req =
            engine.request(request_id);

        if (req.input_ids.size() < input_ids.size()) {
            throw std::runtime_error(
                "run_paged_batch_decode_generation invalid output length"
            );
        }

        total_new_tokens +=
            static_cast<int64_t>(
                req.input_ids.size() - input_ids.size()
            );
    }

    for (int64_t request_id : request_ids) {
        engine.release_request(request_id);
    }

    return total_new_tokens;
}

int64_t count_new_tokens(
    const std::vector<int32_t>& input_ids,
    const std::vector<int32_t>& output_ids
) {
    if (output_ids.size() < input_ids.size()) {
        throw std::runtime_error(
            "benchmark_generate_greedy: invalid output length"
        );
    }

    return static_cast<int64_t>(
        output_ids.size() - input_ids.size()
    );
}

}  // namespace

GenerationBenchmarkResult benchmark_generate_greedy(
    const Qwen3ForCausalLM& model,
    const std::vector<int32_t>& input_ids,
    const GenerationBenchmarkOptions& options
) {
    if (!model.initialized()) {
        throw std::runtime_error(
            "benchmark_generate_greedy: model is not initialized"
        );
    }

    if (input_ids.empty()) {
        throw std::runtime_error(
            "benchmark_generate_greedy: input_ids is empty"
        );
    }

    if (options.num_requests <= 0) {
        throw std::runtime_error(
            "benchmark_generate_greedy: num_requests must be positive"
        );
    }

    if (options.warmup_requests < 0) {
        throw std::runtime_error(
            "benchmark_generate_greedy: warmup_requests must be non-negative"
        );
    }

    if (options.max_new_tokens < 0) {
        throw std::runtime_error(
            "benchmark_generate_greedy: max_new_tokens must be non-negative"
        );
    }

    if (options.use_kv_cache && options.use_paged_kv_cache) {
        throw std::runtime_error(
            "benchmark_generate_greedy: cannot enable both KV cache and Paged KV cache"
        );
    }

    if (options.use_paged_kv_cache && options.page_size <= 0) {
        throw std::runtime_error(
            "benchmark_generate_greedy: page_size must be positive"
        );
    }

    if (options.interleaved && !options.use_paged_kv_cache) {
        throw std::runtime_error(
            "benchmark_generate_greedy: interleaved benchmark currently requires Paged KV cache"
        );
    }

    if (options.batch_decode && !options.use_paged_kv_cache) {
        throw std::runtime_error(
            "benchmark_generate_greedy: batch decode benchmark requires Paged KV cache"
        );
    }

    if (options.interleaved && options.batch_decode) {
        throw std::runtime_error(
            "benchmark_generate_greedy: cannot enable both interleaved and batch decode benchmark modes"
        );
    }

    GreedyGenerateOptions gen_options;
    gen_options.max_new_tokens = options.max_new_tokens;
    gen_options.eos_token_id = options.eos_token_id;
    gen_options.verbose = options.verbose;
    gen_options.device = options.device;

    for (int32_t i = 0; i < options.warmup_requests; ++i) {
        if (options.batch_decode) {
            (void)run_paged_batch_decode_generation(
                model,
                input_ids,
                options,
                gen_options
            );
        } else if (options.interleaved) {
            (void)run_paged_interleaved_generation(
                model,
                input_ids,
                options,
                gen_options
            );
        } else {
            (void)run_one_generation(
                model,
                input_ids,
                options,
                gen_options
            );
        }
    }

    sync_device_if_needed(options.device);

    const double start_ms = now_ms();

    int64_t total_new_tokens = 0;

    if (options.batch_decode) {
        total_new_tokens =
            run_paged_batch_decode_generation(
                model,
                input_ids,
                options,
                gen_options
            );
    } else if (options.interleaved) {
        total_new_tokens =
            run_paged_interleaved_generation(
                model,
                input_ids,
                options,
                gen_options
            );
    } else {
        for (int32_t i = 0; i < options.num_requests; ++i) {
            const std::vector<int32_t> output_ids =
                run_one_generation(
                    model,
                    input_ids,
                    options,
                    gen_options
                );

            total_new_tokens +=
                count_new_tokens(
                    input_ids,
                    output_ids
                );
        }
    }

    sync_device_if_needed(options.device);

    const double end_ms = now_ms();

    const double elapsed_ms = end_ms - start_ms;
    const double elapsed_s = elapsed_ms / 1000.0;

    GenerationBenchmarkResult result;

    result.num_requests = options.num_requests;
    result.prompt_tokens =
        static_cast<int64_t>(input_ids.size());
    result.max_new_tokens = options.max_new_tokens;
    result.total_new_tokens = total_new_tokens;

    result.total_time_ms = elapsed_ms;

    result.use_kv_cache = options.use_kv_cache;
    result.use_paged_kv_cache = options.use_paged_kv_cache;
    result.interleaved = options.interleaved;
    result.batch_decode = options.batch_decode;
    result.page_size = options.page_size;

    if (total_new_tokens > 0 && elapsed_s > 0.0) {
        result.tok_per_sec =
            static_cast<double>(total_new_tokens) /
            elapsed_s;

        result.ms_per_token =
            elapsed_ms /
            static_cast<double>(total_new_tokens);
    }

    return result;
}

void print_generation_benchmark_result(
    const GenerationBenchmarkResult& result
) {
    std::cout << "\nGeneration benchmark result\n";

    std::cout << "  Mode:             "
              << benchmark_mode_name(
                     result.use_kv_cache,
                     result.use_paged_kv_cache,
                     result.interleaved,
                     result.batch_decode
                 )
              << "\n";

    if (result.use_paged_kv_cache) {
        std::cout << "  Page size:        "
                  << result.page_size
                  << "\n";
    }

    std::cout << "  Requests:         "
              << result.num_requests
              << "\n";

    std::cout << "  Prompt tokens:    "
              << result.prompt_tokens
              << "\n";

    std::cout << "  Max new tokens:   "
              << result.max_new_tokens
              << "\n";

    std::cout << "  Total new tokens: "
              << result.total_new_tokens
              << "\n";

    std::cout << "  Total time:       "
              << result.total_time_ms
              << " ms\n";

    std::cout << "  Throughput:       "
              << result.tok_per_sec
              << " tok/s\n";

    std::cout << "  Latency/token:    "
              << result.ms_per_token
              << " ms/token\n";
}

}  // namespace lite_llm

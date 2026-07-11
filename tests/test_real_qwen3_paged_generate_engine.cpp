// tests/test_real_qwen3_paged_generate_engine.cpp

#include "engine/paged_generate_engine.hpp"
#include "model/model_config.hpp"
#include "model/qwen3.hpp"
#include "runtime/generation.hpp"
#include "weights/weight_loader.hpp"
#include "weights/weight_map.hpp"

#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lite_llm;

static std::string getenv_or_empty(const char* name) {
    const char* value = std::getenv(name);

    if (value == nullptr) {
        return "";
    }

    return std::string(value);
}

static Qwen3ForCausalLM load_real_qwen3_model(
    const std::string& model_dir,
    const std::string& weights_dir,
    Device weight_device
) {
    const std::string config_path =
        model_dir + "/config.json";

    ModelConfig config =
        load_model_config(config_path);

    Qwen3ForCausalLM model(config);

    WeightLoaderOptions weight_options;
    weight_options.device = weight_device;
    weight_options.index_filename = "weights_index.json";

    WeightMap weights =
        load_weight_map_from_directory(
            weights_dir,
            weight_options
        );

    model.load_weights(weights);

    if (!model.initialized()) {
        throw std::runtime_error(
            "real Qwen3 model is not initialized"
        );
    }

    return model;
}

static GreedyGenerateOptions make_options(
    Device device,
    int64_t max_new_tokens
) {
    GreedyGenerateOptions options;

    options.device = device;
    options.max_new_tokens = max_new_tokens;
    options.eos_token_id = -1;
    options.verbose = false;

    return options;
}

static std::vector<int32_t> generate_with_paged_engine(
    const Qwen3ForCausalLM& model,
    const std::vector<int32_t>& prompt_ids,
    const GreedyGenerateOptions& options,
    int64_t page_size
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

static void print_tokens(
    const std::string& name,
    const std::vector<int32_t>& tokens
) {
    std::cout << name << ":";

    for (int32_t token : tokens) {
        std::cout << " " << token;
    }

    std::cout << std::endl;
}

static void expect_same_tokens(
    const std::vector<int32_t>& lhs,
    const std::vector<int32_t>& rhs
) {
    if (lhs == rhs) {
        return;
    }

    print_tokens("contiguous", lhs);
    print_tokens("paged", rhs);

    throw std::runtime_error(
        "real Qwen3 paged output mismatch with contiguous KV cache"
    );
}

static void run_real_qwen3_test(Device device) {
    const std::string model_dir =
        getenv_or_empty("LITELLM_QWEN3_MODEL_DIR");

    const std::string weights_dir =
        getenv_or_empty("LITELLM_QWEN3_WEIGHTS_DIR");

    if (model_dir.empty() || weights_dir.empty()) {
        std::cout << "[test_real_qwen3_paged_generate_engine] skipped: "
                  << "set LITELLM_QWEN3_MODEL_DIR and LITELLM_QWEN3_WEIGHTS_DIR"
                  << std::endl;
        return;
    }

    std::cout << "[test_real_qwen3_paged_generate_engine] device="
              << (device == Device::CUDA ? "cuda" : "cpu")
              << std::endl;

    Qwen3ForCausalLM model =
        load_real_qwen3_model(
            model_dir,
            weights_dir,
            device
        );

    /*
     * 这里直接使用合法 token id，不依赖 tokenizer。
     * 目的只是验证 contiguous KVCache 和 paged KVCache 路径输出一致。
     */
    const std::vector<int32_t> prompt_ids = {
        9707, 374, 264, 1296, 13
    };

    constexpr int64_t max_new_tokens = 8;
    constexpr int64_t page_size = 16;

    GreedyGenerateOptions options =
        make_options(
            device,
            max_new_tokens
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
            page_size
        );

    print_tokens("contiguous", contiguous_output);
    print_tokens("paged", paged_output);

    expect_same_tokens(
        contiguous_output,
        paged_output
    );

    std::cout << "[test_real_qwen3_paged_generate_engine] passed"
              << std::endl;
}

int main() {
    try {
        run_real_qwen3_test(Device::CUDA);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[test_real_qwen3_paged_generate_engine] failed: "
                  << e.what()
                  << std::endl;
        return 1;
    }
}
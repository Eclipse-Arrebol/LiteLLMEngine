#include "runtime/args.hpp"
#include "runtime/model_downloader.hpp"
#include "runtime/token_ids.hpp"
#include "runtime/generation.hpp"

#include "model/model_config.hpp"
#include "model/qwen3.hpp"

#include "weights/weight_loader.hpp"

#include "core/device.hpp"

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    try {
        lite_llm::Args args = lite_llm::parse_args(argc, argv);

        const std::string device_arg = args.device.value_or("auto");
        const lite_llm::Device device = lite_llm::resolve_device(device_arg);

        std::cout << "LiteLLMEngine\n";
        std::cout << "Device: " << lite_llm::device_to_string(device) << "\n";

        std::cout << "Config:\n";
        std::cout << "  Model:       " << args.model << "\n";
        std::cout << "  Prompt:      " << args.prompt << "\n";
        std::cout << "  Max tokens:  " << args.max_tokens << "\n";
        std::cout << "  Temperature: " << args.temperature << "\n";
        std::cout << "  Top-k:       " << args.top_k << "\n";
        std::cout << "  Top-p:       " << args.top_p << "\n";
        std::cout << "  Device arg:  " << device_arg << "\n";
        std::cout << "  EOS token:   "
          << (args.eos_token_id.has_value()
              ? std::to_string(args.eos_token_id.value())
              : "<none>")
          << "\n";

        if (!args.input_ids.has_value()) {
            std::cout << "\nNo --input-ids provided.\n";
            std::cout << "Tokenizer is not implemented yet, so prompt text cannot be encoded directly.\n";
            std::cout << "Please pass comma-separated token ids for now.\n";
            std::cout << "\nExample:\n";
            std::cout << "  ./build/LiteLLMEngine \\\n";
            std::cout << "    --model /root/rivermind-data/Qwen_Qwen3-0.6B \\\n";
            std::cout << "    --input-ids \"151646\" \\\n";
            std::cout << "    --max-tokens 1 \\\n";
            std::cout << "    --device cuda\n";
            return 0;
        }

        const std::vector<int32_t> input_ids =
            lite_llm::parse_token_ids(args.input_ids.value());

        std::cout << "\nInput token ids:\n";
        std::cout << "  " << lite_llm::format_token_ids(input_ids) << "\n";

        std::cout << "\nLoading model metadata...\n";

        auto model_files = lite_llm::ensure_model_files(args.model);
        auto model_config = lite_llm::load_model_config(model_files.config_path);

        lite_llm::print_model_config(model_config);

        const std::string weight_dir =
            model_files.model_dir + "/converted_weights";

        std::cout << "\nLoading converted weights from:\n";
        std::cout << "  " << weight_dir << "\n";

        lite_llm::WeightLoaderOptions weight_options;
        weight_options.device = device;

        lite_llm::WeightMap weights =
            lite_llm::load_weight_map_from_directory(
                weight_dir,
                weight_options
            );

        std::cout << "Weights loaded: " << weights.size() << "\n";

        std::cout << "\nBuilding Qwen3ForCausalLM...\n";

        lite_llm::Qwen3ForCausalLM model(model_config);
        model.load_weights(weights);

        if (!weights.empty()) {
            throw std::runtime_error(
                "Some weights were not consumed by Qwen3ForCausalLM"
            );
        }

        if (!model.initialized()) {
            throw std::runtime_error(
                "Qwen3ForCausalLM is not initialized after load_weights"
            );
        }

        std::cout << "Model initialized.\n";

        if (args.temperature != 0.0f) {
            std::cout << "\nWarning: sampling is not implemented yet.\n";
            std::cout << "Current generation uses greedy argmax only.\n";
        }

        lite_llm::GreedyGenerateOptions gen_options;
        gen_options.max_new_tokens = args.max_tokens;
        gen_options.eos_token_id = args.eos_token_id.value_or(-1);
        gen_options.device = device;

        std::cout << "\nGenerating...\n";

        const std::vector<int32_t> generated_ids =
            lite_llm::generate_greedy(
                model,
                input_ids,
                gen_options
            );

        std::cout << "\nGenerated token ids:\n";
        std::cout << "  " << lite_llm::format_token_ids(generated_ids) << "\n";

        std::cout << "\nNew token ids:\n";

        if (generated_ids.size() <= input_ids.size()) {
            std::cout << "  <none>\n";
        } else {
            std::vector<int32_t> new_ids(
                generated_ids.begin() + static_cast<long>(input_ids.size()),
                generated_ids.end()
            );

            std::cout << "  " << lite_llm::format_token_ids(new_ids) << "\n";
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n\n";
        lite_llm::print_usage(argv[0]);
        return 1;
    }
}
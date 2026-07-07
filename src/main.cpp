#include "runtime/args.hpp"
#include "runtime/model_downloader.hpp"
#include "runtime/token_ids.hpp"
#include "runtime/generation.hpp"

#include "model/model_config.hpp"
#include "model/qwen3.hpp"

#include "weights/weight_loader.hpp"

#include "core/device.hpp"

#include "tokenizer/hf_tokenizer.hpp"
#include "tokenizer/qwen3_chat_template.hpp"

#include <exception>
#include <iostream>
#include <memory>
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
        std::cout << "  verbose:       " << args.verbose << "\n";
        std::cout << "  Device arg:  " << device_arg << "\n";
        std::cout << "  Eos Token Id:  " << args.eos_token_id.value_or(-1) << "\n";

        
        std::cout << "\nLoading model metadata...\n";

        auto model_files = lite_llm::ensure_model_files(args.model);
        auto model_config = lite_llm::load_model_config(model_files.config_path);

        lite_llm::print_model_config(model_config);

        std::vector<int32_t> input_ids;
        std::unique_ptr<lite_llm::HFTokenizer> tokenizer;

        if (args.input_ids.has_value()) {
            input_ids = lite_llm::parse_token_ids(args.input_ids.value());

            std::cout << "\nInput token ids:\n";
            std::cout << "  " << lite_llm::format_token_ids(input_ids) << "\n";
            std::cout << "Input length: " << input_ids.size() << "\n";
        } else {
            const std::string tokenizer_path =
                model_files.model_dir + "/tokenizer.json";

            std::cout << "\nLoading tokenizer from:\n";
            std::cout << "  " << tokenizer_path << "\n";

            tokenizer = std::make_unique<lite_llm::HFTokenizer>(
                tokenizer_path
            );

            const std::string prompt_text =
                lite_llm::apply_qwen3_chat_template(
                    args.prompt,
                    false
                );

            input_ids = tokenizer->encode(prompt_text);

            std::cout << "\nInput prompt:\n";
            std::cout << args.prompt << "\n";

            std::cout << "\nEncoded input token ids:\n";
            std::cout << "  " << lite_llm::format_token_ids(input_ids) << "\n";
            std::cout << "Input length: " << input_ids.size() << "\n";
        }

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
        gen_options.verbose = args.verbose;

        std::cout << "\nGenerating...\n";

        const std::vector<int32_t> generated_ids =
            lite_llm::generate_greedy(
                model,
                input_ids,
                gen_options
            );

        std::vector<int32_t> new_ids;

        if (generated_ids.size() > input_ids.size()) {
            new_ids.assign(
                generated_ids.begin() + static_cast<long>(input_ids.size()),
                generated_ids.end()
            );
        }

        std::cout << "\nGenerated token ids:\n";
        std::cout << "  " << lite_llm::format_token_ids(generated_ids) << "\n";

        std::cout << "\nNew token ids:\n";

        if (new_ids.empty()) {
            std::cout << "  <none>\n";
        } else {
            std::cout << "  " << lite_llm::format_token_ids(new_ids) << "\n";
        }

        if (tokenizer) {
            std::cout << "\nDecoded new text:\n";
            std::cout << tokenizer->decode(new_ids) << "\n";
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n\n";
        lite_llm::print_usage(argv[0]);
        return 1;
    }
}
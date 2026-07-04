#include "runtime/args.hpp"
#include "runtime/model_downloader.hpp"
#include "model/model_config.hpp"

#include <exception>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    try {
        lite_llm::Args args = lite_llm::parse_args(argc, argv);
        std::string device = args.device.value_or("auto");

        std::cout << "lite_llm config:\n";
        std::cout << "  Model:       " << args.model << "\n";
        std::cout << "  Prompt:      " << args.prompt << "\n";
        std::cout << "  Max tokens:  " << args.max_tokens << "\n";
        std::cout << "  Temperature: " << args.temperature << "\n";
        std::cout << "  Top-k:       " << args.top_k << "\n";
        std::cout << "  Top-p:       " << args.top_p << "\n";
        std::cout << "  Device:      " << device << "\n";

        auto model_files = lite_llm::ensure_model_files(args.model);
        auto model_config = lite_llm::load_model_config(model_files.config_path);
        lite_llm::print_model_config(model_config);
        // TODO:
        // Engine engine(args);
        // engine.generate();

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n\n";
        lite_llm::print_usage(argv[0]);
        return 1;
    }



}
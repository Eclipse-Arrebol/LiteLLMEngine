#include <iostream>


#include "runtime/args.hpp"

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
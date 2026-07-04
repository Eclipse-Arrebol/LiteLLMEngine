#pragma once

#include <optional>
#include <string>

namespace lite_llm {

    struct Args{
        std::string model;
        std::string prompt = "What is the capital of France?";
        int max_tokens = 128;
        float temperature = 0.7f;
        int top_k = 50;
        float top_p = 0.9f;
        std::optional<std::string> device = std::nullopt;
    };

    Args parse_args(int argc, char** argv);

    void print_usage(const char* program);

}
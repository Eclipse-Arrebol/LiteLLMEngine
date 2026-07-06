#pragma once

#include <optional>
#include <string>

namespace lite_llm {

struct Args {
    std::string model;
    std::string prompt = "What is the capital of France?";

    // 逗号分隔的 token id 字符串，例如 "151646,8948,198"
    // 这是 tokenizer 没实现前的过渡入口
    std::optional<std::string> input_ids = std::nullopt;
    std::optional<int> eos_token_id = std::nullopt;

    int max_tokens = 128;
    float temperature = 0.7f;
    int top_k = 50;
    float top_p = 0.9f;
    std::optional<std::string> device = std::nullopt;
};

Args parse_args(int argc, char** argv);

void print_usage(const char* program);

}  // namespace lite_llm
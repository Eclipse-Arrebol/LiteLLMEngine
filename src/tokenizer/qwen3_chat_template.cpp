#include "tokenizer/qwen3_chat_template.hpp"

#include <string>

namespace lite_llm {

std::string apply_qwen3_chat_template(
    const std::string& user_prompt,
    bool enable_thinking
) {
    std::string text;

    text += "<|im_start|>user\n";
    text += user_prompt;
    text += "<|im_end|>\n";
    text += "<|im_start|>assistant\n";

    if (enable_thinking) {
        text += "<think>\n";
    } else {
        text += "<think>\n\n</think>\n\n";
    }

    return text;
}

}  // namespace lite_llm
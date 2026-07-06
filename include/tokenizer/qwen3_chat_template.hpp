#pragma once

#include <string>

namespace lite_llm {

std::string apply_qwen3_chat_template(
    const std::string& user_prompt,
    bool enable_thinking = false
);

}  // namespace lite_llm
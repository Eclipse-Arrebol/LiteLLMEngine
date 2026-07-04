#pragma once

#include <string>

namespace lite_llm {

struct ModelFiles {
    std::string model_dir;
    std::string config_path;
    std::string tokenizer_path;
    std::string tokenizer_config_path;
    std::string generation_config_path;
};

ModelFiles ensure_model_files(const std::string& model);

} // namespace lite_llm
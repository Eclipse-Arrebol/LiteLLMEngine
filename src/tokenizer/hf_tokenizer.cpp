#include "tokenizer/hf_tokenizer.hpp"

#include "tokenizers_cpp.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace lite_llm {

namespace {

std::string load_file_as_string(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);

    if (!ifs) {
        throw std::runtime_error("Failed to open tokenizer file: " + path);
    }

    std::ostringstream oss;
    oss << ifs.rdbuf();

    return oss.str();
}

}  // namespace

HFTokenizer::HFTokenizer(const std::string& tokenizer_json_path) {
    const std::string blob = load_file_as_string(tokenizer_json_path);

    tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(blob);

    if (!tokenizer_) {
        throw std::runtime_error(
            "Failed to create HF tokenizer from: " + tokenizer_json_path
        );
    }
    
}


HFTokenizer::~HFTokenizer() = default;

HFTokenizer::HFTokenizer(HFTokenizer&&) noexcept = default;

HFTokenizer& HFTokenizer::operator=(HFTokenizer&&) noexcept = default;

std::vector<int32_t> HFTokenizer::encode(const std::string& text) const {
    std::vector<int> ids = tokenizer_->Encode(text);

    std::vector<int32_t> out;
    out.reserve(ids.size());

    for (int id : ids) {
        out.push_back(static_cast<int32_t>(id));
    }

    return out;
}

std::string HFTokenizer::decode(const std::vector<int32_t>& ids) const {
    std::vector<int> token_ids;
    token_ids.reserve(ids.size());

    for (int32_t id : ids) {
        token_ids.push_back(static_cast<int>(id));
    }

    return tokenizer_->Decode(token_ids);
}

int32_t HFTokenizer::token_to_id(const std::string& token) const {
    return static_cast<int32_t>(tokenizer_->TokenToId(token));
}

std::string HFTokenizer::id_to_token(int32_t token_id) const {
    return tokenizer_->IdToToken(static_cast<int>(token_id));
}

}  // namespace lite_llm
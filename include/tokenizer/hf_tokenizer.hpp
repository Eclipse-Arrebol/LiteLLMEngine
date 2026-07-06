#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tokenizers {
class Tokenizer;
}

namespace lite_llm {

class HFTokenizer {
public:
    explicit HFTokenizer(const std::string& tokenizer_json_path);

    ~HFTokenizer();

    HFTokenizer(const HFTokenizer&) = delete;
    HFTokenizer& operator=(const HFTokenizer&) = delete;

    HFTokenizer(HFTokenizer&&) noexcept;
    HFTokenizer& operator=(HFTokenizer&&) noexcept;

    std::vector<int32_t> encode(const std::string& text) const;

    std::string decode(const std::vector<int32_t>& ids) const;

    int32_t token_to_id(const std::string& token) const;

    std::string id_to_token(int32_t token_id) const;

private:
    std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
};

}  // namespace lite_llm
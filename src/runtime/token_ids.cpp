#include "runtime/token_ids.hpp"


#include <cctype>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace lite_llm {

namespace {

std::string trim(const std::string& s) {
    size_t begin = 0;
    while (begin < s.size() &&
           std::isspace(static_cast<unsigned char>(s[begin]))) {
        ++begin;
    }

    size_t end = s.size();
    while (end > begin &&
           std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }

    return s.substr(begin, end - begin);
}

int32_t parse_one_token_id(const std::string& part) {
    const std::string value = trim(part);

    if (value.empty()) {
        throw std::runtime_error("Empty token id");
    }

    size_t pos = 0;
    long long parsed = 0;

    try {
        parsed = std::stoll(value, &pos, 10);
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid token id: " + value);
    }

    if (pos != value.size()) {
        throw std::runtime_error("Invalid token id: " + value);
    }

    if (parsed < 0) {
        throw std::runtime_error("Token id must be non-negative: " + value);
    }

    if (parsed > static_cast<long long>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error("Token id too large: " + value);
    }

    return static_cast<int32_t>(parsed);
}

}  // namespace

std::vector<int32_t> parse_token_ids(const std::string& text) {
    const std::string trimmed = trim(text);

    if (trimmed.empty()) {
        throw std::runtime_error("Token ids string must not be empty");
    }

    std::vector<int32_t> token_ids;

    size_t start = 0;

    while (start <= trimmed.size()) {
        const size_t comma = trimmed.find(',', start);

        if (comma == std::string::npos) {
            token_ids.push_back(parse_one_token_id(trimmed.substr(start)));
            break;
        }

        token_ids.push_back(
            parse_one_token_id(trimmed.substr(start, comma - start))
        );

        start = comma + 1;
    }

    if (token_ids.empty()) {
        throw std::runtime_error("No token ids parsed");
    }

    return token_ids;
}

std::string format_token_ids(const std::vector<int32_t>& token_ids) {
    std::string out;

    for (size_t i = 0; i < token_ids.size(); ++i) {
        if (i > 0) {
            out += ",";
        }

        out += std::to_string(token_ids[i]);
    }

    return out;
}

}  // namespace lite_llm
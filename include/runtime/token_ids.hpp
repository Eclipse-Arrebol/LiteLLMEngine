#pragma once

#include <cstdint>
#include <string>
#include <vector>

/**
 * @brief 主要是实现字符串的token和vector token的互相转换
 * 
 */

namespace lite_llm {

std::vector<int32_t> parse_token_ids(const std::string& text);

std::string format_token_ids(const std::vector<int32_t>& token_ids);

}  // namespace lite_llm
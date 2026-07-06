#include "runtime/token_ids.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lite_llm;

namespace {

void check_ids_equal(
    const std::vector<int32_t>& actual,
    const std::vector<int32_t>& expected
) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error(
            "size mismatch: got " + std::to_string(actual.size()) +
            ", expected " + std::to_string(expected.size())
        );
    }

    for (size_t i = 0; i < expected.size(); ++i) {
        if (actual[i] != expected[i]) {
            throw std::runtime_error(
                "id mismatch at " + std::to_string(i) +
                ": got " + std::to_string(actual[i]) +
                ", expected " + std::to_string(expected[i])
            );
        }
    }
}

void test_parse_token_ids_basic() {
    std::cout << "[test_parse_token_ids_basic] start\n";

    const std::vector<int32_t> ids = parse_token_ids("1,2,3");

    check_ids_equal(ids, {1, 2, 3});

    std::cout << "[test_parse_token_ids_basic] passed\n";
}

void test_parse_token_ids_spaces() {
    std::cout << "[test_parse_token_ids_spaces] start\n";

    const std::vector<int32_t> ids = parse_token_ids("  151646, 8948 , 198  ");

    check_ids_equal(ids, {151646, 8948, 198});

    std::cout << "[test_parse_token_ids_spaces] passed\n";
}

void test_parse_token_ids_single() {
    std::cout << "[test_parse_token_ids_single] start\n";

    const std::vector<int32_t> ids = parse_token_ids("42");

    check_ids_equal(ids, {42});

    std::cout << "[test_parse_token_ids_single] passed\n";
}

void test_format_token_ids() {
    std::cout << "[test_format_token_ids] start\n";

    const std::string text = format_token_ids({151646, 8948, 198});

    if (text != "151646,8948,198") {
        throw std::runtime_error(
            "format_token_ids failed: got " + text
        );
    }

    std::cout << "[test_format_token_ids] passed\n";
}

void expect_parse_error(const std::string& text) {
    bool caught = false;

    try {
        (void)parse_token_ids(text);
    } catch (const std::runtime_error&) {
        caught = true;
    }

    if (!caught) {
        throw std::runtime_error("Expected parse_token_ids error for: " + text);
    }
}

void test_parse_token_ids_errors() {
    std::cout << "[test_parse_token_ids_errors] start\n";

    expect_parse_error("");
    expect_parse_error("   ");
    expect_parse_error("1,,2");
    expect_parse_error(",1,2");
    expect_parse_error("1,2,");
    expect_parse_error("1,a,2");
    expect_parse_error("-1,2");
    expect_parse_error("1,2.5");

    std::cout << "[test_parse_token_ids_errors] passed\n";
}

}  // namespace

int main() {
    try {
        test_parse_token_ids_basic();
        test_parse_token_ids_spaces();
        test_parse_token_ids_single();
        test_format_token_ids();
        test_parse_token_ids_errors();
    } catch (const std::exception& e) {
        std::cerr << "[test_token_ids] failed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "[test_token_ids] all passed\n";
    return 0;
}
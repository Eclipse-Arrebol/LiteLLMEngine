#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

namespace lite_llm {

enum class DType {
    FP32,
    FP16,
    INT32
};

inline size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::FP32:
            return 4;
        case DType::FP16:
            return 2;
        case DType::INT32:
            return 4;
        default:
            throw std::runtime_error("Unknown dtype");
    }
}

inline std::string dtype_to_string(DType dtype) {
    switch (dtype) {
        case DType::FP32:
            return "fp32";
        case DType::FP16:
            return "fp16";
        case DType::INT32:
            return "int32";
        default:
            return "unknown";
    }
}

} // namespace lite_llm
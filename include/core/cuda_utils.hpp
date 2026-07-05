#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace lite_llm {

inline void cuda_check_impl(
    cudaError_t err,
    const char* expr,
    const char* file,
    int line
) {
    if (err == cudaSuccess) {
        return;
    }

    throw std::runtime_error(
        std::string("CUDA error: ") +
        cudaGetErrorString(err) +
        "\n  expression: " + expr +
        "\n  file: " + file +
        "\n  line: " + std::to_string(line)
    );
}

template <typename T>
inline constexpr T cdiv(T a, T b) {
    static_assert(std::is_integral_v<T>, "cdiv only supports integral types");
    return (a + b - 1) / b;
}

inline dim3 cuda_make_1d_grid(int64_t n, int threads = 256) {
    return dim3(static_cast<unsigned int>(cdiv<int>(n, threads)));
}

inline dim3 cuda_make_1d_block(int threads = 256) {
    return dim3(static_cast<unsigned int>(threads));
}

} // namespace lite_llm

#define CUDA_CHECK(expr) \
    ::lite_llm::cuda_check_impl((expr), #expr, __FILE__, __LINE__)

#define CUDA_KERNEL_CHECK() \
    CUDA_CHECK(cudaGetLastError())
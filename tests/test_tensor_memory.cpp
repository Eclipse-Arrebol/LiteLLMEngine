#include "core/tensor.hpp"

#include <cuda_runtime.h>


#include <iostream>
#include <stdexcept>

using namespace lite_llm;

namespace {

size_t cuda_free_memory() {
    size_t free_bytes = 0;
    size_t total_bytes = 0;

    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);

    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return free_bytes;
}

void test_cuda_tensor_release_loop() {
    std::cout << "[test_cuda_tensor_release_loop] start\n";

    const size_t before = cuda_free_memory();

    for (int i = 0; i < 1000; ++i) {
        Tensor t({1024, 1024}, DType::FP32, Device::CUDA);
        t.zero_();
    }

    cudaDeviceSynchronize();

    const size_t after = cuda_free_memory();

    const long long diff =
        static_cast<long long>(before) - static_cast<long long>(after);

    std::cout << "CUDA free before: " << before << "\n";
    std::cout << "CUDA free after:  " << after << "\n";
    std::cout << "Diff bytes:       " << diff << "\n";

    // CUDA allocator/driver 可能有少量上下文开销，不要卡得太死。
    // 这里允许 64MB 以内波动。
    const long long tolerance = 64LL * 1024LL * 1024LL;

    if (std::llabs(diff) > tolerance) {
        throw std::runtime_error("Possible CUDA memory leak in Tensor destructor");
    }

    std::cout << "[test_cuda_tensor_release_loop] passed\n";
}

void test_cuda_tensor_move_assignment_release_old_memory() {
    std::cout << "[test_cuda_tensor_move_assignment_release_old_memory] start\n";

    const size_t before = cuda_free_memory();

    for (int i = 0; i < 1000; ++i) {
        Tensor a({1024, 1024}, DType::FP32, Device::CUDA);
        Tensor b({512, 512}, DType::FP32, Device::CUDA);

        a = std::move(b);
    }

    cudaDeviceSynchronize();

    const size_t after = cuda_free_memory();

    const long long diff =
        static_cast<long long>(before) - static_cast<long long>(after);

    std::cout << "CUDA free before: " << before << "\n";
    std::cout << "CUDA free after:  " << after << "\n";
    std::cout << "Diff bytes:       " << diff << "\n";

    const long long tolerance = 64LL * 1024LL * 1024LL;

    if (std::llabs(diff) > tolerance) {
        throw std::runtime_error("Possible CUDA memory leak in Tensor move assignment");
    }

    std::cout << "[test_cuda_tensor_move_assignment_release_old_memory] passed\n";
}

}  // namespace

int main() {
    try {
        test_cuda_tensor_release_loop();
        test_cuda_tensor_move_assignment_release_old_memory();
    } catch (const std::exception& e) {
        std::cerr << "[test_tensor_memory] failed: " << e.what() << "\n";
        return 1;
    }

    std::cout << "[test_tensor_memory] all passed\n";
    return 0;
}
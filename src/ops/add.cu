#include "ops/add.hpp"
#include "core/device.hpp"
#include "core/tensor.hpp"
#include "core/dtype.hpp"
#include "core/cuda_utils.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace lite_llm {

namespace {

__global__ void add_fp32_kernel(
    const float* a,
    const float* b,
    float* out,
    int64_t n
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

void check_add_args(const Tensor& a, const Tensor& b, const Tensor& out) {
    if (a.dtype() != DType::FP32 || b.dtype() != DType::FP32 || out.dtype() != DType::FP32) {
        throw std::runtime_error("add only supports FP32 tensors");
    }

    if (a.device() != b.device() || a.device() != out.device()) {
        throw std::runtime_error("add requires all tensors on the same device");
    }

    if (a.shape() != b.shape() || a.shape() != out.shape()) {
        throw std::runtime_error("add requires tensors with the same shape");
    }

    if (a.empty() || b.empty() || out.empty()) {
        throw std::runtime_error("add got empty tensor");
    }
}

void add_cpu(const Tensor& a, const Tensor& b, Tensor& out) {
    const float* a_ptr = a.ptr<const float>();
    const float* b_ptr = b.ptr<const float>();
    float* out_ptr = out.ptr<float>();

    int64_t n = static_cast<int64_t>(a.numel());

    for (int64_t i = 0; i < n; ++i) {
        out_ptr[i] = a_ptr[i] + b_ptr[i];
    }
}

void add_cuda(const Tensor& a, const Tensor& b, Tensor& out) {
    int64_t n = static_cast<int64_t>(a.numel());

    int threads = 256;
    dim3 block = cuda_make_1d_block(threads);
    dim3 grid = cuda_make_1d_grid(n, threads);

    add_fp32_kernel<<<grid, block>>>(
        a.ptr<const float>(),
        b.ptr<const float>(),
        out.ptr<float>(),
        n
    );

    CUDA_KERNEL_CHECK();
}

} // namespace

void tensor_add(const Tensor& a, const Tensor& b, Tensor& out) {
    check_add_args(a, b, out);

    if (a.numel() == 0) {
        return;
    }

    if (a.device() == Device::CPU) {
        add_cpu(a, b, out);
    } else if (a.device() == Device::CUDA) {
        add_cuda(a, b, out);
    } else {
        throw std::runtime_error("Unsupported device in add()");
    }
}

} // namespace lite_llm
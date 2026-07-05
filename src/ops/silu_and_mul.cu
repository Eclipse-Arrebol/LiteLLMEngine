#include "ops/silu_and_mul.hpp"

#include "core/cuda_utils.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>

namespace lite_llm {

namespace {

void check_silu_and_mul_args(
    const Tensor& gate,
    const Tensor& up,
    const Tensor& output
) {
    if (gate.dtype() != DType::FP32) {
        throw std::runtime_error("silu_and_mul gate must be FP32");
    }

    if (up.dtype() != DType::FP32) {
        throw std::runtime_error("silu_and_mul up must be FP32");
    }

    if (output.dtype() != DType::FP32) {
        throw std::runtime_error("silu_and_mul output must be FP32");
    }

    if (gate.shape() != up.shape()) {
        throw std::runtime_error("silu_and_mul gate and up shape mismatch");
    }

    if (gate.shape() != output.shape()) {
        throw std::runtime_error("silu_and_mul output shape mismatch");
    }

    if (gate.device() != up.device() || gate.device() != output.device()) {
        throw std::runtime_error("silu_and_mul tensors must be on same device");
    }
}

float silu_cpu(float x) {
    return x / (1.0f + std::exp(-x));
}

void silu_and_mul_cpu(
    const Tensor& gate,
    const Tensor& up,
    Tensor& output
) {
    const float* gate_ptr = gate.ptr<float>();
    const float* up_ptr = up.ptr<float>();
    float* out_ptr = output.ptr<float>();

    const size_t n = gate.numel();

    for (size_t i = 0; i < n; ++i) {
        out_ptr[i] = silu_cpu(gate_ptr[i]) * up_ptr[i];
    }
}

__global__ void silu_and_mul_kernel(
    const float* gate,
    const float* up,
    float* output,
    int64_t n
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx >= n) {
        return;
    }

    const float x = gate[idx];
    const float silu = x / (1.0f + expf(-x));

    output[idx] = silu * up[idx];
}

void silu_and_mul_cuda(
    const Tensor& gate,
    const Tensor& up,
    Tensor& output
) {
    const int64_t n = static_cast<int64_t>(gate.numel());

    if (n == 0) {
        return;
    }

    dim3 block = cuda_make_1d_block(256);
    dim3 grid = cuda_make_1d_grid(n, block.x);

    silu_and_mul_kernel<<<grid, block>>>(
        gate.ptr<float>(),
        up.ptr<float>(),
        output.ptr<float>(),
        n
    );

    CUDA_KERNEL_CHECK();
}

}  // namespace

void silu_and_mul(
    const Tensor& gate,
    const Tensor& up,
    Tensor& output
) {
    check_silu_and_mul_args(gate, up, output);

    if (gate.device() == Device::CPU) {
        silu_and_mul_cpu(gate, up, output);
        return;
    }

    if (gate.device() == Device::CUDA) {
        silu_and_mul_cuda(gate, up, output);
        return;
    }

    throw std::runtime_error("Unsupported device for silu_and_mul");
}

}  // namespace lite_llm
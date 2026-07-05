#include "layers/rms_norm.hpp"
#include "core/cuda_utils.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <utility>
#include <string>

namespace lite_llm {

namespace{
void check_rms_norm_forward_args(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& output
) {
    if (input.dtype() != DType::FP32) {
        throw std::runtime_error("RMSNorm input must be FP32");
    }

    if (weight.dtype() != DType::FP32) {
        throw std::runtime_error("RMSNorm weight must be FP32");
    }

    if (output.dtype() != DType::FP32) {
        throw std::runtime_error("RMSNorm output must be FP32");
    }

    if (input.shape().size() != 2) {
        throw std::runtime_error("RMSNorm input must be 2D: [num_tokens, hidden_size]");
    }

    if (weight.shape().size() != 1) {
        throw std::runtime_error("RMSNorm weight must be 1D: [hidden_size]");
    }

    if (output.shape().size() != 2) {
        throw std::runtime_error("RMSNorm output must be 2D: [num_tokens, hidden_size]");
    }

    const int64_t num_tokens = input.shape()[0];
    const int64_t hidden_size = input.shape()[1];

    if (num_tokens < 0 || hidden_size <= 0) {
        throw std::runtime_error("RMSNorm invalid input shape");
    }

    if (weight.shape()[0] != hidden_size) {
        throw std::runtime_error("RMSNorm weight shape mismatch");
    }

    if (output.shape()[0] != num_tokens || output.shape()[1] != hidden_size) {
        throw std::runtime_error("RMSNorm output shape mismatch");
    }

    if (input.device() != weight.device() || input.device() != output.device()) {
        throw std::runtime_error("RMSNorm input, weight and output must be on same device");
    }
}


void rms_norm_forward_cpu(
    const Tensor& input,
    const Tensor& weight,
    Tensor& output,
    float eps
) {
    const float* x = input.ptr<float>();
    const float* w = weight.ptr<float>();
    float* out = output.ptr<float>();

    const int64_t num_tokens = input.shape()[0];
    const int64_t hidden_size = input.shape()[1];

    for (int64_t t = 0; t < num_tokens; ++t) {
        const float* x_row = x + t * hidden_size;
        float* out_row = out + t * hidden_size;

        float sum_sq = 0.0f;

        for (int64_t h = 0; h < hidden_size; ++h) {
            const float v = x_row[h];
            sum_sq += v * v;
        }

        const float mean_sq = sum_sq / static_cast<float>(hidden_size);
        const float inv_rms = 1.0f / std::sqrt(mean_sq + eps);

        for (int64_t h = 0; h < hidden_size; ++h) {
            out_row[h] = x_row[h] * inv_rms * w[h];
        }
    }
}

__global__ void rms_norm_forward_kernel(
    const float* input,
    const float* weight,
    float* output,
    int64_t num_tokens,
    int64_t hidden_size,
    float eps
) {
    constexpr int kBlockSize = 256;

    __shared__ float shared[kBlockSize];

    const int64_t token_index = static_cast<int64_t>(blockIdx.x);
    const int tid = threadIdx.x;

    if (token_index >= num_tokens) {
        return;
    }

    const float* x_row = input + token_index * hidden_size;
    float* out_row = output + token_index * hidden_size;

    float local_sum = 0.0f;

    for (int64_t h = tid; h < hidden_size; h += blockDim.x) {
        const float v = x_row[h];
        local_sum += v * v;
    }

    shared[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    const float inv_rms = rsqrtf(shared[0] / static_cast<float>(hidden_size) + eps);

    for (int64_t h = tid; h < hidden_size; h += blockDim.x) {
        out_row[h] = x_row[h] * inv_rms * weight[h];
    }
}


void rms_norm_forward_cuda(
    const Tensor& input,
    const Tensor& weight,
    Tensor& output,
    float eps
) {
    const int64_t num_tokens = input.shape()[0];
    const int64_t hidden_size = input.shape()[1];

    if (num_tokens == 0) {
        return;
    }

    constexpr int kBlockSize = 256;

    dim3 block(kBlockSize);
    dim3 grid(static_cast<unsigned int>(num_tokens));

    rms_norm_forward_kernel<<<grid, block>>>(
        input.ptr<float>(),
        weight.ptr<float>(),
        output.ptr<float>(),
        num_tokens,
        hidden_size,
        eps
    );

    CUDA_KERNEL_CHECK();
}


}


RMSNorm::RMSNorm(float eps)
    : eps_(eps) {
    if (eps_ <= 0.0f) {
        throw std::runtime_error("RMSNorm eps must be positive");
    }
}

RMSNorm::RMSNorm(Tensor weight, float eps)
    : WeightedUnaryLayer(std::move(weight)),
      eps_(eps) {
    if (eps_ <= 0.0f) {
        throw std::runtime_error("RMSNorm eps must be positive");
    }

    if (weight_.shape().size() != 1) {
        throw std::runtime_error("RMSNorm weight must be 1D");
    }

    if (weight_.dtype() != DType::FP32) {
        throw std::runtime_error("RMSNorm weight must be FP32");
    }

    hidden_size_ = weight_.shape()[0];

    if (hidden_size_ <= 0) {
        throw std::runtime_error("RMSNorm got invalid hidden_size");
    }
}

void RMSNorm::load_weight(Tensor weight) {
    if (weight.shape().size() != 1) {
        throw std::runtime_error("RMSNorm weight must be 1D");
    }

    if (weight.dtype() != DType::FP32) {
        throw std::runtime_error("RMSNorm weight must be FP32");
    }

    int64_t hidden_size = weight.shape()[0];

    if (hidden_size <= 0) {
        throw std::runtime_error("RMSNorm got invalid hidden_size");
    }

    weight_ = std::move(weight);
    hidden_size_ = hidden_size;
}

void RMSNorm::forward(const Tensor& input, Tensor& output) const {
    if (!initialized()) {
        throw std::runtime_error("RMSNorm::forward called before weight is initialized");
    }

    check_rms_norm_forward_args(input, weight_, output);

    if (input.device() == Device::CPU) {
        rms_norm_forward_cpu(input, weight_, output, eps_);
        return;
    }

    if (input.device() == Device::CUDA) {
        rms_norm_forward_cuda(input, weight_, output, eps_);
        return;
    }

    throw std::runtime_error("RMSNorm::forward not implemented yet");
}

} // namespace lite_llm
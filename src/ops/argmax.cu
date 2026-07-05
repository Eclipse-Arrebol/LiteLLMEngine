#include "ops/argmax.hpp"

#include "core/cuda_utils.hpp"

#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>
#include <stdexcept>

namespace lite_llm {

namespace {

void check_argmax_last_token_args(const Tensor& logits) {
    if (logits.dtype() != DType::FP32) {
        throw std::runtime_error("argmax_last_token logits must be FP32");
    }

    if (logits.shape().size() != 2) {
        throw std::runtime_error("argmax_last_token logits must be 2D: [num_tokens, vocab_size]");
    }

    const int64_t num_tokens = logits.shape()[0];
    const int64_t vocab_size = logits.shape()[1];

    if (num_tokens <= 0) {
        throw std::runtime_error("argmax_last_token num_tokens must be positive");
    }

    if (vocab_size <= 0) {
        throw std::runtime_error("argmax_last_token vocab_size must be positive");
    }

    if (vocab_size > static_cast<int64_t>(INT32_MAX)) {
        throw std::runtime_error("argmax_last_token vocab_size too large for int32 token id");
    }
}

int32_t argmax_last_token_cpu(const Tensor& logits) {
    const float* ptr = logits.ptr<float>();

    const int64_t num_tokens = logits.shape()[0];
    const int64_t vocab_size = logits.shape()[1];

    const float* last = ptr + (num_tokens - 1) * vocab_size;

    int32_t best_id = 0;
    float best_value = last[0];

    for (int64_t i = 1; i < vocab_size; ++i) {
        const float value = last[i];

        if (value > best_value) {
            best_value = value;
            best_id = static_cast<int32_t>(i);
        }
    }

    return best_id;
}

__global__ void argmax_last_token_kernel(
    const float* logits,
    int64_t num_tokens,
    int64_t vocab_size,
    int32_t* out_token_id
) {
    extern __shared__ unsigned char shared_raw[];

    float* shared_values = reinterpret_cast<float*>(shared_raw);
    int32_t* shared_indices = reinterpret_cast<int32_t*>(
        shared_values + blockDim.x
    );

    const int tid = threadIdx.x;

    const float* last = logits + (num_tokens - 1) * vocab_size;

    float local_best_value = -FLT_MAX;
    int32_t local_best_id = 0;

    for (int64_t i = tid; i < vocab_size; i += blockDim.x) {
        const float value = last[i];

        if (value > local_best_value) {
            local_best_value = value;
            local_best_id = static_cast<int32_t>(i);
        }
    }

    shared_values[tid] = local_best_value;
    shared_indices[tid] = local_best_id;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other_value = shared_values[tid + stride];
            const int32_t other_id = shared_indices[tid + stride];

            const float current_value = shared_values[tid];
            const int32_t current_id = shared_indices[tid];

            if (other_value > current_value ||
                (other_value == current_value && other_id < current_id)) {
                shared_values[tid] = other_value;
                shared_indices[tid] = other_id;
            }
        }

        __syncthreads();
    }

    if (tid == 0) {
        *out_token_id = shared_indices[0];
    }
}

int32_t argmax_last_token_cuda(const Tensor& logits) {
    const int64_t num_tokens = logits.shape()[0];
    const int64_t vocab_size = logits.shape()[1];

    constexpr int kBlockSize = 256;

    Tensor out({1}, DType::INT32, Device::CUDA);
    out.zero_();

    const size_t shared_bytes =
        static_cast<size_t>(kBlockSize) * sizeof(float) +
        static_cast<size_t>(kBlockSize) * sizeof(int32_t);

    argmax_last_token_kernel<<<1, kBlockSize, shared_bytes>>>(
        logits.ptr<float>(),
        num_tokens,
        vocab_size,
        out.ptr<int32_t>()
    );

    CUDA_KERNEL_CHECK();

    int32_t host_token_id = 0;
    out.copy_to_cpu(&host_token_id, sizeof(int32_t));

    return host_token_id;
}

}  // namespace

int32_t argmax_last_token(const Tensor& logits) {
    check_argmax_last_token_args(logits);

    if (logits.device() == Device::CPU) {
        return argmax_last_token_cpu(logits);
    }

    if (logits.device() == Device::CUDA) {
        return argmax_last_token_cuda(logits);
    }

    throw std::runtime_error("Unsupported device for argmax_last_token");
}

}  // namespace lite_llm
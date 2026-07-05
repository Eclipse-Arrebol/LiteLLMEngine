#include "layers/embedding.hpp"
#include "core/cuda_utils.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <utility>


namespace lite_llm {


namespace {
    void check_embedding_forward_args(
    const Tensor& input_ids,
    const Tensor& weight,
    const Tensor& output
) {
    if (input_ids.dtype() != DType::INT32) {
        throw std::runtime_error("Embedding input_ids must be INT32");
    }

    if (weight.dtype() != DType::FP32) {
        throw std::runtime_error("Embedding weight must be FP32");
    }

    if (output.dtype() != DType::FP32) {
        throw std::runtime_error("Embedding output must be FP32");
    }

    if (input_ids.shape().size() != 1) {
        throw std::runtime_error("Embedding input_ids must be 1D: [num_tokens]");
    }

    if (weight.shape().size() != 2) {
        throw std::runtime_error("Embedding weight must be 2D: [vocab_size, hidden_size]");
    }

    if (output.shape().size() != 2) {
        throw std::runtime_error("Embedding output must be 2D: [num_tokens, hidden_size]");
    }

    const int64_t num_tokens = input_ids.shape()[0];
    const int64_t vocab_size = weight.shape()[0];
    const int64_t hidden_size = weight.shape()[1];

    if (output.shape()[0] != num_tokens || output.shape()[1] != hidden_size) {
        throw std::runtime_error(
            "Embedding output shape mismatch, expected [num_tokens, hidden_size]"
        );
    }

    if (input_ids.device() != weight.device() || input_ids.device() != output.device()) {
        throw std::runtime_error("Embedding input_ids, weight and output must be on same device");
    }

    if (vocab_size <= 0 || hidden_size <= 0 || num_tokens < 0) {
        throw std::runtime_error("Embedding invalid shape");
    }
}


void embedding_forward_cpu(
    const Tensor& input_ids,
    const Tensor& weight,
    Tensor& output
) {
    const int32_t* ids = input_ids.ptr<int32_t>();
    const float* w = weight.ptr<float>();
    float* out = output.ptr<float>();

    const int64_t num_tokens = input_ids.shape()[0];
    const int64_t vocab_size = weight.shape()[0];
    const int64_t hidden_size = weight.shape()[1];

    for (int64_t t = 0; t < num_tokens; ++t) {
        const int32_t token_id = ids[t];

        if (token_id < 0 || token_id >= vocab_size) {
            throw std::runtime_error("Embedding token id out of range");
        }

        const float* src = w + static_cast<int64_t>(token_id) * hidden_size;
        float* dst = out + t * hidden_size;

        for (int64_t h = 0; h < hidden_size; ++h) {
            dst[h] = src[h];
        }
    }
}

__global__ void embedding_forward_kernel(
    const int32_t* input_ids,
    const float* weight,
    float* output,
    int64_t num_tokens,
    int64_t vocab_size,
    int64_t hidden_size,
    int* error_flag
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = num_tokens * hidden_size;

    if (idx >= total) {
        return;
    }

    const int64_t token_index = idx / hidden_size;
    const int64_t hidden_index = idx % hidden_size;

    const int32_t token_id = input_ids[token_index];

    if (token_id < 0 || token_id >= vocab_size) {
        atomicExch(error_flag, 1);
        return;
    }

    output[idx] = weight[static_cast<int64_t>(token_id) * hidden_size + hidden_index];
}

void embedding_forward_cuda(
    const Tensor& input_ids,
    const Tensor& weight,
    Tensor& output
) {
    const int64_t num_tokens = input_ids.shape()[0];
    const int64_t vocab_size = weight.shape()[0];
    const int64_t hidden_size = weight.shape()[1];

    const int64_t total = num_tokens * hidden_size;

    if (total == 0) {
        return;
    }

    Tensor error_flag({1}, DType::INT32, Device::CUDA);
    error_flag.zero_();

    dim3 block = cuda_make_1d_block(256);
    dim3 grid = cuda_make_1d_grid(total, block.x);

    embedding_forward_kernel<<<grid, block>>>(
        input_ids.ptr<int32_t>(),
        weight.ptr<float>(),
        output.ptr<float>(),
        num_tokens,
        vocab_size,
        hidden_size,
        error_flag.ptr<int>()
    );

    CUDA_KERNEL_CHECK();

    int host_error = 0;
    error_flag.copy_to_cpu(&host_error, sizeof(int));

    if (host_error != 0) {
        throw std::runtime_error("Embedding token id out of range");
    }
}



}




Embedding::Embedding(Tensor weight)
    : WeightedUnaryLayer(std::move(weight)) {
    if (weight_.shape().size() != 2) {
        throw std::runtime_error("Embedding weight must be 2D");
    }

    if (weight_.dtype() != DType::FP32) {
        throw std::runtime_error("Embedding weight must be FP32");
    }

    vocab_size_ = weight_.shape()[0];
    hidden_size_ = weight_.shape()[1];

    if (vocab_size_ <= 0 || hidden_size_ <= 0) {
        throw std::runtime_error("Embedding got invalid weight shape");
    }
}

void Embedding::load_weight(Tensor weight) {
    if (weight.shape().size() != 2) {
        throw std::runtime_error("Embedding weight must be 2D");
    }

    if (weight.dtype() != DType::FP32) {
        throw std::runtime_error("Embedding weight must be FP32");
    }

    int64_t vocab_size = weight.shape()[0];
    int64_t hidden_size = weight.shape()[1];

    if (vocab_size <= 0 || hidden_size <= 0) {
        throw std::runtime_error("Embedding got invalid weight shape");
    }

    weight_ = std::move(weight);
    vocab_size_ = vocab_size;
    hidden_size_ = hidden_size;
}

void Embedding::forward(const Tensor& input_ids, Tensor& output) const {
    if (!initialized()) {
        throw std::runtime_error("Embedding::forward called before weight is initialized");
    }

    check_embedding_forward_args(input_ids, weight_, output);

    if (input_ids.device() == Device::CPU) {
        embedding_forward_cpu(input_ids, weight_, output);
        return;
    }

    if (input_ids.device() == Device::CUDA) {
        embedding_forward_cuda(input_ids, weight_, output);
        return;
    }

    throw std::runtime_error("Embedding::forward not implemented yet");
}

} // namespace lite_llm
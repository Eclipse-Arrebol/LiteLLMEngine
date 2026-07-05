#include "layers/rotary.hpp"
#include "core/cuda_utils.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <cmath>

namespace lite_llm {

namespace{
void check_rotary_apply_args(
    const Tensor& q,
    const Tensor& k,
    const Tensor& position_ids,
    const Tensor& q_out,
    const Tensor& k_out,
    int64_t expected_head_dim
) {
    if (q.dtype() != DType::FP32 || k.dtype() != DType::FP32) {
        throw std::runtime_error("RotaryEmbedding q and k must be FP32");
    }

    if (q_out.dtype() != DType::FP32 || k_out.dtype() != DType::FP32) {
        throw std::runtime_error("RotaryEmbedding q_out and k_out must be FP32");
    }

    if (position_ids.dtype() != DType::INT32) {
        throw std::runtime_error("RotaryEmbedding position_ids must be INT32");
    }

    if (q.shape().size() != 3) {
        throw std::runtime_error("RotaryEmbedding q must be 3D: [num_tokens, num_heads, head_dim]");
    }

    if (k.shape().size() != 3) {
        throw std::runtime_error("RotaryEmbedding k must be 3D: [num_tokens, num_kv_heads, head_dim]");
    }

    if (position_ids.shape().size() != 1) {
        throw std::runtime_error("RotaryEmbedding position_ids must be 1D: [num_tokens]");
    }

    if (q_out.shape() != q.shape()) {
        throw std::runtime_error("RotaryEmbedding q_out shape mismatch");
    }

    if (k_out.shape() != k.shape()) {
        throw std::runtime_error("RotaryEmbedding k_out shape mismatch");
    }

    const int64_t num_tokens = q.shape()[0];
    const int64_t q_head_dim = q.shape()[2];
    const int64_t k_head_dim = k.shape()[2];

    if (k.shape()[0] != num_tokens) {
        throw std::runtime_error("RotaryEmbedding q and k num_tokens mismatch");
    }

    if (position_ids.shape()[0] != num_tokens) {
        throw std::runtime_error("RotaryEmbedding position_ids shape mismatch");
    }

    if (q_head_dim != expected_head_dim || k_head_dim != expected_head_dim) {
        throw std::runtime_error("RotaryEmbedding head_dim mismatch");
    }

    if (expected_head_dim <= 0 || expected_head_dim % 2 != 0) {
        throw std::runtime_error("RotaryEmbedding head_dim must be positive and even");
    }

    if (
        q.device() != k.device() ||
        q.device() != position_ids.device() ||
        q.device() != q_out.device() ||
        q.device() != k_out.device()
    ) {
        throw std::runtime_error("RotaryEmbedding tensors must be on same device");
    }
}

float rope_inv_freq(int64_t dim_index, int64_t head_dim, float rope_theta) {
    return std::pow(rope_theta, -static_cast<float>(2 * dim_index) / static_cast<float>(head_dim));
}

void apply_rotary_one_cpu(
    const float* input,
    const int32_t* position_ids,
    float* output,
    int64_t num_tokens,
    int64_t num_heads,
    int64_t head_dim,
    float rope_theta
) {
    const int64_t half_dim = head_dim / 2;

    for (int64_t token = 0; token < num_tokens; ++token) {
        const int32_t pos = position_ids[token];

        for (int64_t head = 0; head < num_heads; ++head) {
            const int64_t base = (token * num_heads + head) * head_dim;

            for (int64_t i = 0; i < half_dim; ++i) {
                const float inv_freq = rope_inv_freq(i, head_dim, rope_theta);
                const float angle = static_cast<float>(pos) * inv_freq;

                const float c = std::cos(angle);
                const float s = std::sin(angle);

                const float x1 = input[base + i];
                const float x2 = input[base + i + half_dim];

                output[base + i] = x1 * c - x2 * s;
                output[base + i + half_dim] = x2 * c + x1 * s;
            }
        }
    }
}

void rotary_apply_cpu(
    const Tensor& q,
    const Tensor& k,
    const Tensor& position_ids,
    Tensor& q_out,
    Tensor& k_out,
    float rope_theta
) {
    const int64_t num_tokens = q.shape()[0];
    const int64_t num_q_heads = q.shape()[1];
    const int64_t num_kv_heads = k.shape()[1];
    const int64_t head_dim = q.shape()[2];

    apply_rotary_one_cpu(
        q.ptr<float>(),
        position_ids.ptr<int32_t>(),
        q_out.ptr<float>(),
        num_tokens,
        num_q_heads,
        head_dim,
        rope_theta
    );

    apply_rotary_one_cpu(
        k.ptr<float>(),
        position_ids.ptr<int32_t>(),
        k_out.ptr<float>(),
        num_tokens,
        num_kv_heads,
        head_dim,
        rope_theta
    );
}

__global__ void apply_rotary_kernel(
    const float* input,
    const int32_t* position_ids,
    float* output,
    int64_t num_tokens,
    int64_t num_heads,
    int64_t head_dim,
    float rope_theta
) {
    const int64_t half_dim = head_dim / 2;
    const int64_t total = num_tokens * num_heads * half_dim;

    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (idx >= total) {
        return;
    }

    const int64_t i = idx % half_dim;
    const int64_t tmp = idx / half_dim;
    const int64_t head = tmp % num_heads;
    const int64_t token = tmp / num_heads;

    const int32_t pos = position_ids[token];

    const float inv_freq = powf(
        rope_theta,
        -static_cast<float>(2 * i) / static_cast<float>(head_dim)
    );

    const float angle = static_cast<float>(pos) * inv_freq;

    const float c = cosf(angle);
    const float s = sinf(angle);

    const int64_t base = (token * num_heads + head) * head_dim;

    const float x1 = input[base + i];
    const float x2 = input[base + i + half_dim];

    output[base + i] = x1 * c - x2 * s;
    output[base + i + half_dim] = x2 * c + x1 * s;
}

void apply_rotary_one_cuda(
    const Tensor& input,
    const Tensor& position_ids,
    Tensor& output,
    float rope_theta
) {
    const int64_t num_tokens = input.shape()[0];
    const int64_t num_heads = input.shape()[1];
    const int64_t head_dim = input.shape()[2];
    const int64_t half_dim = head_dim / 2;

    const int64_t total = num_tokens * num_heads * half_dim;

    if (total == 0) {
        return;
    }

    dim3 block = cuda_make_1d_block(256);
    dim3 grid = cuda_make_1d_grid(total, block.x);

    apply_rotary_kernel<<<grid, block>>>(
        input.ptr<float>(),
        position_ids.ptr<int32_t>(),
        output.ptr<float>(),
        num_tokens,
        num_heads,
        head_dim,
        rope_theta
    );

    CUDA_KERNEL_CHECK();
}

void rotary_apply_cuda(
    const Tensor& q,
    const Tensor& k,
    const Tensor& position_ids,
    Tensor& q_out,
    Tensor& k_out,
    float rope_theta
) {
    apply_rotary_one_cuda(q, position_ids, q_out, rope_theta);
    apply_rotary_one_cuda(k, position_ids, k_out, rope_theta);
}


}


RotaryEmbedding::RotaryEmbedding(int64_t head_dim, float rope_theta)
    : head_dim_(head_dim),
      rope_theta_(rope_theta) {
    if (head_dim_ <= 0 || head_dim_ % 2 != 0) {
        throw std::runtime_error("RotaryEmbedding head_dim must be positive");
    }

    if (rope_theta_ <= 0.0f) {
        throw std::runtime_error("RotaryEmbedding rope_theta must be positive");
    }
}

void RotaryEmbedding::apply(
    const Tensor& q,
    const Tensor& k,
    const Tensor& position_ids,
    Tensor& q_out,
    Tensor& k_out
) const {
    if (!initialized()) {
        throw std::runtime_error("RotaryEmbedding::apply called before initialized");
    }

    check_rotary_apply_args(
        q,
        k,
        position_ids,
        q_out,
        k_out,
        head_dim_
    );

    if (q.device() == Device::CPU) {
        rotary_apply_cpu(q, k, position_ids, q_out, k_out, rope_theta_);
        return;
    }

    if (q.device() == Device::CUDA) {
        rotary_apply_cuda(q, k, position_ids, q_out, k_out, rope_theta_);
        return;
    }


    throw std::runtime_error("RotaryEmbedding::apply not implemented yet");
}

} // namespace lite_llm
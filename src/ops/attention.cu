#include "ops/attention.hpp"

#include "core/cuda_utils.hpp"
#include "engine/block_table_manager.hpp"
#include "engine/paged_kv_cache.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace lite_llm {

namespace {

void check_flash_attention_args(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& output
) {
    if (q.dtype() != DType::FP32 ||
        k.dtype() != DType::FP32 ||
        v.dtype() != DType::FP32 ||
        output.dtype() != DType::FP32) {
        throw std::runtime_error("flash_attention tensors must be FP32");
    }

    if (q.shape().size() != 3) {
        throw std::runtime_error("flash_attention q must be 3D: [num_tokens, num_q_heads, head_dim]");
    }

    if (k.shape().size() != 3) {
        throw std::runtime_error("flash_attention k must be 3D: [num_tokens, num_kv_heads, head_dim]");
    }

    if (v.shape().size() != 3) {
        throw std::runtime_error("flash_attention v must be 3D: [num_tokens, num_kv_heads, head_dim]");
    }

    if (output.shape() != q.shape()) {
        throw std::runtime_error("flash_attention output shape mismatch");
    }

    const int64_t num_tokens = q.shape()[0];
    const int64_t num_q_heads = q.shape()[1];
    const int64_t head_dim = q.shape()[2];

    const int64_t k_tokens = k.shape()[0];
    const int64_t num_kv_heads = k.shape()[1];
    const int64_t k_head_dim = k.shape()[2];

    const int64_t v_tokens = v.shape()[0];
    const int64_t v_kv_heads = v.shape()[1];
    const int64_t v_head_dim = v.shape()[2];

    if (num_tokens < 0 || num_q_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0) {
        throw std::runtime_error("flash_attention invalid shape");
    }

    if (k_tokens != num_tokens || v_tokens != num_tokens) {
        throw std::runtime_error("flash_attention q/k/v num_tokens mismatch");
    }

    if (v_kv_heads != num_kv_heads) {
        throw std::runtime_error("flash_attention k/v num_kv_heads mismatch");
    }

    if (k_head_dim != head_dim || v_head_dim != head_dim) {
        throw std::runtime_error("flash_attention head_dim mismatch");
    }

    if (num_q_heads % num_kv_heads != 0) {
        throw std::runtime_error("flash_attention num_q_heads must be divisible by num_kv_heads");
    }

    if (q.device() != k.device() ||
        q.device() != v.device() ||
        q.device() != output.device()) {
        throw std::runtime_error("flash_attention tensors must be on same device");
    }
}


void check_flash_attention_kv_cache_args(
    const Tensor& q,
    const Tensor& key_cache,
    const Tensor& value_cache,
    int64_t kv_seq_len,
    const Tensor& output
) {
    if (q.dtype() != DType::FP32 ||
        key_cache.dtype() != DType::FP32 ||
        value_cache.dtype() != DType::FP32 ||
        output.dtype() != DType::FP32) {
        throw std::runtime_error("flash_attention_kv_cache tensors must be FP32");
    }

    if (q.shape().size() != 3) {
        throw std::runtime_error(
            "flash_attention_kv_cache q must be 3D: [1, num_q_heads, head_dim]"
        );
    }

    if (key_cache.shape().size() != 3) {
        throw std::runtime_error(
            "flash_attention_kv_cache key_cache must be 3D: [capacity, num_kv_heads, head_dim]"
        );
    }

    if (value_cache.shape().size() != 3) {
        throw std::runtime_error(
            "flash_attention_kv_cache value_cache must be 3D: [capacity, num_kv_heads, head_dim]"
        );
    }

    if (output.shape() != q.shape()) {
        throw std::runtime_error("flash_attention_kv_cache output shape mismatch");
    }

    const int64_t q_tokens = q.shape()[0];
    const int64_t num_q_heads = q.shape()[1];
    const int64_t head_dim = q.shape()[2];

    const int64_t capacity = key_cache.shape()[0];
    const int64_t num_kv_heads = key_cache.shape()[1];
    const int64_t k_head_dim = key_cache.shape()[2];

    const int64_t v_capacity = value_cache.shape()[0];
    const int64_t v_kv_heads = value_cache.shape()[1];
    const int64_t v_head_dim = value_cache.shape()[2];

    if (q_tokens != 1) {
        throw std::runtime_error(
            "flash_attention_kv_cache only supports q_tokens=1 for decode"
        );
    }

    if (num_q_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0) {
        throw std::runtime_error("flash_attention_kv_cache invalid shape");
    }

    if (capacity <= 0) {
        throw std::runtime_error("flash_attention_kv_cache capacity must be positive");
    }

    if (kv_seq_len <= 0 || kv_seq_len > capacity) {
        throw std::runtime_error("flash_attention_kv_cache kv_seq_len out of range");
    }

    if (v_capacity != capacity || v_kv_heads != num_kv_heads) {
        throw std::runtime_error("flash_attention_kv_cache key/value cache shape mismatch");
    }

    if (k_head_dim != head_dim || v_head_dim != head_dim) {
        throw std::runtime_error("flash_attention_kv_cache head_dim mismatch");
    }

    if (num_q_heads % num_kv_heads != 0) {
        throw std::runtime_error(
            "flash_attention_kv_cache num_q_heads must be divisible by num_kv_heads"
        );
    }

    if (q.device() != key_cache.device() ||
        q.device() != value_cache.device() ||
        q.device() != output.device()) {
        throw std::runtime_error(
            "flash_attention_kv_cache tensors must be on same device"
        );
    }
}

inline int64_t offset3(
    int64_t token,
    int64_t head,
    int64_t dim,
    int64_t num_heads,
    int64_t head_dim
) {
    return (token * num_heads + head) * head_dim + dim;
}

void flash_attention_cpu(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    Tensor& output
) {
    const float* q_ptr = q.ptr<float>();
    const float* k_ptr = k.ptr<float>();
    const float* v_ptr = v.ptr<float>();
    float* out_ptr = output.ptr<float>();

    const int64_t num_tokens = q.shape()[0];
    const int64_t num_q_heads = q.shape()[1];
    const int64_t head_dim = q.shape()[2];
    const int64_t num_kv_heads = k.shape()[1];

    const int64_t group_size = num_q_heads / num_kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    for (int64_t token = 0; token < num_tokens; ++token) {
        for (int64_t q_head = 0; q_head < num_q_heads; ++q_head) {
            const int64_t kv_head = q_head / group_size;

            float m = -FLT_MAX;
            float l = 0.0f;

            for (int64_t d = 0; d < head_dim; ++d) {
                out_ptr[offset3(token, q_head, d, num_q_heads, head_dim)] = 0.0f;
            }

            for (int64_t key_token = 0; key_token <= token; ++key_token) {
                float dot = 0.0f;

                for (int64_t d = 0; d < head_dim; ++d) {
                    dot += q_ptr[offset3(token, q_head, d, num_q_heads, head_dim)] *
                           k_ptr[offset3(key_token, kv_head, d, num_kv_heads, head_dim)];
                }

                const float score = dot * scale;

                const float new_m = std::max(m, score);
                const float alpha = (l == 0.0f) ? 0.0f : std::exp(m - new_m);
                const float beta = std::exp(score - new_m);
                const float new_l = l * alpha + beta;

                for (int64_t d = 0; d < head_dim; ++d) {
                    const int64_t out_idx = offset3(token, q_head, d, num_q_heads, head_dim);
                    const int64_t v_idx = offset3(key_token, kv_head, d, num_kv_heads, head_dim);

                    out_ptr[out_idx] = out_ptr[out_idx] * alpha + v_ptr[v_idx] * beta;
                }

                m = new_m;
                l = new_l;
            }

            for (int64_t d = 0; d < head_dim; ++d) {
                const int64_t out_idx = offset3(token, q_head, d, num_q_heads, head_dim);
                out_ptr[out_idx] /= l;
            }
        }
    }
}

void flash_attention_kv_cache_cpu(
    const Tensor& q,
    const Tensor& key_cache,
    const Tensor& value_cache,
    int64_t kv_seq_len,
    Tensor& output
) {
    const float* q_ptr = q.ptr<float>();
    const float* k_ptr = key_cache.ptr<float>();
    const float* v_ptr = value_cache.ptr<float>();
    float* out_ptr = output.ptr<float>();

    const int64_t num_q_heads = q.shape()[1];
    const int64_t head_dim = q.shape()[2];
    const int64_t num_kv_heads = key_cache.shape()[1];

    const int64_t group_size = num_q_heads / num_kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    for (int64_t q_head = 0; q_head < num_q_heads; ++q_head) {
        const int64_t kv_head = q_head / group_size;

        float m = -FLT_MAX;
        float l = 0.0f;

        for (int64_t d = 0; d < head_dim; ++d) {
            out_ptr[offset3(0, q_head, d, num_q_heads, head_dim)] = 0.0f;
        }

        for (int64_t key_token = 0; key_token < kv_seq_len; ++key_token) {
            float dot = 0.0f;

            for (int64_t d = 0; d < head_dim; ++d) {
                dot += q_ptr[offset3(0, q_head, d, num_q_heads, head_dim)] *
                       k_ptr[offset3(key_token, kv_head, d, num_kv_heads, head_dim)];
            }

            const float score = dot * scale;

            const float new_m = std::max(m, score);
            const float alpha = (l == 0.0f) ? 0.0f : std::exp(m - new_m);
            const float beta = std::exp(score - new_m);
            const float new_l = l * alpha + beta;

            for (int64_t d = 0; d < head_dim; ++d) {
                const int64_t out_idx =
                    offset3(0, q_head, d, num_q_heads, head_dim);

                const int64_t v_idx =
                    offset3(key_token, kv_head, d, num_kv_heads, head_dim);

                out_ptr[out_idx] =
                    out_ptr[out_idx] * alpha + v_ptr[v_idx] * beta;
            }

            m = new_m;
            l = new_l;
        }

        for (int64_t d = 0; d < head_dim; ++d) {
            const int64_t out_idx =
                offset3(0, q_head, d, num_q_heads, head_dim);

            out_ptr[out_idx] /= l;
        }
    }
}

__global__ void flash_attention_kernel(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    int64_t num_tokens,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim
) {
    extern __shared__ float shared[];

    float* reduce = shared;
    float* acc = shared + blockDim.x;

    __shared__ float m_shared;
    __shared__ float l_shared;
    __shared__ float alpha_shared;
    __shared__ float beta_shared;

    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const int64_t token = row / num_q_heads;
    const int64_t q_head = row % num_q_heads;

    const int tid = threadIdx.x;

    if (token >= num_tokens) {
        return;
    }

    const int64_t group_size = num_q_heads / num_kv_heads;
    const int64_t kv_head = q_head / group_size;

    const float scale = rsqrtf(static_cast<float>(head_dim));

    if (tid == 0) {
        m_shared = -FLT_MAX;
        l_shared = 0.0f;
    }

    for (int64_t d = tid; d < head_dim; d += blockDim.x) {
        acc[d] = 0.0f;
    }

    __syncthreads();

    for (int64_t key_token = 0; key_token <= token; ++key_token) {
        float local_dot = 0.0f;

        for (int64_t d = tid; d < head_dim; d += blockDim.x) {
            const int64_t q_idx =
                (token * num_q_heads + q_head) * head_dim + d;

            const int64_t k_idx =
                (key_token * num_kv_heads + kv_head) * head_dim + d;

            local_dot += q[q_idx] * k[k_idx];
        }

        reduce[tid] = local_dot;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            const float score = reduce[0] * scale;

            const float old_m = m_shared;
            const float old_l = l_shared;

            const float new_m = fmaxf(old_m, score);
            const float alpha = (old_l == 0.0f) ? 0.0f : expf(old_m - new_m);
            const float beta = expf(score - new_m);
            const float new_l = old_l * alpha + beta;

            m_shared = new_m;
            l_shared = new_l;
            alpha_shared = alpha;
            beta_shared = beta;
        }

        __syncthreads();

        for (int64_t d = tid; d < head_dim; d += blockDim.x) {
            const int64_t v_idx =
                (key_token * num_kv_heads + kv_head) * head_dim + d;

            acc[d] = acc[d] * alpha_shared + v[v_idx] * beta_shared;
        }

        __syncthreads();
    }

    const float inv_l = 1.0f / l_shared;

    for (int64_t d = tid; d < head_dim; d += blockDim.x) {
        const int64_t out_idx =
            (token * num_q_heads + q_head) * head_dim + d;

        output[out_idx] = acc[d] * inv_l;
    }
}


__global__ void flash_attention_kv_cache_kernel(
    const float* q,
    const float* key_cache,
    const float* value_cache,
    float* output,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t kv_seq_len
) {
    extern __shared__ float shared[];

    float* reduce = shared;
    float* acc = shared + blockDim.x;

    __shared__ float m_shared;
    __shared__ float l_shared;
    __shared__ float alpha_shared;
    __shared__ float beta_shared;

    const int64_t q_head = static_cast<int64_t>(blockIdx.x);
    const int tid = threadIdx.x;

    if (q_head >= num_q_heads) {
        return;
    }

    const int64_t group_size = num_q_heads / num_kv_heads;
    const int64_t kv_head = q_head / group_size;

    const float scale = rsqrtf(static_cast<float>(head_dim));

    if (tid == 0) {
        m_shared = -FLT_MAX;
        l_shared = 0.0f;
    }

    for (int64_t d = tid; d < head_dim; d += blockDim.x) {
        acc[d] = 0.0f;
    }

    __syncthreads();

    for (int64_t key_token = 0; key_token < kv_seq_len; ++key_token) {
        float local_dot = 0.0f;

        for (int64_t d = tid; d < head_dim; d += blockDim.x) {
            const int64_t q_idx =
                q_head * head_dim + d;

            const int64_t k_idx =
                (key_token * num_kv_heads + kv_head) * head_dim + d;

            local_dot += q[q_idx] * key_cache[k_idx];
        }

        reduce[tid] = local_dot;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            const float score = reduce[0] * scale;

            const float old_m = m_shared;
            const float old_l = l_shared;

            const float new_m = fmaxf(old_m, score);
            const float alpha =
                (old_l == 0.0f) ? 0.0f : expf(old_m - new_m);
            const float beta = expf(score - new_m);
            const float new_l = old_l * alpha + beta;

            m_shared = new_m;
            l_shared = new_l;
            alpha_shared = alpha;
            beta_shared = beta;
        }

        __syncthreads();

        for (int64_t d = tid; d < head_dim; d += blockDim.x) {
            const int64_t v_idx =
                (key_token * num_kv_heads + kv_head) * head_dim + d;

            acc[d] =
                acc[d] * alpha_shared +
                value_cache[v_idx] * beta_shared;
        }

        __syncthreads();
    }

    const float inv_l = 1.0f / l_shared;

    for (int64_t d = tid; d < head_dim; d += blockDim.x) {
        const int64_t out_idx =
            q_head * head_dim + d;

        output[out_idx] = acc[d] * inv_l;
    }
}

__global__ void flash_attention_paged_kv_cache_batch_kernel(
    const float* q,
    const float* key_pool,
    const float* value_pool,
    const int32_t* block_tables,
    const int32_t* kv_seq_lens,
    float* output,
    int64_t batch_size,
    int64_t max_blocks_per_request,
    int64_t page_size,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim
) {
    extern __shared__ float shared[];

    float* reduce = shared;
    float* acc = shared + blockDim.x;

    __shared__ float m_shared;
    __shared__ float l_shared;
    __shared__ float alpha_shared;
    __shared__ float beta_shared;

    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const int64_t batch_idx = row / num_q_heads;
    const int64_t q_head = row % num_q_heads;
    const int tid = threadIdx.x;

    if (batch_idx >= batch_size) {
        return;
    }

    const int64_t kv_seq_len =
        static_cast<int64_t>(kv_seq_lens[batch_idx]);

    if (kv_seq_len <= 0) {
        return;
    }

    const int64_t group_size = num_q_heads / num_kv_heads;
    const int64_t kv_head = q_head / group_size;

    const float scale = rsqrtf(static_cast<float>(head_dim));

    if (tid == 0) {
        m_shared = -FLT_MAX;
        l_shared = 0.0f;
    }

    for (int64_t d = tid; d < head_dim; d += blockDim.x) {
        acc[d] = 0.0f;
    }

    __syncthreads();

    for (int64_t key_token = 0; key_token < kv_seq_len; ++key_token) {
        const int64_t logical_block = key_token / page_size;
        const int64_t page_offset = key_token % page_size;

        const int32_t physical_block =
            block_tables[
                batch_idx * max_blocks_per_request +
                logical_block
            ];

        const int64_t physical_token =
            static_cast<int64_t>(physical_block) * page_size +
            page_offset;

        float local_dot = 0.0f;

        for (int64_t d = tid; d < head_dim; d += blockDim.x) {
            const int64_t q_idx =
                (batch_idx * num_q_heads + q_head) * head_dim + d;

            const int64_t k_idx =
                (physical_token * num_kv_heads + kv_head) * head_dim + d;

            local_dot += q[q_idx] * key_pool[k_idx];
        }

        reduce[tid] = local_dot;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduce[tid] += reduce[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            const float score = reduce[0] * scale;

            const float old_m = m_shared;
            const float old_l = l_shared;

            const float new_m = fmaxf(old_m, score);
            const float alpha =
                (old_l == 0.0f) ? 0.0f : expf(old_m - new_m);
            const float beta = expf(score - new_m);
            const float new_l = old_l * alpha + beta;

            m_shared = new_m;
            l_shared = new_l;
            alpha_shared = alpha;
            beta_shared = beta;
        }

        __syncthreads();

        for (int64_t d = tid; d < head_dim; d += blockDim.x) {
            const int64_t v_idx =
                (physical_token * num_kv_heads + kv_head) * head_dim + d;

            acc[d] =
                acc[d] * alpha_shared +
                value_pool[v_idx] * beta_shared;
        }

        __syncthreads();
    }

    const float inv_l = 1.0f / l_shared;

    for (int64_t d = tid; d < head_dim; d += blockDim.x) {
        const int64_t out_idx =
            (batch_idx * num_q_heads + q_head) * head_dim + d;

        output[out_idx] = acc[d] * inv_l;
    }
}

void flash_attention_cuda(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    Tensor& output
) {
    const int64_t num_tokens = q.shape()[0];
    const int64_t num_q_heads = q.shape()[1];
    const int64_t head_dim = q.shape()[2];
    const int64_t num_kv_heads = k.shape()[1];

    if (num_tokens == 0) {
        return;
    }

    constexpr int block_size = 256;

    const int64_t num_rows = num_tokens * num_q_heads;

    dim3 block(block_size);
    dim3 grid(static_cast<unsigned int>(num_rows));

    const size_t shared_bytes =
        static_cast<size_t>(block_size + head_dim) * sizeof(float);

    flash_attention_kernel<<<grid, block, shared_bytes>>>(
        q.ptr<float>(),
        k.ptr<float>(),
        v.ptr<float>(),
        output.ptr<float>(),
        num_tokens,
        num_q_heads,
        num_kv_heads,
        head_dim
    );

    CUDA_KERNEL_CHECK();
}


void flash_attention_kv_cache_cuda(
    const Tensor& q,
    const Tensor& key_cache,
    const Tensor& value_cache,
    int64_t kv_seq_len,
    Tensor& output
) {
    const int64_t num_q_heads = q.shape()[1];
    const int64_t head_dim = q.shape()[2];
    const int64_t num_kv_heads = key_cache.shape()[1];

    constexpr int block_size = 256;

    dim3 block(block_size);
    dim3 grid(static_cast<unsigned int>(num_q_heads));

    const size_t shared_bytes =
        static_cast<size_t>(block_size + head_dim) * sizeof(float);

    flash_attention_kv_cache_kernel<<<grid, block, shared_bytes>>>(
        q.ptr<float>(),
        key_cache.ptr<float>(),
        value_cache.ptr<float>(),
        output.ptr<float>(),
        num_q_heads,
        num_kv_heads,
        head_dim,
        kv_seq_len
    );

    CUDA_KERNEL_CHECK();
}

}  // namespace

void flash_attention(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    Tensor& output
) {
    check_flash_attention_args(q, k, v, output);

    if (q.device() == Device::CPU) {
        flash_attention_cpu(q, k, v, output);
        return;
    }

    if (q.device() == Device::CUDA) {
        flash_attention_cuda(q, k, v, output);
        return;
    }

    throw std::runtime_error("Unsupported device for flash_attention");
}

/**
 * @brief page版的attention 其实就是把page中的kv cache拿出来，然后给到原来那个函数进行计算
 * 
 * @param q 
 * @param paged_kv_cache 
 * @param table_manager 
 * @param table_idx 
 * @param layer_idx 
 * @param kv_seq_len 
 * @param output 
 */
void flash_attention_paged_kv_cache_cuda(
    const Tensor& q,
    const ModelPagedKVCache& paged_kv_cache,
    const BlockTableManager& table_manager,
    int64_t table_idx,
    int64_t layer_idx,
    int64_t kv_seq_len,
    Tensor& output
) {
    if (q.device() != Device::CUDA) {
        throw std::runtime_error(
            "flash_attention_paged_kv_cache_cuda q must be CUDA tensor"
        );
    }

    if (output.device() != Device::CUDA) {
        throw std::runtime_error(
            "flash_attention_paged_kv_cache_cuda output must be CUDA tensor"
        );
    }

    if (kv_seq_len <= 0) {
        throw std::runtime_error(
            "flash_attention_paged_kv_cache_cuda kv_seq_len must be positive"
        );
    }

    Tensor key_contig(
        std::vector<int64_t>{
            kv_seq_len,
            paged_kv_cache.num_kv_heads(),
            paged_kv_cache.head_dim()
        },
        q.dtype(),
        q.device()
    );

    Tensor value_contig(
        std::vector<int64_t>{
            kv_seq_len,
            paged_kv_cache.num_kv_heads(),
            paged_kv_cache.head_dim()
        },
        q.dtype(),
        q.device()
    );

    paged_kv_cache.gather_layer_key(
        layer_idx,
        table_manager,
        table_idx,
        0,
        kv_seq_len,
        key_contig
    );

    paged_kv_cache.gather_layer_value(
        layer_idx,
        table_manager,
        table_idx,
        0,
        kv_seq_len,
        value_contig
    );

    check_flash_attention_kv_cache_args(
        q,
        key_contig,
        value_contig,
        kv_seq_len,
        output
    );

    flash_attention_kv_cache_cuda(
        q,
        key_contig,
        value_contig,
        kv_seq_len,
        output
    );
}

void flash_attention_paged_kv_cache_batch_cuda(
    const Tensor& q,
    const ModelPagedKVCache& paged_kv_cache,
    const BlockTableManager& table_manager,
    const std::vector<int64_t>& table_indices,
    int64_t layer_idx,
    const std::vector<int64_t>& kv_seq_lens,
    Tensor& output
) {
    if (q.device() != Device::CUDA) {
        throw std::runtime_error(
            "flash_attention_paged_kv_cache_batch_cuda q must be CUDA tensor"
        );
    }

    if (output.device() != Device::CUDA) {
        throw std::runtime_error(
            "flash_attention_paged_kv_cache_batch_cuda output must be CUDA tensor"
        );
    }

    if (q.dtype() != DType::FP32 || output.dtype() != DType::FP32) {
        throw std::runtime_error(
            "flash_attention_paged_kv_cache_batch_cuda tensors must be FP32"
        );
    }

    if (q.shape().size() != 3) {
        throw std::runtime_error(
            "flash_attention_paged_kv_cache_batch_cuda q must be 3D"
        );
    }

    if (output.shape() != q.shape()) {
        throw std::runtime_error(
            "flash_attention_paged_kv_cache_batch_cuda output shape mismatch"
        );
    }

    const int64_t batch_size = q.shape()[0];
    const int64_t num_q_heads = q.shape()[1];
    const int64_t head_dim = q.shape()[2];

    const int64_t num_kv_heads = paged_kv_cache.num_kv_heads();
    const int64_t page_size = paged_kv_cache.page_size();

    if (batch_size <= 0 || num_q_heads <= 0 ||
        num_kv_heads <= 0 || head_dim <= 0) {
        throw std::runtime_error(
            "flash_attention_paged_kv_cache_batch_cuda invalid shape"
        );
    }

    if (head_dim != paged_kv_cache.head_dim()) {
        throw std::runtime_error(
            "flash_attention_paged_kv_cache_batch_cuda head_dim mismatch"
        );
    }

    if (num_q_heads % num_kv_heads != 0) {
        throw std::runtime_error(
            "flash_attention_paged_kv_cache_batch_cuda invalid head grouping"
        );
    }

    if (table_indices.size() != static_cast<size_t>(batch_size) ||
        kv_seq_lens.size() != static_cast<size_t>(batch_size)) {
        throw std::runtime_error(
            "flash_attention_paged_kv_cache_batch_cuda metadata size mismatch"
        );
    }

    int64_t max_blocks_per_request = 0;

    for (int64_t row = 0; row < batch_size; ++row) {
        const int64_t kv_seq_len =
            kv_seq_lens[static_cast<size_t>(row)];

        if (kv_seq_len <= 0 ||
            kv_seq_len > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
            throw std::runtime_error(
                "flash_attention_paged_kv_cache_batch_cuda kv_seq_len out of range"
            );
        }

        const int64_t required_blocks =
            (kv_seq_len + page_size - 1) / page_size;

        max_blocks_per_request =
            std::max(max_blocks_per_request, required_blocks);
    }

    if (max_blocks_per_request <= 0 ||
        max_blocks_per_request > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error(
            "flash_attention_paged_kv_cache_batch_cuda invalid block table width"
        );
    }

    std::vector<int32_t> dense_block_tables(
        static_cast<size_t>(batch_size * max_blocks_per_request),
        -1
    );

    std::vector<int32_t> kv_seq_lens_i32(
        static_cast<size_t>(batch_size),
        0
    );

    for (int64_t row = 0; row < batch_size; ++row) {
        const int64_t table_idx =
            table_indices[static_cast<size_t>(row)];

        const int64_t kv_seq_len =
            kv_seq_lens[static_cast<size_t>(row)];

        const int64_t required_blocks =
            (kv_seq_len + page_size - 1) / page_size;

        const std::vector<int32_t>& block_table =
            table_manager.table(table_idx);

        if (required_blocks > static_cast<int64_t>(block_table.size())) {
            throw std::runtime_error(
                "flash_attention_paged_kv_cache_batch_cuda block table too short"
            );
        }

        kv_seq_lens_i32[static_cast<size_t>(row)] =
            static_cast<int32_t>(kv_seq_len);

        for (int64_t block = 0; block < required_blocks; ++block) {
            dense_block_tables[
                static_cast<size_t>(
                    row * max_blocks_per_request + block
                )
            ] = block_table[static_cast<size_t>(block)];
        }
    }

    Tensor block_tables_tensor(
        {batch_size, max_blocks_per_request},
        DType::INT32,
        Device::CUDA
    );

    Tensor kv_seq_lens_tensor(
        {batch_size},
        DType::INT32,
        Device::CUDA
    );

    block_tables_tensor.copy_from_cpu(
        dense_block_tables.data(),
        dense_block_tables.size() * sizeof(int32_t)
    );

    kv_seq_lens_tensor.copy_from_cpu(
        kv_seq_lens_i32.data(),
        kv_seq_lens_i32.size() * sizeof(int32_t)
    );

    const LayerPagedKVCache& layer_cache =
        paged_kv_cache.layer(layer_idx);

    constexpr int block_size = 256;

    dim3 block(block_size);
    dim3 grid(static_cast<unsigned int>(batch_size * num_q_heads));

    const size_t shared_bytes =
        static_cast<size_t>(block_size + head_dim) * sizeof(float);

    flash_attention_paged_kv_cache_batch_kernel<<<
        grid,
        block,
        shared_bytes
    >>>(
        q.ptr<float>(),
        layer_cache.key_pool.ptr<float>(),
        layer_cache.value_pool.ptr<float>(),
        block_tables_tensor.ptr<int32_t>(),
        kv_seq_lens_tensor.ptr<int32_t>(),
        output.ptr<float>(),
        batch_size,
        max_blocks_per_request,
        page_size,
        num_q_heads,
        num_kv_heads,
        head_dim
    );

    CUDA_KERNEL_CHECK();
}


void flash_attention_kv_cache(
    const Tensor& q,
    const Tensor& key_cache,
    const Tensor& value_cache,
    int64_t kv_seq_len,
    Tensor& output
) {
    check_flash_attention_kv_cache_args(
        q,
        key_cache,
        value_cache,
        kv_seq_len,
        output
    );

    if (q.device() == Device::CPU) {
        flash_attention_kv_cache_cpu(
            q,
            key_cache,
            value_cache,
            kv_seq_len,
            output
        );
        return;
    }

    if (q.device() == Device::CUDA) {
        flash_attention_kv_cache_cuda(
            q,
            key_cache,
            value_cache,
            kv_seq_len,
            output
        );
        return;
    }

    throw std::runtime_error("Unsupported device for flash_attention_kv_cache");
}

}  // namespace lite_llm

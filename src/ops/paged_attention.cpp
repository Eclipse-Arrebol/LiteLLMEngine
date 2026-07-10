// src/ops/paged_attention.cpp

#include "ops/paged_attention.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace lite_llm {

namespace {

/**
 * @brief 将三维坐标转换成一维坐标 [token_idx][head_idx][dim_idx]
 * 
 * @param token_idx 
 * @param head_idx 
 * @param dim_idx 
 * @param num_heads 
 * @param head_dim 
 * @return int64_t 
 */
int64_t flat_index(
    int64_t token_idx,
    int64_t head_idx,
    int64_t dim_idx,
    int64_t num_heads,
    int64_t head_dim
) {
    return (token_idx * num_heads + head_idx) * head_dim + dim_idx;
}

/**
 * @brief 检查是否是cpu，是否是fp32
 * 
 * @param t 
 * @param name 
 */
void check_tensor_cpu_fp32(const Tensor& t, const char* name) {
    if (t.device() != Device::CPU) {
        throw std::runtime_error(std::string(name) + " must be CPU tensor");
    }

    if (t.dtype() != DType::FP32) {
        throw std::runtime_error(std::string(name) + " must be FP32 tensor");
    }
}

}  // namespace


/**
 * @brief cpu版本的attention 通过paged_kv_cache拿到kv cache，后面的就和之前的算法类似了
 * 
 * @param query 
 * @param paged_kv_cache 
 * @param table_manager 
 * @param table_idx 
 * @param layer_idx 
 * @param kv_seq_len 
 * @param output 
 */
void paged_attention_decode_cpu(
    const Tensor& query,
    const ModelPagedKVCache& paged_kv_cache,
    const BlockTableManager& table_manager,
    int64_t table_idx,
    int64_t layer_idx,
    int64_t kv_seq_len,
    Tensor& output
) {
    check_tensor_cpu_fp32(query, "query");
    check_tensor_cpu_fp32(output, "output");

    if (query.shape().size() != 3) {
        throw std::runtime_error("query must be 3D [q_seq_len, num_q_heads, head_dim]");
    }

    if (output.shape().size() != 3) {
        throw std::runtime_error("output must be 3D [q_seq_len, num_q_heads, head_dim]");
    }

    if (kv_seq_len <= 0) {
        throw std::runtime_error("kv_seq_len must be positive");
    }

    const int64_t q_seq_len = query.shape()[0];
    const int64_t num_q_heads = query.shape()[1];
    const int64_t head_dim = query.shape()[2];

    const int64_t num_kv_heads = paged_kv_cache.num_kv_heads();

    if (q_seq_len != 1) {
        throw std::runtime_error("paged_attention_decode_cpu currently supports q_seq_len == 1 only");
    }

    if (head_dim != paged_kv_cache.head_dim()) {
        throw std::runtime_error("query head_dim mismatch");
    }

    if (num_q_heads <= 0 || num_kv_heads <= 0) {
        throw std::runtime_error("invalid number of heads");
    }

    if (num_q_heads % num_kv_heads != 0) {
        throw std::runtime_error("num_q_heads must be divisible by num_kv_heads");
    }

    if (output.shape()[0] != q_seq_len ||
        output.shape()[1] != num_q_heads ||
        output.shape()[2] != head_dim) {
        throw std::runtime_error("output shape mismatch");
    }

    Tensor key_contig(
        std::vector<int64_t>{kv_seq_len, num_kv_heads, head_dim},
        DType::FP32,
        Device::CPU
    );

    Tensor value_contig(
        std::vector<int64_t>{kv_seq_len, num_kv_heads, head_dim},
        DType::FP32,
        Device::CPU
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

    const float* q_ptr = query.ptr<float>();
    const float* k_ptr = key_contig.ptr<float>();
    const float* v_ptr = value_contig.ptr<float>();
    float* out_ptr = output.ptr<float>();

    const int64_t group_size = num_q_heads / num_kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    std::vector<float> scores(static_cast<size_t>(kv_seq_len), 0.0f);

    for (int64_t qh = 0; qh < num_q_heads; ++qh) {
        const int64_t kvh = qh / group_size;

        float max_score = -std::numeric_limits<float>::infinity();

        for (int64_t kt = 0; kt < kv_seq_len; ++kt) {
            float dot = 0.0f;

            for (int64_t d = 0; d < head_dim; ++d) {
                const int64_t q_idx =
                    flat_index(0, qh, d, num_q_heads, head_dim);

                const int64_t k_idx =
                    flat_index(kt, kvh, d, num_kv_heads, head_dim);

                dot += q_ptr[q_idx] * k_ptr[k_idx];
            }

            const float score = dot * scale;
            scores[static_cast<size_t>(kt)] = score;

            if (score > max_score) {
                max_score = score;
            }
        }

        float exp_sum = 0.0f;

        for (int64_t kt = 0; kt < kv_seq_len; ++kt) {
            const float e =
                std::exp(scores[static_cast<size_t>(kt)] - max_score);

            scores[static_cast<size_t>(kt)] = e;
            exp_sum += e;
        }

        for (int64_t d = 0; d < head_dim; ++d) {
            float acc = 0.0f;

            for (int64_t kt = 0; kt < kv_seq_len; ++kt) {
                const float prob = scores[static_cast<size_t>(kt)] / exp_sum;

                const int64_t v_idx =
                    flat_index(kt, kvh, d, num_kv_heads, head_dim);

                acc += prob * v_ptr[v_idx];
            }

            const int64_t out_idx =
                flat_index(0, qh, d, num_q_heads, head_dim);

            out_ptr[out_idx] = acc;
        }
    }
}

}  // namespace lite_llm
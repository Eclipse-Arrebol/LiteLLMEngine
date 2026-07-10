// tests/test_paged_attention.cpp

#include "core/tensor.hpp"
#include "engine/block_table_manager.hpp"
#include "engine/paged_kv_cache.hpp"
#include "ops/paged_attention.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace lite_llm;

template <typename Fn>
static bool thrown_runtime_error(Fn fn) {
    try {
        fn();
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

static int64_t flat_index(
    int64_t token_idx,
    int64_t head_idx,
    int64_t dim_idx,
    int64_t num_heads,
    int64_t head_dim
) {
    return (token_idx * num_heads + head_idx) * head_dim + dim_idx;
}

static void fill_tensor_seq(Tensor& tensor, float base) {
    float* p = tensor.ptr<float>();

    for (size_t i = 0; i < tensor.numel(); ++i) {
        p[i] = base + static_cast<float>(i) * 0.01f;
    }
}

static bool almost_equal(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) <= eps;
}

static void reference_decode_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    Tensor& output
) {
    const int64_t q_seq_len = query.shape()[0];
    const int64_t num_q_heads = query.shape()[1];
    const int64_t head_dim = query.shape()[2];

    const int64_t kv_seq_len = key.shape()[0];
    const int64_t num_kv_heads = key.shape()[1];

    assert(q_seq_len == 1);
    assert(value.shape()[0] == kv_seq_len);
    assert(value.shape()[1] == num_kv_heads);
    assert(value.shape()[2] == head_dim);
    assert(num_q_heads % num_kv_heads == 0);

    const int64_t group_size = num_q_heads / num_kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    const float* q_ptr = query.ptr<float>();
    const float* k_ptr = key.ptr<float>();
    const float* v_ptr = value.ptr<float>();
    float* out_ptr = output.ptr<float>();

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

static void expect_tensor_close(
    const Tensor& a,
    const Tensor& b,
    float eps = 1e-4f
) {
    assert(a.numel() == b.numel());

    const float* ap = a.ptr<float>();
    const float* bp = b.ptr<float>();

    for (size_t i = 0; i < a.numel(); ++i) {
        assert(almost_equal(ap[i], bp[i], eps));
    }
}

static void test_paged_attention_matches_reference() {
    constexpr int64_t num_layers = 1;
    constexpr int64_t capacity = 6;
    constexpr int64_t page_size = 2;
    constexpr int64_t num_kv_heads = 2;
    constexpr int64_t num_q_heads = 4;
    constexpr int64_t head_dim = 3;
    constexpr int64_t kv_seq_len = 5;

    ModelPagedKVCache cache(
        num_layers,
        capacity,
        page_size,
        num_kv_heads,
        head_dim,
        DType::FP32,
        Device::CPU
    );

    BlockTableManager table_manager(cache.num_blocks());

    const int64_t table_idx = table_manager.allocate_table();

    table_manager.ensure_blocks(table_idx, kv_seq_len, page_size);

    Tensor key(
        std::vector<int64_t>{kv_seq_len, num_kv_heads, head_dim},
        DType::FP32,
        Device::CPU
    );

    Tensor value(
        std::vector<int64_t>{kv_seq_len, num_kv_heads, head_dim},
        DType::FP32,
        Device::CPU
    );

    Tensor query(
        std::vector<int64_t>{1, num_q_heads, head_dim},
        DType::FP32,
        Device::CPU
    );

    Tensor paged_output(
        std::vector<int64_t>{1, num_q_heads, head_dim},
        DType::FP32,
        Device::CPU
    );

    Tensor ref_output(
        std::vector<int64_t>{1, num_q_heads, head_dim},
        DType::FP32,
        Device::CPU
    );

    fill_tensor_seq(key, 0.1f);
    fill_tensor_seq(value, 1.0f);
    fill_tensor_seq(query, 0.5f);

    cache.update_layer(
        0,
        table_manager,
        table_idx,
        0,
        key,
        value
    );

    paged_attention_decode_cpu(
        query,
        cache,
        table_manager,
        table_idx,
        0,
        kv_seq_len,
        paged_output
    );

    reference_decode_attention(
        query,
        key,
        value,
        ref_output
    );

    expect_tensor_close(paged_output, ref_output);
}

static void test_paged_attention_cross_page_decode() {
    constexpr int64_t num_layers = 1;
    constexpr int64_t capacity = 8;
    constexpr int64_t page_size = 2;
    constexpr int64_t num_kv_heads = 1;
    constexpr int64_t num_q_heads = 2;
    constexpr int64_t head_dim = 4;
    constexpr int64_t kv_seq_len = 7;

    ModelPagedKVCache cache(
        num_layers,
        capacity,
        page_size,
        num_kv_heads,
        head_dim,
        DType::FP32,
        Device::CPU
    );

    BlockTableManager table_manager(cache.num_blocks());

    const int64_t table_idx = table_manager.allocate_table();

    table_manager.ensure_blocks(table_idx, kv_seq_len, page_size);

    Tensor key(
        std::vector<int64_t>{kv_seq_len, num_kv_heads, head_dim},
        DType::FP32,
        Device::CPU
    );

    Tensor value(
        std::vector<int64_t>{kv_seq_len, num_kv_heads, head_dim},
        DType::FP32,
        Device::CPU
    );

    Tensor query(
        std::vector<int64_t>{1, num_q_heads, head_dim},
        DType::FP32,
        Device::CPU
    );

    Tensor paged_output(
        std::vector<int64_t>{1, num_q_heads, head_dim},
        DType::FP32,
        Device::CPU
    );

    Tensor ref_output(
        std::vector<int64_t>{1, num_q_heads, head_dim},
        DType::FP32,
        Device::CPU
    );

    fill_tensor_seq(key, 0.2f);
    fill_tensor_seq(value, 2.0f);
    fill_tensor_seq(query, 0.3f);

    cache.update_layer(
        0,
        table_manager,
        table_idx,
        0,
        key,
        value
    );

    paged_attention_decode_cpu(
        query,
        cache,
        table_manager,
        table_idx,
        0,
        kv_seq_len,
        paged_output
    );

    reference_decode_attention(
        query,
        key,
        value,
        ref_output
    );

    expect_tensor_close(paged_output, ref_output);
}

static void test_invalid_q_seq_len_should_throw() {
    constexpr int64_t num_layers = 1;
    constexpr int64_t capacity = 4;
    constexpr int64_t page_size = 2;
    constexpr int64_t num_kv_heads = 1;
    constexpr int64_t num_q_heads = 2;
    constexpr int64_t head_dim = 2;
    constexpr int64_t kv_seq_len = 2;

    ModelPagedKVCache cache(
        num_layers,
        capacity,
        page_size,
        num_kv_heads,
        head_dim,
        DType::FP32,
        Device::CPU
    );

    BlockTableManager table_manager(cache.num_blocks());

    const int64_t table_idx = table_manager.allocate_table();
    table_manager.ensure_blocks(table_idx, kv_seq_len, page_size);

    Tensor key(
        std::vector<int64_t>{kv_seq_len, num_kv_heads, head_dim},
        DType::FP32,
        Device::CPU
    );

    Tensor value(
        std::vector<int64_t>{kv_seq_len, num_kv_heads, head_dim},
        DType::FP32,
        Device::CPU
    );

    Tensor query(
        std::vector<int64_t>{2, num_q_heads, head_dim},
        DType::FP32,
        Device::CPU
    );

    Tensor output(
        std::vector<int64_t>{2, num_q_heads, head_dim},
        DType::FP32,
        Device::CPU
    );

    fill_tensor_seq(key, 0.1f);
    fill_tensor_seq(value, 1.0f);
    fill_tensor_seq(query, 0.5f);

    cache.update_layer(
        0,
        table_manager,
        table_idx,
        0,
        key,
        value
    );

    const bool thrown = thrown_runtime_error([&]() {
        paged_attention_decode_cpu(
            query,
            cache,
            table_manager,
            table_idx,
            0,
            kv_seq_len,
            output
        );
    });

    assert(thrown);
}

int main() {
    test_paged_attention_matches_reference();
    test_paged_attention_cross_page_decode();
    test_invalid_q_seq_len_should_throw();

    std::cout << "test_paged_attention passed" << std::endl;
    return 0;
}
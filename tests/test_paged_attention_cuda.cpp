// tests/test_paged_attention_cuda.cpp

#include "core/tensor.hpp"
#include "engine/block_table_manager.hpp"
#include "engine/paged_kv_cache.hpp"
#include "ops/attention.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
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

static void fill_cpu_vector(
    std::vector<float>& data,
    float base,
    float step = 0.01f
) {
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = base + static_cast<float>(i) * step;
    }
}

static void copy_vector_to_tensor(
    const std::vector<float>& src,
    Tensor& dst
) {
    assert(src.size() == dst.numel());
    dst.copy_from_cpu(src.data(), src.size() * sizeof(float));
}

static std::vector<float> copy_tensor_to_vector(const Tensor& src) {
    std::vector<float> out(src.numel());
    src.copy_to_cpu(out.data(), out.size() * sizeof(float));
    return out;
}

static bool almost_equal(float a, float b, float eps = 1e-3f) {
    return std::fabs(a - b) <= eps;
}

static void expect_vector_close(
    const std::vector<float>& a,
    const std::vector<float>& b,
    float eps = 1e-3f
) {
    assert(a.size() == b.size());

    for (size_t i = 0; i < a.size(); ++i) {
        if (!almost_equal(a[i], b[i], eps)) {
            std::cerr << "Mismatch at index " << i
                      << ": a=" << a[i]
                      << ", b=" << b[i]
                      << ", diff=" << std::fabs(a[i] - b[i])
                      << std::endl;
            assert(false);
        }
    }
}

static void test_paged_cuda_matches_contiguous_cuda() {
    constexpr int64_t num_layers = 1;
    constexpr int64_t capacity = 8;
    constexpr int64_t page_size = 2;
    constexpr int64_t num_q_heads = 2;
    constexpr int64_t num_kv_heads = 2;
    constexpr int64_t head_dim = 4;
    constexpr int64_t kv_seq_len = 5;

    ModelPagedKVCache paged_cache(
        num_layers,
        capacity,
        page_size,
        num_kv_heads,
        head_dim,
        DType::FP32,
        Device::CUDA
    );

    BlockTableManager table_manager(paged_cache.num_blocks());

    const int64_t table_idx = table_manager.allocate_table();

    table_manager.ensure_blocks(table_idx, kv_seq_len, page_size);

    Tensor query(
        std::vector<int64_t>{1, num_q_heads, head_dim},
        DType::FP32,
        Device::CUDA
    );

    Tensor key_contig(
        std::vector<int64_t>{kv_seq_len, num_kv_heads, head_dim},
        DType::FP32,
        Device::CUDA
    );

    Tensor value_contig(
        std::vector<int64_t>{kv_seq_len, num_kv_heads, head_dim},
        DType::FP32,
        Device::CUDA
    );

    Tensor out_contig(
        std::vector<int64_t>{1, num_q_heads, head_dim},
        DType::FP32,
        Device::CUDA
    );

    Tensor out_paged(
        std::vector<int64_t>{1, num_q_heads, head_dim},
        DType::FP32,
        Device::CUDA
    );

    std::vector<float> query_cpu(query.numel());
    std::vector<float> key_cpu(key_contig.numel());
    std::vector<float> value_cpu(value_contig.numel());

    fill_cpu_vector(query_cpu, 0.5f);
    fill_cpu_vector(key_cpu, 0.1f);
    fill_cpu_vector(value_cpu, 1.0f);

    copy_vector_to_tensor(query_cpu, query);
    copy_vector_to_tensor(key_cpu, key_contig);
    copy_vector_to_tensor(value_cpu, value_contig);

    paged_cache.update_layer(
        0,
        table_manager,
        table_idx,
        0,
        key_contig,
        value_contig
    );

    flash_attention_kv_cache(
        query,
        key_contig,
        value_contig,
        kv_seq_len,
        out_contig
    );

    flash_attention_paged_kv_cache_cuda(
        query,
        paged_cache,
        table_manager,
        table_idx,
        0,
        kv_seq_len,
        out_paged
    );

    const std::vector<float> out_contig_cpu =
        copy_tensor_to_vector(out_contig);

    const std::vector<float> out_paged_cpu =
        copy_tensor_to_vector(out_paged);

    expect_vector_close(out_contig_cpu, out_paged_cpu);
}

static void test_paged_cuda_gqa_matches_contiguous_cuda() {
    constexpr int64_t num_layers = 1;
    constexpr int64_t capacity = 8;
    constexpr int64_t page_size = 2;
    constexpr int64_t num_q_heads = 4;
    constexpr int64_t num_kv_heads = 2;
    constexpr int64_t head_dim = 4;
    constexpr int64_t kv_seq_len = 6;

    ModelPagedKVCache paged_cache(
        num_layers,
        capacity,
        page_size,
        num_kv_heads,
        head_dim,
        DType::FP32,
        Device::CUDA
    );

    BlockTableManager table_manager(paged_cache.num_blocks());

    const int64_t table_idx = table_manager.allocate_table();

    table_manager.ensure_blocks(table_idx, kv_seq_len, page_size);

    Tensor query(
        std::vector<int64_t>{1, num_q_heads, head_dim},
        DType::FP32,
        Device::CUDA
    );

    Tensor key_contig(
        std::vector<int64_t>{kv_seq_len, num_kv_heads, head_dim},
        DType::FP32,
        Device::CUDA
    );

    Tensor value_contig(
        std::vector<int64_t>{kv_seq_len, num_kv_heads, head_dim},
        DType::FP32,
        Device::CUDA
    );

    Tensor out_contig(
        std::vector<int64_t>{1, num_q_heads, head_dim},
        DType::FP32,
        Device::CUDA
    );

    Tensor out_paged(
        std::vector<int64_t>{1, num_q_heads, head_dim},
        DType::FP32,
        Device::CUDA
    );

    std::vector<float> query_cpu(query.numel());
    std::vector<float> key_cpu(key_contig.numel());
    std::vector<float> value_cpu(value_contig.numel());

    fill_cpu_vector(query_cpu, 0.2f);
    fill_cpu_vector(key_cpu, 0.7f);
    fill_cpu_vector(value_cpu, 1.5f);

    copy_vector_to_tensor(query_cpu, query);
    copy_vector_to_tensor(key_cpu, key_contig);
    copy_vector_to_tensor(value_cpu, value_contig);

    paged_cache.update_layer(
        0,
        table_manager,
        table_idx,
        0,
        key_contig,
        value_contig
    );

    flash_attention_kv_cache(
        query,
        key_contig,
        value_contig,
        kv_seq_len,
        out_contig
    );

    flash_attention_paged_kv_cache_cuda(
        query,
        paged_cache,
        table_manager,
        table_idx,
        0,
        kv_seq_len,
        out_paged
    );

    const std::vector<float> out_contig_cpu =
        copy_tensor_to_vector(out_contig);

    const std::vector<float> out_paged_cpu =
        copy_tensor_to_vector(out_paged);

    expect_vector_close(out_contig_cpu, out_paged_cpu);
}

static void test_paged_cuda_cross_page_matches_contiguous_cuda() {
    constexpr int64_t num_layers = 1;
    constexpr int64_t capacity = 10;
    constexpr int64_t page_size = 3;
    constexpr int64_t num_q_heads = 2;
    constexpr int64_t num_kv_heads = 2;
    constexpr int64_t head_dim = 4;
    constexpr int64_t kv_seq_len = 8;

    ModelPagedKVCache paged_cache(
        num_layers,
        capacity,
        page_size,
        num_kv_heads,
        head_dim,
        DType::FP32,
        Device::CUDA
    );

    BlockTableManager table_manager(paged_cache.num_blocks());

    const int64_t table_idx = table_manager.allocate_table();

    table_manager.ensure_blocks(table_idx, kv_seq_len, page_size);

    Tensor query(
        std::vector<int64_t>{1, num_q_heads, head_dim},
        DType::FP32,
        Device::CUDA
    );

    Tensor key_contig(
        std::vector<int64_t>{kv_seq_len, num_kv_heads, head_dim},
        DType::FP32,
        Device::CUDA
    );

    Tensor value_contig(
        std::vector<int64_t>{kv_seq_len, num_kv_heads, head_dim},
        DType::FP32,
        Device::CUDA
    );

    Tensor out_contig(
        std::vector<int64_t>{1, num_q_heads, head_dim},
        DType::FP32,
        Device::CUDA
    );

    Tensor out_paged(
        std::vector<int64_t>{1, num_q_heads, head_dim},
        DType::FP32,
        Device::CUDA
    );

    std::vector<float> query_cpu(query.numel());
    std::vector<float> key_cpu(key_contig.numel());
    std::vector<float> value_cpu(value_contig.numel());

    fill_cpu_vector(query_cpu, 0.3f);
    fill_cpu_vector(key_cpu, 0.2f);
    fill_cpu_vector(value_cpu, 2.0f);

    copy_vector_to_tensor(query_cpu, query);
    copy_vector_to_tensor(key_cpu, key_contig);
    copy_vector_to_tensor(value_cpu, value_contig);

    paged_cache.update_layer(
        0,
        table_manager,
        table_idx,
        0,
        key_contig,
        value_contig
    );

    flash_attention_kv_cache(
        query,
        key_contig,
        value_contig,
        kv_seq_len,
        out_contig
    );

    flash_attention_paged_kv_cache_cuda(
        query,
        paged_cache,
        table_manager,
        table_idx,
        0,
        kv_seq_len,
        out_paged
    );

    const std::vector<float> out_contig_cpu =
        copy_tensor_to_vector(out_contig);

    const std::vector<float> out_paged_cpu =
        copy_tensor_to_vector(out_paged);

    expect_vector_close(out_contig_cpu, out_paged_cpu);
}

static void test_paged_cuda_prefill_then_decode_write() {
    constexpr int64_t num_layers = 1;
    constexpr int64_t capacity = 8;
    constexpr int64_t page_size = 2;
    constexpr int64_t num_q_heads = 2;
    constexpr int64_t num_kv_heads = 2;
    constexpr int64_t head_dim = 4;
    constexpr int64_t prefill_len = 3;
    constexpr int64_t decode_len = 1;
    constexpr int64_t kv_seq_len = prefill_len + decode_len;

    ModelPagedKVCache paged_cache(
        num_layers,
        capacity,
        page_size,
        num_kv_heads,
        head_dim,
        DType::FP32,
        Device::CUDA
    );

    BlockTableManager table_manager(paged_cache.num_blocks());

    const int64_t table_idx = table_manager.allocate_table();

    table_manager.ensure_blocks(table_idx, kv_seq_len, page_size);

    Tensor query(
        std::vector<int64_t>{1, num_q_heads, head_dim},
        DType::FP32,
        Device::CUDA
    );

    Tensor key_prefill(
        std::vector<int64_t>{prefill_len, num_kv_heads, head_dim},
        DType::FP32,
        Device::CUDA
    );

    Tensor value_prefill(
        std::vector<int64_t>{prefill_len, num_kv_heads, head_dim},
        DType::FP32,
        Device::CUDA
    );

    Tensor key_decode(
        std::vector<int64_t>{decode_len, num_kv_heads, head_dim},
        DType::FP32,
        Device::CUDA
    );

    Tensor value_decode(
        std::vector<int64_t>{decode_len, num_kv_heads, head_dim},
        DType::FP32,
        Device::CUDA
    );

    Tensor key_contig(
        std::vector<int64_t>{kv_seq_len, num_kv_heads, head_dim},
        DType::FP32,
        Device::CUDA
    );

    Tensor value_contig(
        std::vector<int64_t>{kv_seq_len, num_kv_heads, head_dim},
        DType::FP32,
        Device::CUDA
    );

    Tensor out_contig(
        std::vector<int64_t>{1, num_q_heads, head_dim},
        DType::FP32,
        Device::CUDA
    );

    Tensor out_paged(
        std::vector<int64_t>{1, num_q_heads, head_dim},
        DType::FP32,
        Device::CUDA
    );

    std::vector<float> query_cpu(query.numel());
    std::vector<float> key_prefill_cpu(key_prefill.numel());
    std::vector<float> value_prefill_cpu(value_prefill.numel());
    std::vector<float> key_decode_cpu(key_decode.numel());
    std::vector<float> value_decode_cpu(value_decode.numel());

    fill_cpu_vector(query_cpu, 0.5f);
    fill_cpu_vector(key_prefill_cpu, 0.1f);
    fill_cpu_vector(value_prefill_cpu, 1.0f);
    fill_cpu_vector(key_decode_cpu, 10.0f);
    fill_cpu_vector(value_decode_cpu, 20.0f);

    copy_vector_to_tensor(query_cpu, query);
    copy_vector_to_tensor(key_prefill_cpu, key_prefill);
    copy_vector_to_tensor(value_prefill_cpu, value_prefill);
    copy_vector_to_tensor(key_decode_cpu, key_decode);
    copy_vector_to_tensor(value_decode_cpu, value_decode);

    paged_cache.update_layer(
        0,
        table_manager,
        table_idx,
        0,
        key_prefill,
        value_prefill
    );

    paged_cache.update_layer(
        0,
        table_manager,
        table_idx,
        prefill_len,
        key_decode,
        value_decode
    );

    paged_cache.gather_layer_key(
        0,
        table_manager,
        table_idx,
        0,
        kv_seq_len,
        key_contig
    );

    paged_cache.gather_layer_value(
        0,
        table_manager,
        table_idx,
        0,
        kv_seq_len,
        value_contig
    );

    flash_attention_kv_cache(
        query,
        key_contig,
        value_contig,
        kv_seq_len,
        out_contig
    );

    flash_attention_paged_kv_cache_cuda(
        query,
        paged_cache,
        table_manager,
        table_idx,
        0,
        kv_seq_len,
        out_paged
    );

    const std::vector<float> out_contig_cpu =
        copy_tensor_to_vector(out_contig);

    const std::vector<float> out_paged_cpu =
        copy_tensor_to_vector(out_paged);

    expect_vector_close(out_contig_cpu, out_paged_cpu);
}

static void test_paged_cuda_batch_matches_single_cuda() {
    constexpr int64_t num_layers = 1;
    constexpr int64_t capacity = 16;
    constexpr int64_t page_size = 3;
    constexpr int64_t batch_size = 2;
    constexpr int64_t num_q_heads = 4;
    constexpr int64_t num_kv_heads = 2;
    constexpr int64_t head_dim = 4;
    constexpr int64_t kv_seq_len0 = 5;
    constexpr int64_t kv_seq_len1 = 8;

    ModelPagedKVCache paged_cache(
        num_layers,
        capacity,
        page_size,
        num_kv_heads,
        head_dim,
        DType::FP32,
        Device::CUDA
    );

    BlockTableManager table_manager(paged_cache.num_blocks());

    const int64_t table_idx0 = table_manager.allocate_table();
    const int64_t table_idx1 = table_manager.allocate_table();

    table_manager.ensure_blocks(table_idx0, kv_seq_len0, page_size);
    table_manager.ensure_blocks(table_idx1, kv_seq_len1, page_size);

    Tensor query_batch(
        std::vector<int64_t>{batch_size, num_q_heads, head_dim},
        DType::FP32,
        Device::CUDA
    );

    Tensor query0(
        std::vector<int64_t>{1, num_q_heads, head_dim},
        DType::FP32,
        Device::CUDA
    );

    Tensor query1(
        std::vector<int64_t>{1, num_q_heads, head_dim},
        DType::FP32,
        Device::CUDA
    );

    Tensor key0(
        std::vector<int64_t>{kv_seq_len0, num_kv_heads, head_dim},
        DType::FP32,
        Device::CUDA
    );

    Tensor value0(
        std::vector<int64_t>{kv_seq_len0, num_kv_heads, head_dim},
        DType::FP32,
        Device::CUDA
    );

    Tensor key1(
        std::vector<int64_t>{kv_seq_len1, num_kv_heads, head_dim},
        DType::FP32,
        Device::CUDA
    );

    Tensor value1(
        std::vector<int64_t>{kv_seq_len1, num_kv_heads, head_dim},
        DType::FP32,
        Device::CUDA
    );

    Tensor out_batch(
        std::vector<int64_t>{batch_size, num_q_heads, head_dim},
        DType::FP32,
        Device::CUDA
    );

    Tensor out0(
        std::vector<int64_t>{1, num_q_heads, head_dim},
        DType::FP32,
        Device::CUDA
    );

    Tensor out1(
        std::vector<int64_t>{1, num_q_heads, head_dim},
        DType::FP32,
        Device::CUDA
    );

    Tensor out_expected(
        std::vector<int64_t>{batch_size, num_q_heads, head_dim},
        DType::FP32,
        Device::CUDA
    );

    std::vector<float> query_cpu(query_batch.numel());
    std::vector<float> key0_cpu(key0.numel());
    std::vector<float> value0_cpu(value0.numel());
    std::vector<float> key1_cpu(key1.numel());
    std::vector<float> value1_cpu(value1.numel());

    fill_cpu_vector(query_cpu, 0.25f);
    fill_cpu_vector(key0_cpu, 0.1f);
    fill_cpu_vector(value0_cpu, 1.0f);
    fill_cpu_vector(key1_cpu, 0.7f);
    fill_cpu_vector(value1_cpu, 2.0f);

    copy_vector_to_tensor(query_cpu, query_batch);
    copy_vector_to_tensor(key0_cpu, key0);
    copy_vector_to_tensor(value0_cpu, value0);
    copy_vector_to_tensor(key1_cpu, key1);
    copy_vector_to_tensor(value1_cpu, value1);

    const size_t query_row_bytes =
        static_cast<size_t>(num_q_heads * head_dim) * sizeof(float);

    query0.copy_from_tensor(
        query_batch,
        0,
        0,
        query_row_bytes
    );

    query1.copy_from_tensor(
        query_batch,
        0,
        query_row_bytes,
        query_row_bytes
    );

    paged_cache.update_layer(
        0,
        table_manager,
        table_idx0,
        0,
        key0,
        value0
    );

    paged_cache.update_layer(
        0,
        table_manager,
        table_idx1,
        0,
        key1,
        value1
    );

    flash_attention_paged_kv_cache_cuda(
        query0,
        paged_cache,
        table_manager,
        table_idx0,
        0,
        kv_seq_len0,
        out0
    );

    flash_attention_paged_kv_cache_cuda(
        query1,
        paged_cache,
        table_manager,
        table_idx1,
        0,
        kv_seq_len1,
        out1
    );

    flash_attention_paged_kv_cache_batch_cuda(
        query_batch,
        paged_cache,
        table_manager,
        std::vector<int64_t>{table_idx0, table_idx1},
        0,
        std::vector<int64_t>{kv_seq_len0, kv_seq_len1},
        out_batch
    );

    out_expected.copy_from_tensor(
        out0,
        0,
        0,
        query_row_bytes
    );

    out_expected.copy_from_tensor(
        out1,
        query_row_bytes,
        0,
        query_row_bytes
    );

    const std::vector<float> out_batch_cpu =
        copy_tensor_to_vector(out_batch);

    const std::vector<float> out_expected_cpu =
        copy_tensor_to_vector(out_expected);

    expect_vector_close(out_batch_cpu, out_expected_cpu);
}

static void test_unallocated_page_should_throw() {
    constexpr int64_t num_layers = 1;
    constexpr int64_t capacity = 4;
    constexpr int64_t page_size = 2;
    constexpr int64_t num_q_heads = 2;
    constexpr int64_t num_kv_heads = 2;
    constexpr int64_t head_dim = 4;
    constexpr int64_t kv_seq_len = 3;

    ModelPagedKVCache paged_cache(
        num_layers,
        capacity,
        page_size,
        num_kv_heads,
        head_dim,
        DType::FP32,
        Device::CUDA
    );

    BlockTableManager table_manager(paged_cache.num_blocks());

    const int64_t table_idx = table_manager.allocate_table();

    // 这里只分配 token 0,1 对应的 page。
    // 但是下面 kv_seq_len = 3，需要访问 token 2，所以应该抛异常。
    table_manager.ensure_blocks(table_idx, 2, page_size);

    Tensor query(
        std::vector<int64_t>{1, num_q_heads, head_dim},
        DType::FP32,
        Device::CUDA
    );

    Tensor output(
        std::vector<int64_t>{1, num_q_heads, head_dim},
        DType::FP32,
        Device::CUDA
    );

    std::vector<float> query_cpu(query.numel());
    fill_cpu_vector(query_cpu, 0.5f);
    copy_vector_to_tensor(query_cpu, query);

    const bool thrown = thrown_runtime_error([&]() {
        flash_attention_paged_kv_cache_cuda(
            query,
            paged_cache,
            table_manager,
            table_idx,
            0,
            kv_seq_len,
            output
        );
    });

    assert(thrown);
}

static void test_invalid_kv_seq_len_should_throw() {
    constexpr int64_t num_layers = 1;
    constexpr int64_t capacity = 4;
    constexpr int64_t page_size = 2;
    constexpr int64_t num_q_heads = 2;
    constexpr int64_t num_kv_heads = 2;
    constexpr int64_t head_dim = 4;

    ModelPagedKVCache paged_cache(
        num_layers,
        capacity,
        page_size,
        num_kv_heads,
        head_dim,
        DType::FP32,
        Device::CUDA
    );

    BlockTableManager table_manager(paged_cache.num_blocks());

    const int64_t table_idx = table_manager.allocate_table();

    Tensor query(
        std::vector<int64_t>{1, num_q_heads, head_dim},
        DType::FP32,
        Device::CUDA
    );

    Tensor output(
        std::vector<int64_t>{1, num_q_heads, head_dim},
        DType::FP32,
        Device::CUDA
    );

    std::vector<float> query_cpu(query.numel());
    fill_cpu_vector(query_cpu, 0.5f);
    copy_vector_to_tensor(query_cpu, query);

    assert(thrown_runtime_error([&]() {
        flash_attention_paged_kv_cache_cuda(
            query,
            paged_cache,
            table_manager,
            table_idx,
            0,
            0,
            output
        );
    }));

    assert(thrown_runtime_error([&]() {
        flash_attention_paged_kv_cache_cuda(
            query,
            paged_cache,
            table_manager,
            table_idx,
            0,
            -1,
            output
        );
    }));
}

int main() {
    test_paged_cuda_matches_contiguous_cuda();
    test_paged_cuda_gqa_matches_contiguous_cuda();
    test_paged_cuda_cross_page_matches_contiguous_cuda();
    test_paged_cuda_prefill_then_decode_write();
    test_paged_cuda_batch_matches_single_cuda();
    test_unallocated_page_should_throw();
    test_invalid_kv_seq_len_should_throw();

    std::cout << "test_paged_attention_cuda passed" << std::endl;
    return 0;
}

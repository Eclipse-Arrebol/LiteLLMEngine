// tests/test_paged_kv_cache.cpp

#include "engine/block_table_manager.hpp"
#include "engine/paged_kv_cache.hpp"

#include <cassert>
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

static int64_t flat_index(
    int64_t token_idx,
    int64_t kv_head,
    int64_t dim,
    int64_t num_kv_heads,
    int64_t head_dim
) {
    return (token_idx * num_kv_heads + kv_head) * head_dim + dim;
}

static void fill_tensor_seq(Tensor& tensor, float base) {
    float* p = tensor.ptr<float>();

    for (size_t i = 0; i < tensor.numel(); ++i) {
        p[i] = base + static_cast<float>(i);
    }
}

static void fill_kv(
    Tensor& key,
    Tensor& value,
    float key_base,
    float value_base
) {
    fill_tensor_seq(key, key_base);
    fill_tensor_seq(value, value_base);
}

static void expect_token_equal(
    const Tensor& src,
    int64_t src_token_idx,
    const Tensor& dst,
    int64_t dst_token_idx,
    int64_t num_kv_heads,
    int64_t head_dim
) {
    const float* src_ptr = src.ptr<float>();
    const float* dst_ptr = dst.ptr<float>();

    for (int64_t h = 0; h < num_kv_heads; ++h) {
        for (int64_t d = 0; d < head_dim; ++d) {
            const int64_t src_idx =
                flat_index(src_token_idx, h, d, num_kv_heads, head_dim);

            const int64_t dst_idx =
                flat_index(dst_token_idx, h, d, num_kv_heads, head_dim);

            assert(dst_ptr[dst_idx] == src_ptr[src_idx]);
        }
    }
}

static void test_constructor() {
    ModelPagedKVCache cache(
        2,          // num_layers
        5,          // capacity
        2,          // page_size
        2,          // num_kv_heads
        3,          // head_dim
        DType::FP32,
        Device::CPU
    );

    assert(cache.num_layers() == 2);
    assert(cache.capacity() == 5);
    assert(cache.page_size() == 2);
    assert(cache.num_blocks() == 3);
    assert(cache.num_kv_heads() == 2);
    assert(cache.head_dim() == 3);

    const LayerPagedKVCache& layer0 = cache.layer(0);

    assert(layer0.key_pool.shape().size() == 3);
    assert(layer0.key_pool.shape()[0] == 6);
    assert(layer0.key_pool.shape()[1] == 2);
    assert(layer0.key_pool.shape()[2] == 3);

    assert(layer0.value_pool.shape().size() == 3);
    assert(layer0.value_pool.shape()[0] == 6);
    assert(layer0.value_pool.shape()[1] == 2);
    assert(layer0.value_pool.shape()[2] == 3);
}

static void test_prefill_update() {
    constexpr int64_t num_kv_heads = 2;
    constexpr int64_t head_dim = 2;
    constexpr int64_t page_size = 2;

    ModelPagedKVCache cache(
        1,
        5,
        page_size,
        num_kv_heads,
        head_dim,
        DType::FP32,
        Device::CPU
    );

    BlockTableManager table_manager(cache.num_blocks());

    const int64_t table_idx = table_manager.allocate_table();

    // prefill 5 tokens
    table_manager.ensure_blocks(table_idx, 5, page_size);

    Tensor key(
        std::vector<int64_t>{5, num_kv_heads, head_dim},
        DType::FP32,
        Device::CPU
    );

    Tensor value(
        std::vector<int64_t>{5, num_kv_heads, head_dim},
        DType::FP32,
        Device::CPU
    );

    fill_kv(key, value, 10.0f, 100.0f);

    cache.update_layer(
        0,
        table_manager,
        table_idx,
        0,
        key,
        value
    );

    const LayerPagedKVCache& layer0 = cache.layer(0);

    for (int64_t token_idx = 0; token_idx < 5; ++token_idx) {
        const int64_t physical_token_idx =
            table_manager.physical_token_index(
                table_idx,
                token_idx,
                page_size
            );

        expect_token_equal(
            key,
            token_idx,
            layer0.key_pool,
            physical_token_idx,
            num_kv_heads,
            head_dim
        );

        expect_token_equal(
            value,
            token_idx,
            layer0.value_pool,
            physical_token_idx,
            num_kv_heads,
            head_dim
        );
    }
}

static void test_decode_update_with_start_pos() {
    constexpr int64_t num_kv_heads = 2;
    constexpr int64_t head_dim = 2;
    constexpr int64_t page_size = 2;

    ModelPagedKVCache cache(
        1,
        5,
        page_size,
        num_kv_heads,
        head_dim,
        DType::FP32,
        Device::CPU
    );

    BlockTableManager table_manager(cache.num_blocks());

    const int64_t table_idx = table_manager.allocate_table();

    // 先保证 token 0,1,2,3 都有 block
    table_manager.ensure_blocks(table_idx, 4, page_size);

    Tensor key_prefill(
        std::vector<int64_t>{3, num_kv_heads, head_dim},
        DType::FP32,
        Device::CPU
    );

    Tensor value_prefill(
        std::vector<int64_t>{3, num_kv_heads, head_dim},
        DType::FP32,
        Device::CPU
    );

    Tensor key_decode(
        std::vector<int64_t>{1, num_kv_heads, head_dim},
        DType::FP32,
        Device::CPU
    );

    Tensor value_decode(
        std::vector<int64_t>{1, num_kv_heads, head_dim},
        DType::FP32,
        Device::CPU
    );

    fill_kv(key_prefill, value_prefill, 10.0f, 100.0f);
    fill_kv(key_decode, value_decode, 1000.0f, 2000.0f);

    // prefill 写 token 0,1,2
    cache.update_layer(
        0,
        table_manager,
        table_idx,
        0,
        key_prefill,
        value_prefill
    );

    // decode 写 token 3
    cache.update_layer(
        0,
        table_manager,
        table_idx,
        3,
        key_decode,
        value_decode
    );

    const LayerPagedKVCache& layer0 = cache.layer(0);

    for (int64_t token_idx = 0; token_idx < 3; ++token_idx) {
        const int64_t physical_token_idx =
            table_manager.physical_token_index(
                table_idx,
                token_idx,
                page_size
            );

        expect_token_equal(
            key_prefill,
            token_idx,
            layer0.key_pool,
            physical_token_idx,
            num_kv_heads,
            head_dim
        );

        expect_token_equal(
            value_prefill,
            token_idx,
            layer0.value_pool,
            physical_token_idx,
            num_kv_heads,
            head_dim
        );
    }

    {
        const int64_t physical_token_idx =
            table_manager.physical_token_index(
                table_idx,
                3,
                page_size
            );

        expect_token_equal(
            key_decode,
            0,
            layer0.key_pool,
            physical_token_idx,
            num_kv_heads,
            head_dim
        );

        expect_token_equal(
            value_decode,
            0,
            layer0.value_pool,
            physical_token_idx,
            num_kv_heads,
            head_dim
        );
    }
}

static void test_two_requests_do_not_overlap() {
    constexpr int64_t num_kv_heads = 1;
    constexpr int64_t head_dim = 2;
    constexpr int64_t page_size = 2;

    ModelPagedKVCache cache(
        1,
        6,
        page_size,
        num_kv_heads,
        head_dim,
        DType::FP32,
        Device::CPU
    );

    BlockTableManager table_manager(cache.num_blocks());

    const int64_t table0 = table_manager.allocate_table();
    const int64_t table1 = table_manager.allocate_table();

    table_manager.ensure_blocks(table0, 2, page_size);
    table_manager.ensure_blocks(table1, 2, page_size);

    assert(table0 == 0);
    assert(table1 == 1);

    assert(table_manager.table(table0)[0] == 0);
    assert(table_manager.table(table1)[0] == 1);

    Tensor key0(
        std::vector<int64_t>{2, num_kv_heads, head_dim},
        DType::FP32,
        Device::CPU
    );

    Tensor value0(
        std::vector<int64_t>{2, num_kv_heads, head_dim},
        DType::FP32,
        Device::CPU
    );

    Tensor key1(
        std::vector<int64_t>{2, num_kv_heads, head_dim},
        DType::FP32,
        Device::CPU
    );

    Tensor value1(
        std::vector<int64_t>{2, num_kv_heads, head_dim},
        DType::FP32,
        Device::CPU
    );

    fill_kv(key0, value0, 10.0f, 100.0f);
    fill_kv(key1, value1, 1000.0f, 2000.0f);

    cache.update_layer(
        0,
        table_manager,
        table0,
        0,
        key0,
        value0
    );

    cache.update_layer(
        0,
        table_manager,
        table1,
        0,
        key1,
        value1
    );

    const LayerPagedKVCache& layer0 = cache.layer(0);

    for (int64_t token_idx = 0; token_idx < 2; ++token_idx) {
        const int64_t physical0 =
            table_manager.physical_token_index(
                table0,
                token_idx,
                page_size
            );

        const int64_t physical1 =
            table_manager.physical_token_index(
                table1,
                token_idx,
                page_size
            );

        expect_token_equal(
            key0,
            token_idx,
            layer0.key_pool,
            physical0,
            num_kv_heads,
            head_dim
        );

        expect_token_equal(
            value0,
            token_idx,
            layer0.value_pool,
            physical0,
            num_kv_heads,
            head_dim
        );

        expect_token_equal(
            key1,
            token_idx,
            layer0.key_pool,
            physical1,
            num_kv_heads,
            head_dim
        );

        expect_token_equal(
            value1,
            token_idx,
            layer0.value_pool,
            physical1,
            num_kv_heads,
            head_dim
        );
    }
}

static void test_reset() {
    constexpr int64_t num_kv_heads = 1;
    constexpr int64_t head_dim = 2;
    constexpr int64_t page_size = 2;

    ModelPagedKVCache cache(
        1,
        4,
        page_size,
        num_kv_heads,
        head_dim,
        DType::FP32,
        Device::CPU
    );

    BlockTableManager table_manager(cache.num_blocks());

    const int64_t table_idx = table_manager.allocate_table();

    table_manager.ensure_blocks(table_idx, 2, page_size);

    Tensor key(
        std::vector<int64_t>{2, num_kv_heads, head_dim},
        DType::FP32,
        Device::CPU
    );

    Tensor value(
        std::vector<int64_t>{2, num_kv_heads, head_dim},
        DType::FP32,
        Device::CPU
    );

    fill_kv(key, value, 10.0f, 100.0f);

    cache.update_layer(
        0,
        table_manager,
        table_idx,
        0,
        key,
        value
    );

    cache.reset();

    const LayerPagedKVCache& layer0 = cache.layer(0);

    const float* key_pool = layer0.key_pool.ptr<float>();
    const float* value_pool = layer0.value_pool.ptr<float>();

    for (size_t i = 0; i < layer0.key_pool.numel(); ++i) {
        assert(key_pool[i] == 0.0f);
    }

    for (size_t i = 0; i < layer0.value_pool.numel(); ++i) {
        assert(value_pool[i] == 0.0f);
    }
}

static void test_unallocated_logical_page_should_throw() {
    constexpr int64_t num_kv_heads = 1;
    constexpr int64_t head_dim = 2;
    constexpr int64_t page_size = 2;

    ModelPagedKVCache cache(
        1,
        4,
        page_size,
        num_kv_heads,
        head_dim,
        DType::FP32,
        Device::CPU
    );

    BlockTableManager table_manager(cache.num_blocks());

    const int64_t table_idx = table_manager.allocate_table();

    // 只分配 token 0,1 对应的 page 0
    table_manager.ensure_blocks(table_idx, 2, page_size);

    Tensor key(
        std::vector<int64_t>{1, num_kv_heads, head_dim},
        DType::FP32,
        Device::CPU
    );

    Tensor value(
        std::vector<int64_t>{1, num_kv_heads, head_dim},
        DType::FP32,
        Device::CPU
    );

    fill_kv(key, value, 10.0f, 100.0f);

    // start_pos = 2 属于 logical page 1，但是只分配了 page 0
    const bool thrown = thrown_runtime_error([&]() {
        cache.update_layer(
            0,
            table_manager,
            table_idx,
            2,
            key,
            value
        );
    });

    assert(thrown);
}

static void test_invalid_shape_should_throw() {
    ModelPagedKVCache cache(
        1,
        4,
        2,
        2,
        3,
        DType::FP32,
        Device::CPU
    );

    BlockTableManager table_manager(cache.num_blocks());

    const int64_t table_idx = table_manager.allocate_table();

    table_manager.ensure_blocks(table_idx, 2, 2);

    Tensor key(
        std::vector<int64_t>{2, 2, 3},
        DType::FP32,
        Device::CPU
    );

    Tensor value(
        std::vector<int64_t>{2, 2, 4},
        DType::FP32,
        Device::CPU
    );

    fill_kv(key, value, 10.0f, 100.0f);

    const bool thrown = thrown_runtime_error([&]() {
        cache.update_layer(
            0,
            table_manager,
            table_idx,
            0,
            key,
            value
        );
    });

    assert(thrown);
}

int main() {
    test_constructor();
    test_prefill_update();
    test_decode_update_with_start_pos();
    test_two_requests_do_not_overlap();
    test_reset();
    test_unallocated_logical_page_should_throw();
    test_invalid_shape_should_throw();

    std::cout << "test_paged_kv_cache passed" << std::endl;
    return 0;
}
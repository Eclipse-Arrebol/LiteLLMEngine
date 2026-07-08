// tests/test_paged_kv_cache.cpp

#include "engine/paged_kv_cache.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace lite_llm;

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
    assert(cache.current_len() == 0);

    const LayerPagedKVCache& layer0 = cache.layer(0);

    assert(layer0.page_table.size() == 3);

    assert(layer0.page_table[0].key_block_id == 0);
    assert(layer0.page_table[0].value_block_id == 0);
    assert(layer0.page_table[1].key_block_id == 1);
    assert(layer0.page_table[1].value_block_id == 1);
    assert(layer0.page_table[2].key_block_id == 2);
    assert(layer0.page_table[2].value_block_id == 2);

    assert(layer0.key_pool.shape().size() == 3);
    assert(layer0.key_pool.shape()[0] == 6);
    assert(layer0.key_pool.shape()[1] == 2);
    assert(layer0.key_pool.shape()[2] == 3);

    assert(layer0.value_pool.shape().size() == 3);
    assert(layer0.value_pool.shape()[0] == 6);
    assert(layer0.value_pool.shape()[1] == 2);
    assert(layer0.value_pool.shape()[2] == 3);
}

static void test_update_without_advance() {
    constexpr int64_t num_kv_heads = 2;
    constexpr int64_t head_dim = 3;

    ModelPagedKVCache cache(
        1,
        4,
        2,
        num_kv_heads,
        head_dim,
        DType::FP32,
        Device::CPU
    );

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

    cache.update_layer(0, key, value);

    // update_layer 只写数据，不推进 current_len
    assert(cache.current_len() == 0);

    const LayerPagedKVCache& layer0 = cache.layer(0);

    expect_token_equal(
        key,
        0,
        layer0.key_pool,
        0,
        num_kv_heads,
        head_dim
    );

    expect_token_equal(
        key,
        1,
        layer0.key_pool,
        1,
        num_kv_heads,
        head_dim
    );

    expect_token_equal(
        value,
        0,
        layer0.value_pool,
        0,
        num_kv_heads,
        head_dim
    );

    expect_token_equal(
        value,
        1,
        layer0.value_pool,
        1,
        num_kv_heads,
        head_dim
    );

    cache.advance(2);
    assert(cache.current_len() == 2);
}

static void test_cross_page_update() {
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

    cache.update_layer(0, key, value);

    const LayerPagedKVCache& layer0 = cache.layer(0);

    for (int64_t token_idx = 0; token_idx < 5; ++token_idx) {
        const int64_t physical_key_token_idx =
            layer0.physical_key_token_index(token_idx, page_size);

        const int64_t physical_value_token_idx =
            layer0.physical_value_token_index(token_idx, page_size);

        expect_token_equal(
            key,
            token_idx,
            layer0.key_pool,
            physical_key_token_idx,
            num_kv_heads,
            head_dim
        );

        expect_token_equal(
            value,
            token_idx,
            layer0.value_pool,
            physical_value_token_idx,
            num_kv_heads,
            head_dim
        );
    }

    cache.advance(5);
    assert(cache.current_len() == 5);
}

static void test_two_step_update_prefill_then_decode() {
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

    Tensor key_decode0(
        std::vector<int64_t>{1, num_kv_heads, head_dim},
        DType::FP32,
        Device::CPU
    );

    Tensor value_decode0(
        std::vector<int64_t>{1, num_kv_heads, head_dim},
        DType::FP32,
        Device::CPU
    );

    Tensor key_decode1(
        std::vector<int64_t>{1, num_kv_heads, head_dim},
        DType::FP32,
        Device::CPU
    );

    Tensor value_decode1(
        std::vector<int64_t>{1, num_kv_heads, head_dim},
        DType::FP32,
        Device::CPU
    );

    fill_kv(key_prefill, value_prefill, 10.0f, 100.0f);
    fill_kv(key_decode0, value_decode0, 1000.0f, 2000.0f);
    fill_kv(key_decode1, value_decode1, 3000.0f, 4000.0f);

    cache.update_layer(0, key_prefill, value_prefill);
    cache.advance(3);

    cache.update_layer(0, key_decode0, value_decode0);
    cache.advance(1);

    cache.update_layer(0, key_decode1, value_decode1);
    cache.advance(1);

    assert(cache.current_len() == 5);

    const LayerPagedKVCache& layer0 = cache.layer(0);

    for (int64_t token_idx = 0; token_idx < 3; ++token_idx) {
        const int64_t physical_key_token_idx =
            layer0.physical_key_token_index(token_idx, page_size);

        const int64_t physical_value_token_idx =
            layer0.physical_value_token_index(token_idx, page_size);

        expect_token_equal(
            key_prefill,
            token_idx,
            layer0.key_pool,
            physical_key_token_idx,
            num_kv_heads,
            head_dim
        );

        expect_token_equal(
            value_prefill,
            token_idx,
            layer0.value_pool,
            physical_value_token_idx,
            num_kv_heads,
            head_dim
        );
    }

    {
        const int64_t logical_token_idx = 3;

        expect_token_equal(
            key_decode0,
            0,
            layer0.key_pool,
            layer0.physical_key_token_index(logical_token_idx, page_size),
            num_kv_heads,
            head_dim
        );

        expect_token_equal(
            value_decode0,
            0,
            layer0.value_pool,
            layer0.physical_value_token_index(logical_token_idx, page_size),
            num_kv_heads,
            head_dim
        );
    }

    {
        const int64_t logical_token_idx = 4;

        expect_token_equal(
            key_decode1,
            0,
            layer0.key_pool,
            layer0.physical_key_token_index(logical_token_idx, page_size),
            num_kv_heads,
            head_dim
        );

        expect_token_equal(
            value_decode1,
            0,
            layer0.value_pool,
            layer0.physical_value_token_index(logical_token_idx, page_size),
            num_kv_heads,
            head_dim
        );
    }
}

static void test_custom_page_table_mapping() {
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

    LayerPagedKVCache& layer0 = cache.layer(0);

    // 手动打乱 logical page -> physical block 映射
    layer0.page_table[0].key_block_id = 2;
    layer0.page_table[0].value_block_id = 1;

    layer0.page_table[1].key_block_id = 0;
    layer0.page_table[1].value_block_id = 2;

    layer0.page_table[2].key_block_id = 1;
    layer0.page_table[2].value_block_id = 0;

    Tensor key(
        std::vector<int64_t>{4, num_kv_heads, head_dim},
        DType::FP32,
        Device::CPU
    );

    Tensor value(
        std::vector<int64_t>{4, num_kv_heads, head_dim},
        DType::FP32,
        Device::CPU
    );

    fill_kv(key, value, 10.0f, 100.0f);

    cache.update_layer(0, key, value);

    // logical token 0,1 属于 logical page 0
    // key 写到 physical block 2 -> physical token 4,5
    // value 写到 physical block 1 -> physical token 2,3
    expect_token_equal(key, 0, layer0.key_pool, 4, num_kv_heads, head_dim);
    expect_token_equal(key, 1, layer0.key_pool, 5, num_kv_heads, head_dim);
    expect_token_equal(value, 0, layer0.value_pool, 2, num_kv_heads, head_dim);
    expect_token_equal(value, 1, layer0.value_pool, 3, num_kv_heads, head_dim);

    // logical token 2,3 属于 logical page 1
    // key 写到 physical block 0 -> physical token 0,1
    // value 写到 physical block 2 -> physical token 4,5
    expect_token_equal(key, 2, layer0.key_pool, 0, num_kv_heads, head_dim);
    expect_token_equal(key, 3, layer0.key_pool, 1, num_kv_heads, head_dim);
    expect_token_equal(value, 2, layer0.value_pool, 4, num_kv_heads, head_dim);
    expect_token_equal(value, 3, layer0.value_pool, 5, num_kv_heads, head_dim);
}

static void test_reset() {
    ModelPagedKVCache cache(
        1,
        4,
        2,
        1,
        2,
        DType::FP32,
        Device::CPU
    );

    Tensor key(
        std::vector<int64_t>{2, 1, 2},
        DType::FP32,
        Device::CPU
    );

    Tensor value(
        std::vector<int64_t>{2, 1, 2},
        DType::FP32,
        Device::CPU
    );

    fill_kv(key, value, 10.0f, 100.0f);

    cache.update_layer(0, key, value);
    cache.advance(2);

    assert(cache.current_len() == 2);

    cache.reset();

    assert(cache.current_len() == 0);

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

static void test_invalid_layer_idx() {
    ModelPagedKVCache cache(
        1,
        4,
        2,
        1,
        2,
        DType::FP32,
        Device::CPU
    );

    bool thrown = false;

    try {
        cache.layer(10);
    } catch (const std::runtime_error&) {
        thrown = true;
    }

    assert(thrown);
}

static void test_capacity_exceeded_by_update() {
    ModelPagedKVCache cache(
        1,
        4,
        2,
        1,
        2,
        DType::FP32,
        Device::CPU
    );

    Tensor key(
        std::vector<int64_t>{5, 1, 2},
        DType::FP32,
        Device::CPU
    );

    Tensor value(
        std::vector<int64_t>{5, 1, 2},
        DType::FP32,
        Device::CPU
    );

    fill_kv(key, value, 10.0f, 100.0f);

    bool thrown = false;

    try {
        cache.update_layer(0, key, value);
    } catch (const std::runtime_error&) {
        thrown = true;
    }

    assert(thrown);
}

static void test_capacity_exceeded_by_advance() {
    ModelPagedKVCache cache(
        1,
        4,
        2,
        1,
        2,
        DType::FP32,
        Device::CPU
    );

    bool thrown = false;

    try {
        cache.advance(5);
    } catch (const std::runtime_error&) {
        thrown = true;
    }

    assert(thrown);
}

static void test_shape_mismatch() {
    ModelPagedKVCache cache(
        1,
        4,
        2,
        2,
        3,
        DType::FP32,
        Device::CPU
    );

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

    bool thrown = false;

    try {
        cache.update_layer(0, key, value);
    } catch (const std::runtime_error&) {
        thrown = true;
    }

    assert(thrown);
}

static void test_seq_len_mismatch() {
    ModelPagedKVCache cache(
        1,
        4,
        2,
        2,
        3,
        DType::FP32,
        Device::CPU
    );

    Tensor key(
        std::vector<int64_t>{2, 2, 3},
        DType::FP32,
        Device::CPU
    );

    Tensor value(
        std::vector<int64_t>{3, 2, 3},
        DType::FP32,
        Device::CPU
    );

    fill_kv(key, value, 10.0f, 100.0f);

    bool thrown = false;

    try {
        cache.update_layer(0, key, value);
    } catch (const std::runtime_error&) {
        thrown = true;
    }

    assert(thrown);
}

static void test_dtype_mismatch() {
    ModelPagedKVCache cache(
        1,
        4,
        2,
        1,
        2,
        DType::FP32,
        Device::CPU
    );

    Tensor key(
        std::vector<int64_t>{2, 1, 2},
        DType::FP16,
        Device::CPU
    );

    Tensor value(
        std::vector<int64_t>{2, 1, 2},
        DType::FP32,
        Device::CPU
    );

    bool thrown = false;

    try {
        cache.update_layer(0, key, value);
    } catch (const std::runtime_error&) {
        thrown = true;
    }

    assert(thrown);
}

int main() {
    test_constructor();
    test_update_without_advance();
    test_cross_page_update();
    test_two_step_update_prefill_then_decode();
    test_custom_page_table_mapping();
    test_reset();
    test_invalid_layer_idx();
    test_capacity_exceeded_by_update();
    test_capacity_exceeded_by_advance();
    test_shape_mismatch();
    test_seq_len_mismatch();
    test_dtype_mismatch();

    std::cout << "test_paged_kv_cache passed" << std::endl;
    return 0;
}
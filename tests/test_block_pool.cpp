// tests/test_block_pool.cpp

#include "engine/block_pool.hpp"

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

static void test_constructor() {
    BlockPool pool(4);

    assert(pool.num_blocks() == 4);
    assert(pool.num_free_blocks() == 4);
    assert(pool.num_used_blocks() == 0);

    assert(!pool.is_used(0));
    assert(!pool.is_used(1));
    assert(!pool.is_used(2));
    assert(!pool.is_used(3));
}

static void test_allocate_one_block() {
    BlockPool pool(4);

    const int32_t block0 = pool.allocate_block();

    assert(block0 == 0);
    assert(pool.num_free_blocks() == 3);
    assert(pool.num_used_blocks() == 1);
    assert(pool.is_used(0));
}

static void test_allocate_multiple_blocks() {
    BlockPool pool(4);

    std::vector<int32_t> blocks = pool.allocate_blocks(3);

    assert(blocks.size() == 3);
    assert(blocks[0] == 0);
    assert(blocks[1] == 1);
    assert(blocks[2] == 2);

    assert(pool.num_free_blocks() == 1);
    assert(pool.num_used_blocks() == 3);

    assert(pool.is_used(0));
    assert(pool.is_used(1));
    assert(pool.is_used(2));
    assert(!pool.is_used(3));
}

static void test_free_one_block() {
    BlockPool pool(4);

    const int32_t block0 = pool.allocate_block();
    const int32_t block1 = pool.allocate_block();

    assert(block0 == 0);
    assert(block1 == 1);

    pool.free_block(block0);

    assert(pool.num_free_blocks() == 3);
    assert(pool.num_used_blocks() == 1);

    assert(!pool.is_used(block0));
    assert(pool.is_used(block1));
}

static void test_reuse_freed_block() {
    BlockPool pool(4);

    const int32_t block0 = pool.allocate_block();
    const int32_t block1 = pool.allocate_block();

    assert(block0 == 0);
    assert(block1 == 1);

    pool.free_block(block0);

    const int32_t reused = pool.allocate_block();

    assert(reused == block0);
    assert(pool.is_used(reused));
    assert(pool.num_free_blocks() == 2);
    assert(pool.num_used_blocks() == 2);
}

static void test_free_multiple_blocks() {
    BlockPool pool(5);

    std::vector<int32_t> blocks = pool.allocate_blocks(3);

    assert(pool.num_free_blocks() == 2);
    assert(pool.num_used_blocks() == 3);

    pool.free_blocks(blocks);

    assert(pool.num_free_blocks() == 5);
    assert(pool.num_used_blocks() == 0);

    for (int32_t block_id : blocks) {
        assert(!pool.is_used(block_id));
    }
}

static void test_allocate_until_full() {
    BlockPool pool(3);

    const int32_t block0 = pool.allocate_block();
    const int32_t block1 = pool.allocate_block();
    const int32_t block2 = pool.allocate_block();

    assert(block0 == 0);
    assert(block1 == 1);
    assert(block2 == 2);

    assert(pool.num_free_blocks() == 0);
    assert(pool.num_used_blocks() == 3);

    const bool thrown = thrown_runtime_error([&]() {
        pool.allocate_block();
    });

    assert(thrown);
}

static void test_allocate_blocks_not_enough() {
    BlockPool pool(3);

    pool.allocate_block();

    assert(pool.num_free_blocks() == 2);

    const bool thrown = thrown_runtime_error([&]() {
        pool.allocate_blocks(3);
    });

    assert(thrown);

    assert(pool.num_free_blocks() == 2);
    assert(pool.num_used_blocks() == 1);
}

static void test_has_free_blocks() {
    BlockPool pool(4);

    assert(pool.has_free_blocks(0));
    assert(pool.has_free_blocks(1));
    assert(pool.has_free_blocks(4));
    assert(!pool.has_free_blocks(5));
    assert(!pool.has_free_blocks(-1));

    pool.allocate_blocks(3);

    assert(pool.has_free_blocks(1));
    assert(!pool.has_free_blocks(2));
}

static void test_double_free_should_throw() {
    BlockPool pool(3);

    const int32_t block0 = pool.allocate_block();

    pool.free_block(block0);

    const bool thrown = thrown_runtime_error([&]() {
        pool.free_block(block0);
    });

    assert(thrown);
}

static void test_invalid_block_id_should_throw() {
    BlockPool pool(3);

    {
        const bool thrown = thrown_runtime_error([&]() {
            pool.is_used(-1);
        });

        assert(thrown);
    }

    {
        const bool thrown = thrown_runtime_error([&]() {
            pool.is_used(3);
        });

        assert(thrown);
    }

    {
        const bool thrown = thrown_runtime_error([&]() {
            pool.free_block(100);
        });

        assert(thrown);
    }
}

static void test_reset() {
    BlockPool pool(4);

    pool.allocate_blocks(3);

    assert(pool.num_free_blocks() == 1);
    assert(pool.num_used_blocks() == 3);

    pool.reset();

    assert(pool.num_free_blocks() == 4);
    assert(pool.num_used_blocks() == 0);

    assert(!pool.is_used(0));
    assert(!pool.is_used(1));
    assert(!pool.is_used(2));
    assert(!pool.is_used(3));

    const int32_t block0 = pool.allocate_block();
    assert(block0 == 0);
}

static void test_zero_blocks_should_throw() {
    const bool thrown = thrown_runtime_error([]() {
        BlockPool pool(0);
    });

    assert(thrown);
}

static void test_negative_blocks_should_throw() {
    const bool thrown = thrown_runtime_error([]() {
        BlockPool pool(-1);
    });

    assert(thrown);
}

int main() {
    test_constructor();
    test_allocate_one_block();
    test_allocate_multiple_blocks();
    test_free_one_block();
    test_reuse_freed_block();
    test_free_multiple_blocks();
    test_allocate_until_full();
    test_allocate_blocks_not_enough();
    test_has_free_blocks();
    test_double_free_should_throw();
    test_invalid_block_id_should_throw();
    test_reset();
    test_zero_blocks_should_throw();
    test_negative_blocks_should_throw();

    std::cout << "test_block_pool passed" << std::endl;
    return 0;
}
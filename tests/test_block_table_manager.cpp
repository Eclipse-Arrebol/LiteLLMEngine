// tests/test_block_table_manager.cpp

#include "engine/block_table_manager.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <stdexcept>

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
    BlockTableManager manager(4);

    assert(manager.num_blocks() == 4);
    assert(manager.num_free_blocks() == 4);
    assert(manager.num_used_blocks() == 0);

    assert(manager.num_tables() == 0);
    assert(manager.num_used_tables() == 0);
    assert(manager.num_free_tables() == 0);

    assert(!manager.is_block_used(0));
    assert(!manager.is_block_used(1));
    assert(!manager.is_block_used(2));
    assert(!manager.is_block_used(3));
}

static void test_allocate_table() {
    BlockTableManager manager(4);

    const int64_t idx = manager.allocate_table();

    assert(idx == 0);
    assert(manager.num_tables() == 1);
    assert(manager.num_used_tables() == 1);
    assert(manager.num_free_tables() == 0);
    assert(manager.is_table_used(idx));
    assert(manager.table(idx).empty());

    // allocate_table 只分配 table，不分配 physical block
    assert(manager.num_used_blocks() == 0);
    assert(manager.num_free_blocks() == 4);
}

static void test_multiple_tables() {
    BlockTableManager manager(4);

    const int64_t idx0 = manager.allocate_table();
    const int64_t idx1 = manager.allocate_table();

    assert(idx0 == 0);
    assert(idx1 == 1);

    assert(manager.num_tables() == 2);
    assert(manager.num_used_tables() == 2);
    assert(manager.num_free_tables() == 0);

    assert(manager.table(idx0).empty());
    assert(manager.table(idx1).empty());
}

static void test_ensure_blocks() {
    BlockTableManager manager(4);

    const int64_t idx = manager.allocate_table();

    // token_len = 5, page_size = 2
    // required blocks = ceil(5 / 2) = 3
    manager.ensure_blocks(idx, 5, 2);

    const auto& table = manager.table(idx);

    assert(table.size() == 3);
    assert(table[0] == 0);
    assert(table[1] == 1);
    assert(table[2] == 2);

    assert(manager.num_used_blocks() == 3);
    assert(manager.num_free_blocks() == 1);

    assert(manager.is_block_used(0));
    assert(manager.is_block_used(1));
    assert(manager.is_block_used(2));
    assert(!manager.is_block_used(3));
}

static void test_ensure_blocks_incremental() {
    BlockTableManager manager(5);

    const int64_t idx = manager.allocate_table();

    // token_len = 1, page_size = 2，需要 1 个 block
    manager.ensure_blocks(idx, 1, 2);

    assert(manager.table(idx).size() == 1);
    assert(manager.table(idx)[0] == 0);
    assert(manager.num_used_blocks() == 1);
    assert(manager.num_free_blocks() == 4);

    // token_len = 2，仍然只需要 1 个 block
    manager.ensure_blocks(idx, 2, 2);

    assert(manager.table(idx).size() == 1);
    assert(manager.table(idx)[0] == 0);
    assert(manager.num_used_blocks() == 1);
    assert(manager.num_free_blocks() == 4);

    // token_len = 3，需要第 2 个 block
    manager.ensure_blocks(idx, 3, 2);

    assert(manager.table(idx).size() == 2);
    assert(manager.table(idx)[0] == 0);
    assert(manager.table(idx)[1] == 1);
    assert(manager.num_used_blocks() == 2);
    assert(manager.num_free_blocks() == 3);

    // token_len = 5，需要第 3 个 block
    manager.ensure_blocks(idx, 5, 2);

    assert(manager.table(idx).size() == 3);
    assert(manager.table(idx)[0] == 0);
    assert(manager.table(idx)[1] == 1);
    assert(manager.table(idx)[2] == 2);
    assert(manager.num_used_blocks() == 3);
    assert(manager.num_free_blocks() == 2);
}

static void test_multiple_tables_get_different_blocks() {
    BlockTableManager manager(5);

    const int64_t idx0 = manager.allocate_table();
    const int64_t idx1 = manager.allocate_table();

    // idx0 需要 2 个 block
    manager.ensure_blocks(idx0, 4, 2);

    // idx1 也需要 2 个 block
    manager.ensure_blocks(idx1, 3, 2);

    const auto& table0 = manager.table(idx0);
    const auto& table1 = manager.table(idx1);

    assert(table0.size() == 2);
    assert(table1.size() == 2);

    assert(table0[0] == 0);
    assert(table0[1] == 1);

    assert(table1[0] == 2);
    assert(table1[1] == 3);

    assert(manager.num_used_tables() == 2);
    assert(manager.num_used_blocks() == 4);
    assert(manager.num_free_blocks() == 1);

    assert(manager.is_block_used(0));
    assert(manager.is_block_used(1));
    assert(manager.is_block_used(2));
    assert(manager.is_block_used(3));
    assert(!manager.is_block_used(4));
}

static void test_physical_token_index_with_allocated_blocks() {
    BlockTableManager manager(6);

    const int64_t idx = manager.allocate_table();

    // page_size = 2
    // token_len = 5
    // table = [0, 1, 2]
    manager.ensure_blocks(idx, 5, 2);

    assert(manager.physical_token_index(idx, 0, 2) == 0);
    assert(manager.physical_token_index(idx, 1, 2) == 1);
    assert(manager.physical_token_index(idx, 2, 2) == 2);
    assert(manager.physical_token_index(idx, 3, 2) == 3);
    assert(manager.physical_token_index(idx, 4, 2) == 4);
}

static void test_physical_token_index_with_custom_valid_mapping() {
    BlockTableManager manager(10);

    const int64_t idx = manager.allocate_table();

    // 这个测试只是验证公式：
    // logical page -> physical block
    //
    // 注意：这里手动写的是合法 block id，且不会调用 free_table。
    auto& table = manager.table(idx);
    table.push_back(5);
    table.push_back(2);
    table.push_back(7);

    // page_size = 4
    //
    // logical token 0 -> page 0 offset 0 -> block 5 token 20
    // logical token 3 -> page 0 offset 3 -> block 5 token 23
    // logical token 4 -> page 1 offset 0 -> block 2 token 8
    // logical token 9 -> page 2 offset 1 -> block 7 token 29

    assert(manager.physical_token_index(idx, 0, 4) == 20);
    assert(manager.physical_token_index(idx, 3, 4) == 23);
    assert(manager.physical_token_index(idx, 4, 4) == 8);
    assert(manager.physical_token_index(idx, 9, 4) == 29);
}

static void test_free_table_releases_blocks() {
    BlockTableManager manager(4);

    const int64_t idx = manager.allocate_table();

    manager.ensure_blocks(idx, 5, 2);

    assert(manager.num_used_blocks() == 3);
    assert(manager.num_free_blocks() == 1);

    manager.free_table(idx);

    assert(manager.num_tables() == 1);
    assert(manager.num_used_tables() == 0);
    assert(manager.num_free_tables() == 1);
    assert(!manager.is_table_used(idx));

    // free_table 应该同时释放 table 占用的 physical blocks
    assert(manager.num_used_blocks() == 0);
    assert(manager.num_free_blocks() == 4);

    assert(!manager.is_block_used(0));
    assert(!manager.is_block_used(1));
    assert(!manager.is_block_used(2));
    assert(!manager.is_block_used(3));

    assert(thrown_runtime_error([&]() {
        manager.table(idx);
    }));
}

static void test_reuse_table() {
    BlockTableManager manager(4);

    const int64_t idx0 = manager.allocate_table();
    const int64_t idx1 = manager.allocate_table();

    manager.ensure_blocks(idx0, 2, 2);

    assert(manager.table(idx0).size() == 1);
    assert(manager.table(idx0)[0] == 0);
    assert(manager.num_used_blocks() == 1);

    manager.free_table(idx0);

    const int64_t reused = manager.allocate_table();

    assert(reused == idx0);
    assert(manager.is_table_used(reused));
    assert(manager.table(reused).empty());

    assert(manager.is_table_used(idx1));
    assert(manager.num_tables() == 2);
    assert(manager.num_used_tables() == 2);
    assert(manager.num_free_tables() == 0);

    // idx0 的 block 已经释放
    assert(manager.num_used_blocks() == 0);
    assert(manager.num_free_blocks() == 4);
}

static void test_reuse_blocks_after_free_table() {
    BlockTableManager manager(4);

    const int64_t idx0 = manager.allocate_table();

    manager.ensure_blocks(idx0, 4, 2);

    assert(manager.table(idx0).size() == 2);
    assert(manager.table(idx0)[0] == 0);
    assert(manager.table(idx0)[1] == 1);

    manager.free_table(idx0);

    const int64_t idx1 = manager.allocate_table();

    assert(idx1 == idx0);

    manager.ensure_blocks(idx1, 2, 2);

    const auto& table1 = manager.table(idx1);

    assert(table1.size() == 1);

    // free_table 释放 block 0,1 时，free_blocks 是栈式结构。
    // 最后释放的 block 1 会先被重新分配。
    assert(table1[0] == 1);

    assert(manager.num_used_blocks() == 1);
    assert(manager.num_free_blocks() == 3);
}

static void test_not_enough_blocks_should_throw() {
    BlockTableManager manager(2);

    const int64_t idx = manager.allocate_table();

    // token_len = 5, page_size = 2
    // required blocks = 3, 但是 manager 只有 2 个 block
    const bool thrown = thrown_runtime_error([&]() {
        manager.ensure_blocks(idx, 5, 2);
    });

    assert(thrown);

    assert(manager.table(idx).empty());
    assert(manager.num_used_blocks() == 0);
    assert(manager.num_free_blocks() == 2);
}

static void test_double_free_should_throw() {
    BlockTableManager manager(4);

    const int64_t idx = manager.allocate_table();

    manager.free_table(idx);

    assert(thrown_runtime_error([&]() {
        manager.free_table(idx);
    }));
}

static void test_invalid_table_idx_should_throw() {
    BlockTableManager manager(4);

    assert(thrown_runtime_error([&]() {
        manager.table(0);
    }));

    assert(thrown_runtime_error([&]() {
        manager.is_table_used(-1);
    }));

    assert(thrown_runtime_error([&]() {
        manager.free_table(100);
    }));

    assert(thrown_runtime_error([&]() {
        manager.ensure_blocks(100, 1, 2);
    }));
}

static void test_invalid_block_idx_should_throw() {
    BlockTableManager manager(4);

    assert(thrown_runtime_error([&]() {
        manager.is_block_used(-1);
    }));

    assert(thrown_runtime_error([&]() {
        manager.is_block_used(4);
    }));
}

static void test_invalid_physical_token_index_should_throw() {
    BlockTableManager manager(4);

    const int64_t idx = manager.allocate_table();

    assert(thrown_runtime_error([&]() {
        manager.physical_token_index(idx, -1, 2);
    }));

    assert(thrown_runtime_error([&]() {
        manager.physical_token_index(idx, 0, 0);
    }));

    // 还没有 ensure_blocks，所以 logical page 0 未分配
    assert(thrown_runtime_error([&]() {
        manager.physical_token_index(idx, 0, 2);
    }));

    manager.ensure_blocks(idx, 2, 2);

    // 只分配了 logical page 0，logical token 2 属于 page 1
    assert(thrown_runtime_error([&]() {
        manager.physical_token_index(idx, 2, 2);
    }));
}

static void test_invalid_page_size_should_throw() {
    BlockTableManager manager(4);

    const int64_t idx = manager.allocate_table();

    assert(thrown_runtime_error([&]() {
        manager.ensure_blocks(idx, 1, 0);
    }));

    assert(thrown_runtime_error([&]() {
        manager.ensure_blocks(idx, 1, -1);
    }));

    assert(manager.table(idx).empty());
    assert(manager.num_used_blocks() == 0);
    assert(manager.num_free_blocks() == 4);
}

static void test_reset() {
    BlockTableManager manager(4);

    const int64_t idx0 = manager.allocate_table();
    const int64_t idx1 = manager.allocate_table();

    manager.ensure_blocks(idx0, 4, 2);
    manager.ensure_blocks(idx1, 2, 2);

    assert(manager.num_tables() == 2);
    assert(manager.num_used_tables() == 2);
    assert(manager.num_used_blocks() == 3);
    assert(manager.num_free_blocks() == 1);

    manager.reset();

    assert(manager.num_tables() == 0);
    assert(manager.num_used_tables() == 0);
    assert(manager.num_free_tables() == 0);

    assert(manager.num_blocks() == 4);
    assert(manager.num_used_blocks() == 0);
    assert(manager.num_free_blocks() == 4);

    assert(!manager.is_block_used(0));
    assert(!manager.is_block_used(1));
    assert(!manager.is_block_used(2));
    assert(!manager.is_block_used(3));

    const int64_t idx2 = manager.allocate_table();

    assert(idx2 == 0);
    assert(manager.num_tables() == 1);
    assert(manager.num_used_tables() == 1);
}

static void test_invalid_constructor_should_throw() {
    assert(thrown_runtime_error([]() {
        BlockTableManager manager(0);
    }));

    assert(thrown_runtime_error([]() {
        BlockTableManager manager(-1);
    }));
}

int main() {
    test_constructor();
    test_allocate_table();
    test_multiple_tables();
    test_ensure_blocks();
    test_ensure_blocks_incremental();
    test_multiple_tables_get_different_blocks();
    test_physical_token_index_with_allocated_blocks();
    test_physical_token_index_with_custom_valid_mapping();
    test_free_table_releases_blocks();
    test_reuse_table();
    test_reuse_blocks_after_free_table();
    test_not_enough_blocks_should_throw();
    test_double_free_should_throw();
    test_invalid_table_idx_should_throw();
    test_invalid_block_idx_should_throw();
    test_invalid_physical_token_index_should_throw();
    test_invalid_page_size_should_throw();
    test_reset();
    test_invalid_constructor_should_throw();

    std::cout << "test_block_table_manager passed" << std::endl;
    return 0;
}
// tests/test_request_block_table.cpp

#include "engine/block_table_manager.hpp"
#include "engine/request.hpp"

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

static GenerationRequest make_request(
    int64_t request_id,
    const std::vector<int32_t>& input_ids,
    int64_t max_new_tokens
) {
    GenerationRequest req;

    req.request_id = request_id;
    req.input_ids = input_ids;
    req.generated_ids.clear();

    req.prompt_len = static_cast<int64_t>(input_ids.size());
    req.cached_len = 0;
    req.max_new_tokens = max_new_tokens;

    req.eos_token_id = -1;
    req.table_idx = -1;

    req.status = RequestStatus::WaitingPrefill;

    return req;
}

static void test_ensure_device_blocks_allocates_table() {
    BlockTableManager table_manager(4);

    GenerationRequest req = make_request(0, {1, 2, 3}, 8);

    assert(req.table_idx == -1);
    assert(table_manager.num_tables() == 0);
    assert(table_manager.num_used_tables() == 0);
    assert(table_manager.num_used_blocks() == 0);

    // input_ids.size() = 3
    // page_size = 2
    // required blocks = ceil(3 / 2) = 2
    req.ensure_device_blocks(table_manager, 2);

    assert(req.table_idx == 0);

    const auto& table = table_manager.table(req.table_idx);

    assert(table.size() == 2);
    assert(table[0] == 0);
    assert(table[1] == 1);

    assert(table_manager.num_tables() == 1);
    assert(table_manager.num_used_tables() == 1);
    assert(table_manager.num_free_tables() == 0);

    assert(table_manager.num_blocks() == 4);
    assert(table_manager.num_used_blocks() == 2);
    assert(table_manager.num_free_blocks() == 2);

    assert(table_manager.is_block_used(0));
    assert(table_manager.is_block_used(1));
    assert(!table_manager.is_block_used(2));
    assert(!table_manager.is_block_used(3));
}

static void test_ensure_blocks_allocates_table() {
    BlockTableManager table_manager(4);

    GenerationRequest req = make_request(0, {1, 2, 3}, 8);

    assert(req.table_idx == -1);

    // token_len = 5
    // page_size = 2
    // required blocks = 3
    req.ensure_blocks(table_manager, 5, 2);

    assert(req.table_idx == 0);

    const auto& table = table_manager.table(req.table_idx);

    assert(table.size() == 3);
    assert(table[0] == 0);
    assert(table[1] == 1);
    assert(table[2] == 2);

    assert(table_manager.num_used_blocks() == 3);
    assert(table_manager.num_free_blocks() == 1);
}

static void test_ensure_blocks_incremental() {
    BlockTableManager table_manager(5);

    GenerationRequest req = make_request(0, {1}, 8);

    assert(req.table_idx == -1);

    req.ensure_blocks(table_manager, 1, 2);

    assert(req.table_idx == 0);
    assert(table_manager.table(req.table_idx).size() == 1);
    assert(table_manager.table(req.table_idx)[0] == 0);
    assert(table_manager.num_used_blocks() == 1);
    assert(table_manager.num_free_blocks() == 4);

    // token_len = 2 仍然只需要 1 个 block
    req.ensure_blocks(table_manager, 2, 2);

    assert(table_manager.table(req.table_idx).size() == 1);
    assert(table_manager.table(req.table_idx)[0] == 0);
    assert(table_manager.num_used_blocks() == 1);
    assert(table_manager.num_free_blocks() == 4);

    // token_len = 3 需要第 2 个 block
    req.ensure_blocks(table_manager, 3, 2);

    assert(table_manager.table(req.table_idx).size() == 2);
    assert(table_manager.table(req.table_idx)[0] == 0);
    assert(table_manager.table(req.table_idx)[1] == 1);
    assert(table_manager.num_used_blocks() == 2);
    assert(table_manager.num_free_blocks() == 3);

    // token_len = 5 需要第 3 个 block
    req.ensure_blocks(table_manager, 5, 2);

    assert(table_manager.table(req.table_idx).size() == 3);
    assert(table_manager.table(req.table_idx)[0] == 0);
    assert(table_manager.table(req.table_idx)[1] == 1);
    assert(table_manager.table(req.table_idx)[2] == 2);
    assert(table_manager.num_used_blocks() == 3);
    assert(table_manager.num_free_blocks() == 2);
}

static void test_ensure_device_blocks_after_append_token() {
    BlockTableManager table_manager(4);

    GenerationRequest req = make_request(0, {1, 2}, 8);

    // input_ids.size() = 2, page_size = 2
    // 只需要 1 个 block
    req.ensure_device_blocks(table_manager, 2);

    assert(req.table_idx == 0);
    assert(table_manager.table(req.table_idx).size() == 1);
    assert(table_manager.table(req.table_idx)[0] == 0);
    assert(table_manager.num_used_blocks() == 1);

    // 模拟 append_token 后，device_len 从 2 变 3
    // 这时跨 page，需要第 2 个 block
    req.input_ids.push_back(10);
    req.generated_ids.push_back(10);

    req.ensure_device_blocks(table_manager, 2);

    assert(table_manager.table(req.table_idx).size() == 2);
    assert(table_manager.table(req.table_idx)[0] == 0);
    assert(table_manager.table(req.table_idx)[1] == 1);
    assert(table_manager.num_used_blocks() == 2);
    assert(table_manager.num_free_blocks() == 2);
}

static void test_multiple_requests_get_different_tables_and_blocks() {
    BlockTableManager table_manager(6);

    GenerationRequest req0 = make_request(0, {1, 2, 3, 4}, 8);
    GenerationRequest req1 = make_request(1, {5, 6, 7}, 8);

    req0.ensure_device_blocks(table_manager, 2);
    req1.ensure_device_blocks(table_manager, 2);

    assert(req0.table_idx == 0);
    assert(req1.table_idx == 1);

    const auto& table0 = table_manager.table(req0.table_idx);
    const auto& table1 = table_manager.table(req1.table_idx);

    // req0 input_len = 4, page_size = 2 => 2 blocks
    assert(table0.size() == 2);
    assert(table0[0] == 0);
    assert(table0[1] == 1);

    // req1 input_len = 3, page_size = 2 => 2 blocks
    assert(table1.size() == 2);
    assert(table1[0] == 2);
    assert(table1[1] == 3);

    assert(table_manager.num_used_tables() == 2);
    assert(table_manager.num_used_blocks() == 4);
    assert(table_manager.num_free_blocks() == 2);
}

static void test_physical_token_index() {
    BlockTableManager table_manager(10);

    GenerationRequest req = make_request(0, {1, 2, 3}, 8);

    req.table_idx = table_manager.allocate_table();

    auto& table = table_manager.table(req.table_idx);

    table.push_back(5);
    table.push_back(2);
    table.push_back(7);

    // page_size = 4
    //
    // logical token 0 -> page 0 offset 0 -> block 5 token 20
    // logical token 3 -> page 0 offset 3 -> block 5 token 23
    // logical token 4 -> page 1 offset 0 -> block 2 token 8
    // logical token 9 -> page 2 offset 1 -> block 7 token 29

    assert(req.physical_token_index(table_manager, 0, 4) == 20);
    assert(req.physical_token_index(table_manager, 3, 4) == 23);
    assert(req.physical_token_index(table_manager, 4, 4) == 8);
    assert(req.physical_token_index(table_manager, 9, 4) == 29);
}

static void test_release_blocks() {
    BlockTableManager table_manager(4);

    GenerationRequest req = make_request(0, {1, 2, 3}, 8);

    req.ensure_device_blocks(table_manager, 2);

    const int64_t old_table_idx = req.table_idx;

    assert(old_table_idx == 0);
    assert(table_manager.is_table_used(old_table_idx));
    assert(table_manager.num_used_blocks() == 2);
    assert(table_manager.num_free_blocks() == 2);

    req.release_blocks(table_manager);

    assert(req.table_idx == -1);

    assert(!table_manager.is_table_used(old_table_idx));
    assert(table_manager.num_used_tables() == 0);
    assert(table_manager.num_free_tables() == 1);

    assert(table_manager.num_used_blocks() == 0);
    assert(table_manager.num_free_blocks() == 4);

    assert(!table_manager.is_block_used(0));
    assert(!table_manager.is_block_used(1));
    assert(!table_manager.is_block_used(2));
    assert(!table_manager.is_block_used(3));

    assert(thrown_runtime_error([&]() {
        table_manager.table(old_table_idx);
    }));
}

static void test_release_blocks_no_table() {
    BlockTableManager table_manager(4);

    GenerationRequest req = make_request(0, {1, 2, 3}, 8);

    assert(req.table_idx == -1);

    req.release_blocks(table_manager);

    assert(req.table_idx == -1);
    assert(table_manager.num_tables() == 0);
    assert(table_manager.num_used_blocks() == 0);
    assert(table_manager.num_free_blocks() == 4);
}

static void test_reuse_table_and_blocks_after_release() {
    BlockTableManager table_manager(4);

    GenerationRequest req0 = make_request(0, {1, 2, 3, 4}, 8);
    GenerationRequest req1 = make_request(1, {5, 6}, 8);

    req0.ensure_device_blocks(table_manager, 2);

    assert(req0.table_idx == 0);
    assert(table_manager.table(req0.table_idx).size() == 2);
    assert(table_manager.table(req0.table_idx)[0] == 0);
    assert(table_manager.table(req0.table_idx)[1] == 1);

    req0.release_blocks(table_manager);

    assert(req0.table_idx == -1);
    assert(table_manager.num_free_tables() == 1);
    assert(table_manager.num_free_blocks() == 4);

    req1.ensure_device_blocks(table_manager, 2);

    // table_idx 会复用 0
    assert(req1.table_idx == 0);

    const auto& table1 = table_manager.table(req1.table_idx);

    assert(table1.size() == 1);

    // free_table 释放 block 0,1 时是顺序 free，
    // free_blocks 是栈式结构，所以最后释放的 1 会先被复用
    assert(table1[0] == 1);

    assert(table_manager.num_used_tables() == 1);
    assert(table_manager.num_used_blocks() == 1);
    assert(table_manager.num_free_blocks() == 3);
}

static void test_not_enough_blocks_rolls_back_new_table() {
    BlockTableManager table_manager(2);

    GenerationRequest req = make_request(0, {1, 2, 3, 4, 5}, 8);

    assert(req.table_idx == -1);

    // input_len = 5, page_size = 2
    // 需要 3 个 block，但是 manager 只有 2 个 block
    const bool thrown = thrown_runtime_error([&]() {
        req.ensure_device_blocks(table_manager, 2);
    });

    assert(thrown);

    // ensure_device_blocks 失败后，刚申请的 table 应该回滚释放
    assert(req.table_idx == -1);

    assert(table_manager.num_tables() == 1);
    assert(table_manager.num_used_tables() == 0);
    assert(table_manager.num_free_tables() == 1);

    assert(table_manager.num_used_blocks() == 0);
    assert(table_manager.num_free_blocks() == 2);
}

static void test_not_enough_blocks_keeps_existing_table() {
    BlockTableManager table_manager(2);

    GenerationRequest req = make_request(0, {1, 2}, 8);

    req.ensure_device_blocks(table_manager, 2);

    assert(req.table_idx == 0);
    assert(table_manager.table(req.table_idx).size() == 1);
    assert(table_manager.num_used_blocks() == 1);
    assert(table_manager.num_free_blocks() == 1);

    // 尝试扩到 token_len = 5，需要 3 个 block
    // 当前已有 1 个 block，还需要 2 个，但只剩 1 个
    const bool thrown = thrown_runtime_error([&]() {
        req.ensure_blocks(table_manager, 5, 2);
    });

    assert(thrown);

    // 已有 table 不应该被释放
    assert(req.table_idx == 0);
    assert(table_manager.is_table_used(req.table_idx));

    // 原来的 block 仍然保留
    assert(table_manager.table(req.table_idx).size() == 1);
    assert(table_manager.table(req.table_idx)[0] == 0);

    assert(table_manager.num_used_blocks() == 1);
    assert(table_manager.num_free_blocks() == 1);
}

static void test_physical_token_index_without_table_should_throw() {
    BlockTableManager table_manager(4);

    GenerationRequest req = make_request(0, {1, 2, 3}, 8);

    assert(req.table_idx == -1);

    assert(thrown_runtime_error([&]() {
        req.physical_token_index(table_manager, 0, 2);
    }));
}

static void test_physical_token_index_out_of_range_should_throw() {
    BlockTableManager table_manager(4);

    GenerationRequest req = make_request(0, {1, 2}, 8);

    req.ensure_device_blocks(table_manager, 2);

    assert(req.table_idx == 0);
    assert(table_manager.table(req.table_idx).size() == 1);

    assert(thrown_runtime_error([&]() {
        req.physical_token_index(table_manager, -1, 2);
    }));

    // logical token 2 属于 logical page 1，
    // 但是当前只分配了 page 0
    assert(thrown_runtime_error([&]() {
        req.physical_token_index(table_manager, 2, 2);
    }));

    assert(thrown_runtime_error([&]() {
        req.physical_token_index(table_manager, 0, 0);
    }));
}

static void test_invalid_page_size_should_throw() {
    BlockTableManager table_manager(4);

    GenerationRequest req = make_request(0, {1, 2, 3}, 8);

    assert(thrown_runtime_error([&]() {
        req.ensure_device_blocks(table_manager, 0);
    }));

    assert(thrown_runtime_error([&]() {
        req.ensure_device_blocks(table_manager, -1);
    }));

    assert(req.table_idx == -1);
    assert(table_manager.num_used_tables() == 0);
    assert(table_manager.num_used_blocks() == 0);
}

int main() {
    test_ensure_device_blocks_allocates_table();
    test_ensure_blocks_allocates_table();
    test_ensure_blocks_incremental();
    test_ensure_device_blocks_after_append_token();
    test_multiple_requests_get_different_tables_and_blocks();
    test_physical_token_index();
    test_release_blocks();
    test_release_blocks_no_table();
    test_reuse_table_and_blocks_after_release();
    test_not_enough_blocks_rolls_back_new_table();
    test_not_enough_blocks_keeps_existing_table();
    test_physical_token_index_without_table_should_throw();
    test_physical_token_index_out_of_range_should_throw();
    test_invalid_page_size_should_throw();

    std::cout << "test_request_block_table passed" << std::endl;
    return 0;
}
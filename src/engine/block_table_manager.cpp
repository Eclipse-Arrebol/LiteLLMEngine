// src/engine/block_table_manager.cpp

#include "engine/block_table_manager.hpp"

#include <stdexcept>

namespace lite_llm {

namespace {

int64_t ceil_div(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

}  // namespace

BlockTableManager::BlockTableManager(int64_t num_blocks) {
    if (num_blocks <= 0) {
        throw std::runtime_error("num_blocks must be positive");
    }

    num_blocks_ = num_blocks;
    reset_blocks();
}

int64_t BlockTableManager::allocate_table() {
    if (!free_table_indices_.empty()) {
        const int64_t table_idx = free_table_indices_.back();
        free_table_indices_.pop_back();

        table_used_[static_cast<size_t>(table_idx)] = 1;
        tables_[static_cast<size_t>(table_idx)].clear();

        return table_idx;
    }

    const int64_t table_idx = static_cast<int64_t>(tables_.size());

    tables_.emplace_back();
    table_used_.push_back(1);

    return table_idx;
}

void BlockTableManager::free_table(int64_t table_idx) {
    check_table_used(table_idx);

    std::vector<int32_t>& block_table =
        tables_[static_cast<size_t>(table_idx)];

    if (!block_table.empty()) {
        free_blocks(block_table);
        block_table.clear();
    }

    table_used_[static_cast<size_t>(table_idx)] = 0;
    free_table_indices_.push_back(table_idx);
}

void BlockTableManager::ensure_blocks(
    int64_t table_idx,
    int64_t token_len,
    int64_t page_size
) {
    check_table_used(table_idx);

    const int64_t required_blocks =
        num_required_blocks(token_len, page_size);

    std::vector<int32_t>& block_table =
        tables_[static_cast<size_t>(table_idx)];

    const int64_t current_blocks =
        static_cast<int64_t>(block_table.size());

    if (required_blocks <= current_blocks) {
        return;
    }

    const int64_t need_blocks = required_blocks - current_blocks;

    if (!has_free_blocks(need_blocks)) {
        throw std::runtime_error("not enough free physical blocks");
    }

    for (int64_t i = 0; i < need_blocks; ++i) {
        block_table.push_back(allocate_block());
    }
}

int64_t BlockTableManager::physical_token_index(
    int64_t table_idx,
    int64_t logical_token_index,
    int64_t page_size
) const {
    check_table_used(table_idx);

    if (logical_token_index < 0) {
        throw std::runtime_error("logical_token_index must be non-negative");
    }

    if (page_size <= 0) {
        throw std::runtime_error("page_size must be positive");
    }

    const std::vector<int32_t>& block_table =
        tables_[static_cast<size_t>(table_idx)];

    const int64_t logical_page = logical_token_index / page_size;
    const int64_t page_offset = logical_token_index % page_size;

    if (logical_page >= static_cast<int64_t>(block_table.size())) {
        throw std::runtime_error("logical page not allocated");
    }

    const int32_t physical_block =
        block_table[static_cast<size_t>(logical_page)];

    check_block_id(physical_block);

    return static_cast<int64_t>(physical_block) * page_size + page_offset;
}

std::vector<int32_t>& BlockTableManager::table(int64_t table_idx) {
    check_table_used(table_idx);
    return tables_[static_cast<size_t>(table_idx)];
}

const std::vector<int32_t>& BlockTableManager::table(
    int64_t table_idx
) const {
    check_table_used(table_idx);
    return tables_[static_cast<size_t>(table_idx)];
}

void BlockTableManager::reset() {
    tables_.clear();
    table_used_.clear();
    free_table_indices_.clear();

    reset_blocks();
}

bool BlockTableManager::is_table_used(int64_t table_idx) const {
    check_table_idx(table_idx);
    return table_used_[static_cast<size_t>(table_idx)] != 0;
}

bool BlockTableManager::is_block_used(int32_t block_id) const {
    check_block_id(block_id);
    return block_used_[static_cast<size_t>(block_id)] != 0;
}

int64_t BlockTableManager::num_tables() const {
    return static_cast<int64_t>(tables_.size());
}

int64_t BlockTableManager::num_used_tables() const {
    int64_t count = 0;

    for (uint8_t used : table_used_) {
        if (used) {
            ++count;
        }
    }

    return count;
}

int64_t BlockTableManager::num_free_tables() const {
    return static_cast<int64_t>(free_table_indices_.size());
}

int64_t BlockTableManager::num_blocks() const {
    return num_blocks_;
}

int64_t BlockTableManager::num_free_blocks() const {
    return static_cast<int64_t>(free_blocks_.size());
}

int64_t BlockTableManager::num_used_blocks() const {
    return num_blocks_ - num_free_blocks();
}

bool BlockTableManager::has_free_blocks(int64_t num_blocks) const {
    if (num_blocks < 0) {
        return false;
    }

    return static_cast<int64_t>(free_blocks_.size()) >= num_blocks;
}

int64_t BlockTableManager::num_required_blocks(
    int64_t token_len,
    int64_t page_size
) {
    if (token_len < 0) {
        throw std::runtime_error("token_len must be non-negative");
    }

    if (page_size <= 0) {
        throw std::runtime_error("page_size must be positive");
    }

    if (token_len == 0) {
        return 0;
    }

    return ceil_div(token_len, page_size);
}

int32_t BlockTableManager::allocate_block() {
    if (free_blocks_.empty()) {
        throw std::runtime_error("no free physical block available");
    }

    const int32_t block_id = free_blocks_.back();
    free_blocks_.pop_back();

    block_used_[static_cast<size_t>(block_id)] = 1;

    return block_id;
}

void BlockTableManager::free_block(int32_t block_id) {
    check_block_id(block_id);

    if (!block_used_[static_cast<size_t>(block_id)]) {
        throw std::runtime_error("physical block is already free");
    }

    block_used_[static_cast<size_t>(block_id)] = 0;
    free_blocks_.push_back(block_id);
}

void BlockTableManager::free_blocks(
    const std::vector<int32_t>& block_ids
) {
    for (int32_t block_id : block_ids) {
        free_block(block_id);
    }
}

void BlockTableManager::reset_blocks() {
    free_blocks_.clear();
    block_used_.assign(static_cast<size_t>(num_blocks_), 0);

    // 反着 push，这样 allocate_block() pop_back() 时先拿到 0
    for (int64_t i = num_blocks_ - 1; i >= 0; --i) {
        free_blocks_.push_back(static_cast<int32_t>(i));
    }
}

void BlockTableManager::check_table_idx(int64_t table_idx) const {
    if (table_idx < 0 ||
        table_idx >= static_cast<int64_t>(tables_.size())) {
        throw std::runtime_error("block table index out of range");
    }
}

void BlockTableManager::check_table_used(int64_t table_idx) const {
    check_table_idx(table_idx);

    if (!table_used_[static_cast<size_t>(table_idx)]) {
        throw std::runtime_error("block table is not used");
    }
}

void BlockTableManager::check_block_id(int32_t block_id) const {
    if (block_id < 0 || block_id >= num_blocks_) {
        throw std::runtime_error("physical block id out of range");
    }
}

}  // namespace lite_llm
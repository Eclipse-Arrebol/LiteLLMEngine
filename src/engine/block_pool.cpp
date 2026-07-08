// src/engine/block_pool.cpp
#include "engine/block_pool.hpp"

#include <stdexcept>

namespace lite_llm {

BlockPool::BlockPool(int64_t num_blocks) {
    if (num_blocks <= 0) {
        throw std::runtime_error("num_blocks must be positive");
    }

    num_blocks_ = num_blocks;
    used_.assign(static_cast<size_t>(num_blocks_), 0);

    // 反着 push，这样 allocate_block() pop_back() 时先拿到 0
    for (int64_t i = num_blocks_ - 1; i >= 0; --i) {
        free_blocks_.push_back(static_cast<int32_t>(i));
    }
}

int32_t BlockPool::allocate_block() {
    if (free_blocks_.empty()) {
        throw std::runtime_error("no free block available");
    }

    const int32_t block_id = free_blocks_.back();
    free_blocks_.pop_back();

    used_[static_cast<size_t>(block_id)] = 1;
    return block_id;
}

std::vector<int32_t> BlockPool::allocate_blocks(int64_t num_blocks) {
    if (num_blocks < 0) {
        throw std::runtime_error("num_blocks must be non-negative");
    }

    if (!has_free_blocks(num_blocks)) {
        throw std::runtime_error("not enough free blocks");
    }

    std::vector<int32_t> result;
    result.reserve(static_cast<size_t>(num_blocks));

    for (int64_t i = 0; i < num_blocks; ++i) {
        result.push_back(allocate_block());
    }

    return result;
}

void BlockPool::free_block(int32_t block_id) {
    check_block_id(block_id);

    if (!used_[static_cast<size_t>(block_id)]) {
        throw std::runtime_error("block is already free");
    }

    used_[static_cast<size_t>(block_id)] = 0;
    free_blocks_.push_back(block_id);
}

void BlockPool::free_blocks(const std::vector<int32_t>& block_ids) {
    for (int32_t block_id : block_ids) {
        free_block(block_id);
    }
}

void BlockPool::reset() {
    free_blocks_.clear();

    for (int64_t i = 0; i < num_blocks_; ++i) {
        used_[static_cast<size_t>(i)] = 0;
    }

    for (int64_t i = num_blocks_ - 1; i >= 0; --i) {
        free_blocks_.push_back(static_cast<int32_t>(i));
    }
}

bool BlockPool::has_free_blocks(int64_t num_blocks) const {
    if (num_blocks < 0) {
        return false;
    }

    return static_cast<int64_t>(free_blocks_.size()) >= num_blocks;
}

int64_t BlockPool::num_blocks() const {
    return num_blocks_;
}

int64_t BlockPool::num_free_blocks() const {
    return static_cast<int64_t>(free_blocks_.size());
}

int64_t BlockPool::num_used_blocks() const {
    return num_blocks_ - num_free_blocks();
}

bool BlockPool::is_used(int32_t block_id) const {
    check_block_id(block_id);
    return used_[static_cast<size_t>(block_id)] != 0;
}

void BlockPool::check_block_id(int32_t block_id) const {
    if (block_id < 0 || block_id >= num_blocks_) {
        throw std::runtime_error("block_id out of range");
    }
}

}  // namespace lite_llm
// include/engine/block_pool.hpp
#pragma once

#include <cstdint>
#include <vector>

namespace lite_llm {

class BlockPool {
public:
    explicit BlockPool(int64_t num_blocks);

    int32_t allocate_block();
    std::vector<int32_t> allocate_blocks(int64_t num_blocks);

    void free_block(int32_t block_id);
    void free_blocks(const std::vector<int32_t>& block_ids);

    void reset();

    bool has_free_blocks(int64_t num_blocks) const;

    int64_t num_blocks() const;
    int64_t num_free_blocks() const;
    int64_t num_used_blocks() const;

    bool is_used(int32_t block_id) const;

private:
    void check_block_id(int32_t block_id) const;

    int64_t num_blocks_ = 0;
    std::vector<int32_t> free_blocks_;
    std::vector<uint8_t> used_;
};

}  // namespace lite_llm
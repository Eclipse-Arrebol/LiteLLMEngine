// include/engine/block_table_manager.hpp
#pragma once

#include <cstdint>
#include <vector>

namespace lite_llm {

class BlockTableManager {
public:
    explicit BlockTableManager(int64_t num_blocks);

    int64_t allocate_table();
    void free_table(int64_t table_idx);

    void ensure_blocks(
        int64_t table_idx,
        int64_t token_len,
        int64_t page_size
    );

    int64_t physical_token_index(
        int64_t table_idx,
        int64_t logical_token_index,
        int64_t page_size
    ) const;

    std::vector<int32_t>& table(int64_t table_idx);
    const std::vector<int32_t>& table(int64_t table_idx) const;

    void reset();

    bool is_table_used(int64_t table_idx) const;
    bool is_block_used(int32_t block_id) const;

    int64_t num_tables() const;
    int64_t num_used_tables() const;
    int64_t num_free_tables() const;

    int64_t num_blocks() const;
    int64_t num_free_blocks() const;
    int64_t num_used_blocks() const;

    bool has_free_blocks(int64_t num_blocks) const;

private:
    static int64_t num_required_blocks(
        int64_t token_len,
        int64_t page_size
    );

    int32_t allocate_block();
    void free_block(int32_t block_id);
    void free_blocks(const std::vector<int32_t>& block_ids);

    void reset_blocks();

    void check_table_idx(int64_t table_idx) const;
    void check_table_used(int64_t table_idx) const;
    void check_block_id(int32_t block_id) const;

private:
    int64_t num_blocks_ = 0;

    // physical block pool
    std::vector<int32_t> free_blocks_;
    std::vector<uint8_t> block_used_;

    // table_idx -> block_table
    std::vector<std::vector<int32_t>> tables_;
    std::vector<uint8_t> table_used_;
    std::vector<int64_t> free_table_indices_;
};

}  // namespace lite_llm
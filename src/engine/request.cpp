#include "engine/request.hpp"

#include "engine/block_table_manager.hpp"

#include <stdexcept>

namespace lite_llm {

void GenerationRequest::ensure_blocks(
    BlockTableManager& table_manager,
    int64_t token_len,
    int64_t page_size
) {
    bool allocated_table = false;

    if (table_idx < 0) {
        table_idx = table_manager.allocate_table();
        allocated_table = true;
    }

    try {
        table_manager.ensure_blocks(
            table_idx,
            token_len,
            page_size
        );
    } catch (...) {
        if (allocated_table) {
            table_manager.free_table(table_idx);
            table_idx = -1;
        }

        throw;
    }
}

void GenerationRequest::ensure_device_blocks(
    BlockTableManager& table_manager,
    int64_t page_size
) {
    ensure_blocks(
        table_manager,
        device_len(),
        page_size
    );
}

void GenerationRequest::release_blocks(
    BlockTableManager& table_manager
) {
    if (table_idx < 0) {
        return;
    }

    table_manager.free_table(table_idx);
    table_idx = -1;
}

int64_t GenerationRequest::physical_token_index(
    const BlockTableManager& table_manager,
    int64_t logical_token_index,
    int64_t page_size
) const {
    if (table_idx < 0) {
        throw std::runtime_error("request has no block table");
    }

    return table_manager.physical_token_index(
        table_idx,
        logical_token_index,
        page_size
    );
}

}  // namespace lite_llm
// include/engine/paged_generate_engine.hpp
#pragma once

#include "engine/paged_kv_cache_manager.hpp"
#include "engine/request_manager.hpp"
#include "model/qwen3.hpp"
#include "runtime/generation.hpp"

#include <cstdint>
#include <vector>

namespace lite_llm {

class PagedGenerateEngine {
public:
    PagedGenerateEngine(
        const Qwen3ForCausalLM& model,
        const GreedyGenerateOptions& options,
        int64_t max_total_tokens,
        int64_t page_size
    );

    int64_t add_request(
        const std::vector<int32_t>& prompt_ids,
        int64_t max_new_tokens,
        int32_t eos_token_id
    );

    void append_input_tokens(
        int64_t request_id,
        const std::vector<int32_t>& token_ids,
        int64_t max_new_tokens,
        int32_t eos_token_id
    );

    int32_t prefill(int64_t request_id);
    int32_t decode_one_step(int64_t request_id);

    // Returns sampled tokens for unfinished requests kept from active_request_ids.
    std::vector<int32_t> decode_batch(
        const std::vector<int64_t>& active_request_ids
    );

    std::vector<int32_t> generate_until_finished(int64_t request_id);

    const GenerationRequest& request(int64_t request_id) const;
    GenerationRequest& request(int64_t request_id);

    bool finished(int64_t request_id) const;

    void release_request(int64_t request_id);

    RequestManager& request_manager();
    const RequestManager& request_manager() const;

    PagedKVCacheManager& paged_kv_manager();
    const PagedKVCacheManager& paged_kv_manager() const;

private:
    int32_t forward_uncached_chunk_and_sample(
        int64_t request_id,
        int64_t seq_len,
        const char* stage_name
    );

    std::vector<int32_t> make_uncached_input_ids(
        const GenerationRequest& request,
        int64_t seq_len
    ) const;

    std::vector<int32_t> make_uncached_position_ids(
        const GenerationRequest& request,
        int64_t seq_len
    ) const;

    void append_sampled_token(
        int64_t request_id,
        int32_t token_id
    );

private:
    const Qwen3ForCausalLM& model_;
    GreedyGenerateOptions options_;

    RequestManager request_manager_;
    PagedKVCacheManager paged_kv_manager_;
};

}  // namespace lite_llm

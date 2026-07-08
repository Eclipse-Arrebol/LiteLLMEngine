// include/engine/request.hpp
#pragma once

#include <cassert>
#include <cstdint>
#include <vector>

namespace lite_llm {

enum class RequestStatus {
    WaitingPrefill,
    Decoding,
    Finished
};

struct GenerationRequest {
    int64_t request_id = -1;

    // prompt + already generated tokens
    std::vector<int32_t> input_ids;

    // only generated tokens, used as final output
    std::vector<int32_t> generated_ids;

    int64_t prompt_len = 0;
    int64_t cached_len = 0;
    int64_t max_new_tokens = 0;

    int32_t eos_token_id = -1;

    // reserved for future PagedKVCache page table
    int64_t table_idx = -1;

    RequestStatus status = RequestStatus::WaitingPrefill;

    int64_t device_len() const {
        return static_cast<int64_t>(input_ids.size());
    }

    int64_t max_device_len() const {
        return prompt_len + max_new_tokens;
    }

    int64_t output_len() const {
        return max_new_tokens;
    }

    int64_t num_new_tokens() const {
        return static_cast<int64_t>(generated_ids.size());
    }

    int64_t remain_len() const {
        return max_device_len() - device_len();
    }

    int64_t extend_len() const {
        return device_len() - cached_len;
    }

    bool can_decode() const {
        return status == RequestStatus::Decoding && remain_len() > 0;
    }

    bool finished() const {
        return status == RequestStatus::Finished;
    }

    void check_valid() const {
        assert(request_id >= 0);
        assert(prompt_len >= 0);
        assert(cached_len >= 0);
        assert(max_new_tokens >= 0);
        assert(cached_len <= device_len());
        assert(device_len() <= max_device_len());
    }
};

}  // namespace lite_llm
// include/engine/request_manager.hpp
#pragma once

#include "engine/request.hpp"

#include <cstdint>
#include <unordered_map>
#include <vector>


/**
 * @brief 
add_request(prompt)
  input_ids = prompt
  cached_len = 0
  status = WaitingPrefill

prefill forward
  model 输入 input_ids[cached_len : device_len]
  也就是整个 prompt
  forward 完成后 mark_prefill_done()
  cached_len = prompt_len

采样 first_token
  append_token(first_token)
  input_ids = prompt + first_token
  generated_ids = first_token
  cached_len 仍然是 prompt_len
  extend_len = 1

decode forward
  model 输入 input_ids[cached_len : device_len]
  也就是刚生成的那个 token
  forward 完成后 cached_len = device_len

采样 next_token
  append_token(next_token)
 * 
 */


namespace lite_llm {

class RequestManager {
public:
    int64_t add_request(
        const std::vector<int32_t>& prompt_ids,
        int64_t max_new_tokens,
        int32_t eos_token_id
    );

    GenerationRequest& request(int64_t request_id);
    const GenerationRequest& request(int64_t request_id) const;

    std::vector<int64_t> waiting_prefill_requests() const;
    std::vector<int64_t> decoding_requests() const;
    std::vector<int64_t> finished_requests() const;

    void mark_prefill_done(int64_t request_id);
    void append_token(int64_t request_id, int32_t token_id);
    void finish_request(int64_t request_id);

    int64_t num_requests() const;

    void mark_forward_done(int64_t request_id);
    void mark_forward_done(
        int64_t request_id,
        int64_t num_tokens
    );


    void append_input_tokens(
        int64_t request_id,
        const std::vector<int32_t>& token_ids
    );

    void reset_generation_options(
      int64_t request_id,
      int64_t max_new_tokens,
      int32_t eos_token_id
  );

private:
    int64_t next_request_id_ = 0;
    std::unordered_map<int64_t, GenerationRequest> requests_;
};

}  // namespace lite_llm
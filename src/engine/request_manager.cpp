// src/engine/request_manager.cpp
#include "engine/request_manager.hpp"

#include <stdexcept>
#include <utility>

namespace lite_llm {

int64_t RequestManager::add_request(
    const std::vector<int32_t>& prompt_ids,
    int64_t max_new_tokens,
    int32_t eos_token_id
) {
    if (max_new_tokens < 0) {
        throw std::runtime_error("max_new_tokens must be non-negative");
    }

    const int64_t request_id = next_request_id_++;

    GenerationRequest req;
    req.request_id = request_id;
    req.input_ids = prompt_ids;
    req.generated_ids.clear();
    req.prompt_len = static_cast<int64_t>(prompt_ids.size());
    req.cached_len = 0;
    req.max_new_tokens = max_new_tokens;
    req.eos_token_id = eos_token_id;
    req.table_idx = -1;

    if (max_new_tokens == 0) {
        req.status = RequestStatus::Finished;
    } else {
        req.status = RequestStatus::WaitingPrefill;
    }

    req.check_valid();

    requests_.emplace(request_id, std::move(req));
    return request_id;
}

GenerationRequest& RequestManager::request(int64_t request_id) {
    auto it = requests_.find(request_id);
    if (it == requests_.end()) {
        throw std::runtime_error("request_id not found");
    }
    return it->second;
}

const GenerationRequest& RequestManager::request(int64_t request_id) const {
    auto it = requests_.find(request_id);
    if (it == requests_.end()) {
        throw std::runtime_error("request_id not found");
    }
    return it->second;
}

std::vector<int64_t> RequestManager::waiting_prefill_requests() const {
    std::vector<int64_t> ids;

    for (const auto& [id, req] : requests_) {
        if (req.status == RequestStatus::WaitingPrefill) {
            ids.push_back(id);
        }
    }

    return ids;
}

std::vector<int64_t> RequestManager::decoding_requests() const {
    std::vector<int64_t> ids;

    for (const auto& [id, req] : requests_) {
        if (req.status == RequestStatus::Decoding) {
            ids.push_back(id);
        }
    }

    return ids;
}

std::vector<int64_t> RequestManager::finished_requests() const {
    std::vector<int64_t> ids;

    for (const auto& [id, req] : requests_) {
        if (req.status == RequestStatus::Finished) {
            ids.push_back(id);
        }
    }

    return ids;
}

void RequestManager::mark_prefill_done(int64_t request_id) {
    GenerationRequest& req = request(request_id);

    if (req.status == RequestStatus::Finished) {
        return;
    }

    if (req.status != RequestStatus::WaitingPrefill) {
        throw std::runtime_error("mark_prefill_done expects WaitingPrefill request");
    }

    req.cached_len = req.device_len();

    if (req.remain_len() > 0) {
        req.status = RequestStatus::Decoding;
    } else {
        req.status = RequestStatus::Finished;
    }

    req.check_valid();
}

/**
 * @brief 向指定对话添加token
 * 
 * @param request_id 
 * @param token_id 
 */
void RequestManager::append_token(
    int64_t request_id,
    int32_t token_id
) {
    GenerationRequest& req = request(request_id);

    if (req.status == RequestStatus::Finished) {
        throw std::runtime_error(
            "RequestManager append_token called on finished request"
        );
    }

    if (req.status != RequestStatus::Decoding) {
        throw std::runtime_error(
            "RequestManager append_token expects Decoding request"
        );
    }

    req.input_ids.push_back(token_id);
    req.generated_ids.push_back(token_id);

    if (req.eos_token_id >= 0 &&
        token_id == req.eos_token_id) {
        req.status = RequestStatus::Finished;
        req.check_valid();
        return;
    }

    if (req.max_new_tokens >= 0 &&
        static_cast<int64_t>(req.generated_ids.size()) >= req.max_new_tokens) {
        req.status = RequestStatus::Finished;
        req.check_valid();
        return;
    }

    req.status = RequestStatus::Decoding;
    req.check_valid();
}

void RequestManager::finish_request(int64_t request_id) {
    GenerationRequest& req = request(request_id);
    req.status = RequestStatus::Finished;
    req.check_valid();
}

int64_t RequestManager::num_requests() const {
    return static_cast<int64_t>(requests_.size());
}


void RequestManager::mark_forward_done(int64_t request_id) {
    GenerationRequest& req = request(request_id);

    if (req.status == RequestStatus::Finished) {
        return;
    }

    req.cached_len = req.device_len();

    if (req.remain_len() <= 0) {
        req.status = RequestStatus::Finished;
    } else {
        req.status = RequestStatus::Decoding;
    }

    req.check_valid();
}

void RequestManager::mark_forward_done(
    int64_t request_id,
    int64_t num_tokens
) {
    GenerationRequest& req = request(request_id);

    if (num_tokens < 0) {
        throw std::runtime_error(
            "RequestManager mark_forward_done num_tokens must be non-negative"
        );
    }

    if (req.cached_len + num_tokens > req.device_len()) {
        throw std::runtime_error(
            "RequestManager mark_forward_done exceeds device_len"
        );
    }

    req.cached_len += num_tokens;

    if (req.status != RequestStatus::Finished) {
        req.status = RequestStatus::Decoding;
    }
}


/**
 * @brief 添加一系列的tokens
 * 
 * @param request_id 
 * @param token_ids 
 */
void RequestManager::append_input_tokens(
    int64_t request_id,
    const std::vector<int32_t>& token_ids
) {
    if (token_ids.empty()) {
        return;
    }

    GenerationRequest& req = request(request_id);

    req.input_ids.insert(
        req.input_ids.end(),
        token_ids.begin(),
        token_ids.end()
    );

    // 新一轮对话开始，上一轮 assistant 生成结果不应该继续计数。
    req.generated_ids.clear();
    req.prompt_len = req.device_len();

    if (req.cached_len == 0) {
        req.status = RequestStatus::WaitingPrefill;
    } else {
        req.status = RequestStatus::Decoding;
    }
}

/**
 * @brief 重新设置最大token和eos
 * 
 * @param request_id 
 * @param max_new_tokens 
 * @param eos_token_id 
 */
void RequestManager::reset_generation_options(
    int64_t request_id,
    int64_t max_new_tokens,
    int32_t eos_token_id
) {
    if (max_new_tokens < 0) {
        throw std::runtime_error(
            "RequestManager reset_generation_options max_new_tokens must be non-negative"
        );
    }

    GenerationRequest& req = request(request_id);

    req.max_new_tokens = max_new_tokens;
    req.eos_token_id = eos_token_id;
    req.generated_ids.clear();
    req.prompt_len = req.device_len();

    if (req.cached_len == 0) {
        req.status = RequestStatus::WaitingPrefill;
    } else {
        req.status = RequestStatus::Decoding;
    }
}


}  // namespace lite_llm

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

void RequestManager::append_token(
    int64_t request_id,
    int32_t token_id
) {
    GenerationRequest& req = request(request_id);

    if (req.status == RequestStatus::Finished) {
        return;
    }

    if (req.status != RequestStatus::Decoding) {
        throw std::runtime_error("append_token expects Decoding request");
    }

    if (req.eos_token_id >= 0 && token_id == req.eos_token_id) {
        req.status = RequestStatus::Finished;
        req.check_valid();
        return;
    }

    if (req.remain_len() <= 0) {
        req.status = RequestStatus::Finished;
        req.check_valid();
        return;
    }

    req.input_ids.push_back(token_id);
    req.generated_ids.push_back(token_id);

    if (req.remain_len() <= 0) {
        req.status = RequestStatus::Finished;
    } else {
        req.status = RequestStatus::Decoding;
    }

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

}  // namespace lite_llm
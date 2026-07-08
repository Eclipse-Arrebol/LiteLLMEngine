// tests/test_request_manager.cpp

#include "engine/request_manager.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace lite_llm;

static bool contains_id(const std::vector<int64_t>& ids, int64_t target) {
    for (int64_t id : ids) {
        if (id == target) {
            return true;
        }
    }
    return false;
}

static void test_add_request() {
    RequestManager manager;

    const int64_t id = manager.add_request({1, 2, 3}, 4, 99);

    assert(id == 0);
    assert(manager.num_requests() == 1);

    const GenerationRequest& req = manager.request(id);

    assert(req.request_id == id);
    assert(req.input_ids.size() == 3);
    assert(req.input_ids[0] == 1);
    assert(req.input_ids[1] == 2);
    assert(req.input_ids[2] == 3);

    assert(req.generated_ids.empty());

    assert(req.prompt_len == 3);
    assert(req.cached_len == 0);
    assert(req.max_new_tokens == 4);
    assert(req.eos_token_id == 99);
    assert(req.table_idx == -1);

    assert(req.device_len() == 3);
    assert(req.max_device_len() == 7);
    assert(req.output_len() == 4);
    assert(req.num_new_tokens() == 0);
    assert(req.remain_len() == 4);
    assert(req.extend_len() == 3);

    assert(req.status == RequestStatus::WaitingPrefill);
    assert(!req.finished());
    assert(!req.can_decode());
}

static void test_status_lists_after_add() {
    RequestManager manager;

    const int64_t id0 = manager.add_request({1, 2}, 3, 99);
    const int64_t id1 = manager.add_request({3, 4}, 3, 99);

    auto waiting = manager.waiting_prefill_requests();
    auto decoding = manager.decoding_requests();
    auto finished = manager.finished_requests();

    assert(contains_id(waiting, id0));
    assert(contains_id(waiting, id1));
    assert(decoding.empty());
    assert(finished.empty());
}

static void test_mark_forward_done_after_prefill() {
    RequestManager manager;

    const int64_t id = manager.add_request({10, 11, 12}, 4, 99);

    manager.mark_forward_done(id);

    const GenerationRequest& req = manager.request(id);

    assert(req.cached_len == 3);
    assert(req.device_len() == 3);
    assert(req.extend_len() == 0);
    assert(req.remain_len() == 4);

    assert(req.status == RequestStatus::Decoding);
    assert(req.can_decode());
    assert(!req.finished());
}

static void test_append_token() {
    RequestManager manager;

    const int64_t id = manager.add_request({1, 2}, 3, 99);

    manager.mark_forward_done(id);
    manager.append_token(id, 10);

    const GenerationRequest& req = manager.request(id);

    assert(req.status == RequestStatus::Decoding);

    assert(req.input_ids.size() == 3);
    assert(req.input_ids[0] == 1);
    assert(req.input_ids[1] == 2);
    assert(req.input_ids[2] == 10);

    assert(req.generated_ids.size() == 1);
    assert(req.generated_ids[0] == 10);

    assert(req.prompt_len == 2);
    assert(req.cached_len == 2);
    assert(req.device_len() == 3);
    assert(req.max_device_len() == 5);
    assert(req.remain_len() == 2);
    assert(req.extend_len() == 1);
    assert(req.num_new_tokens() == 1);

    assert(req.can_decode());
}

static void test_decode_forward_done() {
    RequestManager manager;

    const int64_t id = manager.add_request({1, 2}, 3, 99);

    manager.mark_forward_done(id);
    manager.append_token(id, 10);

    {
        const GenerationRequest& req = manager.request(id);
        assert(req.cached_len == 2);
        assert(req.device_len() == 3);
        assert(req.extend_len() == 1);
    }

    manager.mark_forward_done(id);

    {
        const GenerationRequest& req = manager.request(id);
        assert(req.cached_len == 3);
        assert(req.device_len() == 3);
        assert(req.extend_len() == 0);
        assert(req.remain_len() == 2);
        assert(req.status == RequestStatus::Decoding);
    }
}

static void test_multiple_decode_steps() {
    RequestManager manager;

    const int64_t id = manager.add_request({1, 2}, 3, 99);

    manager.mark_forward_done(id);

    manager.append_token(id, 10);
    assert(manager.request(id).extend_len() == 1);

    manager.mark_forward_done(id);
    assert(manager.request(id).extend_len() == 0);

    manager.append_token(id, 11);
    assert(manager.request(id).extend_len() == 1);

    manager.mark_forward_done(id);
    assert(manager.request(id).extend_len() == 0);

    const GenerationRequest& req = manager.request(id);

    assert(req.input_ids.size() == 4);
    assert(req.generated_ids.size() == 2);

    assert(req.input_ids[0] == 1);
    assert(req.input_ids[1] == 2);
    assert(req.input_ids[2] == 10);
    assert(req.input_ids[3] == 11);

    assert(req.generated_ids[0] == 10);
    assert(req.generated_ids[1] == 11);

    assert(req.cached_len == 4);
    assert(req.device_len() == 4);
    assert(req.max_device_len() == 5);
    assert(req.remain_len() == 1);
    assert(req.status == RequestStatus::Decoding);
}

static void test_eos_token_not_appended() {
    RequestManager manager;

    const int64_t id = manager.add_request({1, 2}, 5, 99);

    manager.mark_forward_done(id);
    manager.append_token(id, 10);
    manager.mark_forward_done(id);

    manager.append_token(id, 99);

    const GenerationRequest& req = manager.request(id);

    assert(req.status == RequestStatus::Finished);
    assert(req.finished());
    assert(!req.can_decode());

    assert(req.input_ids.size() == 3);
    assert(req.generated_ids.size() == 1);

    assert(req.input_ids[0] == 1);
    assert(req.input_ids[1] == 2);
    assert(req.input_ids[2] == 10);

    assert(req.generated_ids[0] == 10);
}

static void test_max_new_tokens_finish() {
    RequestManager manager;

    const int64_t id = manager.add_request({1, 2}, 2, 99);

    manager.mark_forward_done(id);

    manager.append_token(id, 10);
    assert(manager.request(id).status == RequestStatus::Decoding);

    manager.mark_forward_done(id);

    manager.append_token(id, 11);

    const GenerationRequest& req = manager.request(id);

    assert(req.status == RequestStatus::Finished);
    assert(req.finished());
    assert(!req.can_decode());

    assert(req.generated_ids.size() == 2);
    assert(req.generated_ids[0] == 10);
    assert(req.generated_ids[1] == 11);

    assert(req.input_ids.size() == 4);
    assert(req.input_ids[0] == 1);
    assert(req.input_ids[1] == 2);
    assert(req.input_ids[2] == 10);
    assert(req.input_ids[3] == 11);

    assert(req.device_len() == 4);
    assert(req.max_device_len() == 4);
    assert(req.remain_len() == 0);
}

static void test_zero_max_new_tokens() {
    RequestManager manager;

    const int64_t id = manager.add_request({1, 2, 3}, 0, 99);

    const GenerationRequest& req = manager.request(id);

    assert(req.status == RequestStatus::Finished);
    assert(req.finished());
    assert(!req.can_decode());

    assert(req.input_ids.size() == 3);
    assert(req.generated_ids.empty());

    assert(req.prompt_len == 3);
    assert(req.cached_len == 0);
    assert(req.max_new_tokens == 0);
    assert(req.device_len() == 3);
    assert(req.max_device_len() == 3);
    assert(req.remain_len() == 0);
}

static void test_status_lists() {
    RequestManager manager;

    const int64_t waiting_id = manager.add_request({1}, 3, 99);
    const int64_t decoding_id = manager.add_request({2}, 3, 99);
    const int64_t finished_id = manager.add_request({3}, 3, 99);

    manager.mark_forward_done(decoding_id);
    manager.finish_request(finished_id);

    auto waiting = manager.waiting_prefill_requests();
    auto decoding = manager.decoding_requests();
    auto finished = manager.finished_requests();

    assert(contains_id(waiting, waiting_id));
    assert(!contains_id(waiting, decoding_id));
    assert(!contains_id(waiting, finished_id));

    assert(!contains_id(decoding, waiting_id));
    assert(contains_id(decoding, decoding_id));
    assert(!contains_id(decoding, finished_id));

    assert(!contains_id(finished, waiting_id));
    assert(!contains_id(finished, decoding_id));
    assert(contains_id(finished, finished_id));
}

static void test_invalid_request_id() {
    RequestManager manager;

    bool thrown = false;

    try {
        manager.request(123);
    } catch (const std::runtime_error&) {
        thrown = true;
    }

    assert(thrown);
}

static void test_append_before_prefill_done_should_throw() {
    RequestManager manager;

    const int64_t id = manager.add_request({1, 2}, 3, 99);

    bool thrown = false;

    try {
        manager.append_token(id, 10);
    } catch (const std::runtime_error&) {
        thrown = true;
    }

    assert(thrown);
}

static void test_mark_forward_done_finished_noop() {
    RequestManager manager;

    const int64_t id = manager.add_request({1, 2}, 1, 99);

    manager.mark_forward_done(id);
    manager.append_token(id, 10);

    assert(manager.request(id).status == RequestStatus::Finished);

    manager.mark_forward_done(id);

    const GenerationRequest& req = manager.request(id);

    assert(req.status == RequestStatus::Finished);
    assert(req.generated_ids.size() == 1);
    assert(req.generated_ids[0] == 10);
}

static void test_finish_request() {
    RequestManager manager;

    const int64_t id = manager.add_request({1, 2}, 3, 99);

    manager.finish_request(id);

    const GenerationRequest& req = manager.request(id);

    assert(req.status == RequestStatus::Finished);
    assert(req.finished());
    assert(!req.can_decode());
}

int main() {
    test_add_request();
    test_status_lists_after_add();
    test_mark_forward_done_after_prefill();
    test_append_token();
    test_decode_forward_done();
    test_multiple_decode_steps();
    test_eos_token_not_appended();
    test_max_new_tokens_finish();
    test_zero_max_new_tokens();
    test_status_lists();
    test_invalid_request_id();
    test_append_before_prefill_done_should_throw();
    test_mark_forward_done_finished_noop();
    test_finish_request();

    std::cout << "test_request_manager passed" << std::endl;
    return 0;
}
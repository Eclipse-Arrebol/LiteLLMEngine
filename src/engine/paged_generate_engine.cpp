// src/engine/paged_generate_engine.cpp

#include "engine/paged_generate_engine.hpp"

#include "core/tensor.hpp"
#include "core/tensor_memory_tracker.hpp"

#include "ops/argmax.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace lite_llm {


PagedGenerateEngine::PagedGenerateEngine(
    const Qwen3ForCausalLM& model,
    const GreedyGenerateOptions& options,
    int64_t max_total_tokens,
    int64_t page_size
)
    : model_(model),
      options_(options),
      paged_kv_manager_(
          model.config(),
          options.device,
          DType::FP32,
          max_total_tokens,
          page_size
      ) {
    if (!model_.initialized()) {
        throw std::runtime_error(
            "PagedGenerateEngine model is not initialized"
        );
    }

    if (max_total_tokens <= 0) {
        throw std::runtime_error(
            "PagedGenerateEngine max_total_tokens must be positive"
        );
    }

    if (page_size <= 0) {
        throw std::runtime_error(
            "PagedGenerateEngine page_size must be positive"
        );
    }

    if (model_.config().vocab_size <= 0) {
        throw std::runtime_error(
            "PagedGenerateEngine vocab_size must be positive"
        );
    }
}

int64_t PagedGenerateEngine::add_request(
    const std::vector<int32_t>& prompt_ids,
    int64_t max_new_tokens,
    int32_t eos_token_id
) {
    if (prompt_ids.empty()) {
        throw std::runtime_error(
            "PagedGenerateEngine add_request prompt_ids must not be empty"
        );
    }

    if (max_new_tokens < 0) {
        throw std::runtime_error(
            "PagedGenerateEngine add_request max_new_tokens must be non-negative"
        );
    }

    return request_manager_.add_request(
        prompt_ids,
        max_new_tokens,
        eos_token_id
    );
}

void PagedGenerateEngine::append_input_tokens(
    int64_t request_id,
    const std::vector<int32_t>& token_ids,
    int64_t max_new_tokens,
    int32_t eos_token_id
) {
    request_manager_.append_input_tokens(
        request_id,
        token_ids
    );

    request_manager_.reset_generation_options(
        request_id,
        max_new_tokens,
        eos_token_id
    );
}

int32_t PagedGenerateEngine::prefill(int64_t request_id) {
    GenerationRequest& req =
        request_manager_.request(request_id);

    if (req.status == RequestStatus::Finished) {
        return -1;
    }

    if (req.extend_len() <= 0) {
        throw std::runtime_error(
            "PagedGenerateEngine prefill requires extend_len > 0"
        );
    }

    int32_t next_token_id = -1;

    /*
     * 初始 prompt:
     *   cached_len = 0
     *   可以一次性 prefill 多个 token。
     *
     * 多轮对话追加 user tokens:
     *   cached_len > 0
     *   当前 Qwen3Attention 的 cached decode 只支持 seq_len=1。
     *   所以这里逐 token 补进 KV cache，最后一次 logits 用来采样 assistant 首 token。
     */
    if (req.cached_len == 0) {
        const int64_t seq_len = req.extend_len();

        next_token_id = forward_uncached_chunk_and_sample(
            request_id,
            seq_len,
            "prefill"
        );
    } else {
        while (req.extend_len() > 0) {
            next_token_id = forward_uncached_chunk_and_sample(
                request_id,
                1,
                "prefill_extend"
            );
        }
    }

    append_sampled_token(
        request_id,
        next_token_id
    );

    return next_token_id;
}

int32_t PagedGenerateEngine::decode_one_step(
    int64_t request_id
) {
    GenerationRequest& req =
        request_manager_.request(request_id);

    if (req.status == RequestStatus::Finished) {
        return -1;
    }

    if (req.extend_len() != 1) {
        throw std::runtime_error(
            "PagedGenerateEngine decode_one_step expects extend_len == 1"
        );
    }

    const int32_t next_token_id =
        forward_uncached_chunk_and_sample(
            request_id,
            1,
            "decode"
        );

    append_sampled_token(
        request_id,
        next_token_id
    );

    return next_token_id;
}

std::vector<int32_t> PagedGenerateEngine::decode_batch(
    const std::vector<int64_t>& active_request_ids
) {
    if (active_request_ids.empty()) {
        return {};
    }

    std::vector<int64_t> request_ids;
    request_ids.reserve(active_request_ids.size());

    for (int64_t request_id : active_request_ids) {
        GenerationRequest& req =
            request_manager_.request(request_id);

        if (req.status == RequestStatus::Finished) {
            continue;
        }

        if (req.extend_len() != 1) {
            throw std::runtime_error(
                "PagedGenerateEngine decode_batch expects every active request extend_len == 1"
            );
        }

        paged_kv_manager_.ensure_blocks(
            req,
            req.cached_len + 1
        );

        request_ids.push_back(request_id);
    }

    if (request_ids.empty()) {
        return {};
    }

    const int64_t batch_size =
        static_cast<int64_t>(request_ids.size());

    std::vector<int32_t> input_ids_cpu;
    std::vector<int32_t> position_ids_cpu;
    std::vector<int64_t> table_indices;
    std::vector<int64_t> past_lens;
    std::vector<int64_t> kv_seq_lens;

    input_ids_cpu.reserve(static_cast<size_t>(batch_size));
    position_ids_cpu.reserve(static_cast<size_t>(batch_size));
    table_indices.reserve(static_cast<size_t>(batch_size));
    past_lens.reserve(static_cast<size_t>(batch_size));
    kv_seq_lens.reserve(static_cast<size_t>(batch_size));

    for (int64_t request_id : request_ids) {
        const GenerationRequest& req =
            request_manager_.request(request_id);

        if (req.table_idx < 0) {
            throw std::runtime_error(
                "PagedGenerateEngine decode_batch request has no table_idx"
            );
        }

        input_ids_cpu.push_back(req.input_ids.back());
        position_ids_cpu.push_back(
            static_cast<int32_t>(req.cached_len)
        );
        table_indices.push_back(req.table_idx);
        past_lens.push_back(req.cached_len);
        kv_seq_lens.push_back(req.cached_len + 1);
    }

    Tensor input_ids_tensor =
        Tensor::from_int32_vector(
            input_ids_cpu,
            options_.device
        );

    Tensor position_ids_tensor =
        Tensor::from_int32_vector(
            position_ids_cpu,
            options_.device
        );

    Tensor logits(
        {batch_size, model_.config().vocab_size},
        DType::FP32,
        options_.device
    );

    BatchDecodeForwardContext context;
    context.position_ids = &position_ids_tensor;
    context.batch_size = batch_size;
    context.use_paged_kv_cache = true;
    context.paged_kv_cache = &paged_kv_manager_.paged_kv_cache();
    context.block_table_manager = &paged_kv_manager_.block_table_manager();
    context.table_indices = &table_indices;
    context.past_lens = &past_lens;
    context.kv_seq_lens = &kv_seq_lens;

    const TensorMemorySnapshot mem_before =
        tensor_memory_snapshot();

    model_.forward_decode_batch(
        input_ids_tensor,
        context,
        logits
    );

    const TensorMemorySnapshot mem_after =
        tensor_memory_snapshot();

    if (options_.verbose) {
        print_tensor_memory_delta(
            "paged_engine batch_decode forward",
            mem_before,
            mem_after
        );
    }

    const std::vector<int32_t> next_token_ids =
        argmax_each_row(logits);

    if (next_token_ids.size() != request_ids.size()) {
        throw std::runtime_error(
            "PagedGenerateEngine decode_batch sampled token count mismatch"
        );
    }

    for (size_t i = 0; i < request_ids.size(); ++i) {
        const int64_t request_id = request_ids[i];
        const int32_t token_id = next_token_ids[i];

        request_manager_.mark_forward_done(
            request_id,
            1
        );

        append_sampled_token(
            request_id,
            token_id
        );

        if (options_.verbose) {
            const GenerationRequest& updated_req =
                request_manager_.request(request_id);

            std::cerr << "[paged_engine] batch_decode"
                      << ", request_id="
                      << request_id
                      << ", cached_len="
                      << updated_req.cached_len
                      << ", device_len="
                      << updated_req.device_len()
                      << ", next_token_id="
                      << token_id
                      << std::endl;
        }
    }

    return next_token_ids;
}

std::vector<int32_t> PagedGenerateEngine::generate_until_finished(
    int64_t request_id
) {
    GenerationRequest& req =
        request_manager_.request(request_id);

    if (req.max_new_tokens == 0) {
        request_manager_.finish_request(request_id);
        return req.input_ids;
    }

    if (req.extend_len() > 0 &&
        req.generated_ids.empty() &&
        req.status != RequestStatus::Finished) {
        prefill(request_id);
    }

    while (!finished(request_id)) {
        decode_one_step(request_id);
    }

    return request_manager_.request(request_id).input_ids;
}

const GenerationRequest& PagedGenerateEngine::request(
    int64_t request_id
) const {
    return request_manager_.request(request_id);
}

GenerationRequest& PagedGenerateEngine::request(
    int64_t request_id
) {
    return request_manager_.request(request_id);
}

bool PagedGenerateEngine::finished(
    int64_t request_id
) const {
    return request_manager_.request(request_id).status ==
           RequestStatus::Finished;
}

void PagedGenerateEngine::release_request(
    int64_t request_id
) {
    GenerationRequest& req =
        request_manager_.request(request_id);

    paged_kv_manager_.release(req);

    request_manager_.finish_request(request_id);
}

RequestManager& PagedGenerateEngine::request_manager() {
    return request_manager_;
}

const RequestManager& PagedGenerateEngine::request_manager() const {
    return request_manager_;
}

PagedKVCacheManager& PagedGenerateEngine::paged_kv_manager() {
    return paged_kv_manager_;
}

const PagedKVCacheManager& PagedGenerateEngine::paged_kv_manager() const {
    return paged_kv_manager_;
}

std::vector<int32_t> PagedGenerateEngine::make_uncached_input_ids(
    const GenerationRequest& request,
    int64_t seq_len
) const {
    if (seq_len <= 0) {
        throw std::runtime_error(
            "PagedGenerateEngine seq_len must be positive"
        );
    }

    const int64_t start = request.cached_len;
    const int64_t end = start + seq_len;

    if (start < 0 || end < start) {
        throw std::runtime_error(
            "PagedGenerateEngine invalid uncached input range"
        );
    }

    if (end > request.device_len()) {
        throw std::runtime_error(
            "PagedGenerateEngine uncached input range exceeds device_len"
        );
    }

    return std::vector<int32_t>(
        request.input_ids.begin() + start,
        request.input_ids.begin() + end
    );
}

std::vector<int32_t> PagedGenerateEngine::make_uncached_position_ids(
    const GenerationRequest& request,
    int64_t seq_len
) const {
    if (seq_len <= 0) {
        throw std::runtime_error(
            "PagedGenerateEngine seq_len must be positive"
        );
    }

    const int64_t start = request.cached_len;
    const int64_t end = start + seq_len;

    if (end > request.device_len()) {
        throw std::runtime_error(
            "PagedGenerateEngine position range exceeds device_len"
        );
    }

    std::vector<int32_t> position_ids;
    position_ids.reserve(static_cast<size_t>(seq_len));

    for (int64_t i = start; i < end; ++i) {
        position_ids.push_back(
            static_cast<int32_t>(i)
        );
    }

    return position_ids;
}

int32_t PagedGenerateEngine::forward_uncached_chunk_and_sample(
    int64_t request_id,
    int64_t seq_len,
    const char* stage_name
) {
    GenerationRequest& req =
        request_manager_.request(request_id);

    if (req.status == RequestStatus::Finished) {
        throw std::runtime_error(
            "PagedGenerateEngine forward called on finished request"
        );
    }

    if (seq_len <= 0) {
        throw std::runtime_error(
            "PagedGenerateEngine forward seq_len must be positive"
        );
    }

    if (seq_len > req.extend_len()) {
        throw std::runtime_error(
            "PagedGenerateEngine forward seq_len exceeds extend_len"
        );
    }

    paged_kv_manager_.ensure_blocks(
        req,
        req.cached_len + seq_len
    );

    const std::vector<int32_t> uncached_input_ids =
        make_uncached_input_ids(
            req,
            seq_len
        );

    const std::vector<int32_t> position_ids_cpu =
        make_uncached_position_ids(
            req,
            seq_len
        );

    Tensor input_ids_tensor =
        Tensor::from_int32_vector(
            uncached_input_ids,
            options_.device
        );

    Tensor position_ids_tensor =
        Tensor::from_int32_vector(
            position_ids_cpu,
            options_.device
        );

    Tensor logits(
        {seq_len, model_.config().vocab_size},
        DType::FP32,
        options_.device
    );

    ForwardContext context;

    paged_kv_manager_.fill_forward_context(
        context,
        req,
        &position_ids_tensor,
        seq_len
    );

    const TensorMemorySnapshot mem_before =
        tensor_memory_snapshot();

    model_.forward(
        input_ids_tensor,
        context,
        logits
    );

    const TensorMemorySnapshot mem_after =
        tensor_memory_snapshot();

    if (options_.verbose) {
        const std::string tag =
            std::string("paged_engine ") +
            stage_name +
            " forward";

        print_tensor_memory_delta(
            tag,
            mem_before,
            mem_after
        );
    }

    request_manager_.mark_forward_done(
        request_id,
        seq_len
    );

    const int32_t next_token_id =
        argmax_last_token(logits);

    if (options_.verbose) {
        const GenerationRequest& updated_req =
            request_manager_.request(request_id);

        std::cerr << "[paged_engine] "
                  << stage_name
                  << ", request_id="
                  << request_id
                  << ", seq_len="
                  << seq_len
                  << ", cached_len="
                  << updated_req.cached_len
                  << ", device_len="
                  << updated_req.device_len()
                  << ", next_token_id="
                  << next_token_id
                  << std::endl;
    }

    return next_token_id;
}

void PagedGenerateEngine::append_sampled_token(
    int64_t request_id,
    int32_t token_id
) {
    request_manager_.append_token(
        request_id,
        token_id
    );
}

}  // namespace lite_llm

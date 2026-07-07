#pragma once

#include "core/tensor.hpp"

namespace lite_llm {

void flash_attention(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    Tensor& output
);

void flash_attention_kv_cache(
    const Tensor& q,
    const Tensor& key_cache,
    const Tensor& value_cache,
    int64_t kv_seq_len,
    Tensor& output
);

}  // namespace lite_llm
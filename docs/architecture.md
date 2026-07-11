# Architecture

LiteLLMEngine is organized as a small C++/CUDA inference stack. The main namespace is `lite_llm`.

## Directory Layout

```text
LiteLLMEngine/
├── CMakeLists.txt
├── include/
│   ├── benchmark/
│   ├── core/
│   ├── engine/
│   ├── layers/
│   ├── model/
│   ├── ops/
│   ├── runtime/
│   ├── tokenizer/
│   └── weights/
├── src/
│   ├── benchmark/
│   ├── core/
│   ├── engine/
│   ├── layers/
│   ├── model/
│   ├── ops/
│   ├── runtime/
│   ├── tokenizer/
│   ├── weights/
│   └── main.cpp
├── scripts/
├── tests/
└── docs/
```

## Core Layers

### `core/`

Owns low-level runtime primitives:

- `Tensor`: device-aware tensor storage, CPU/CUDA allocation, copies, zero fill.
- `Device` / `DType`: runtime device and data type metadata.
- CUDA error utilities.
- Tensor memory tracking helpers.

### `weights/`

Loads converted model weights and maps HuggingFace-style names to `Tensor` values. Model layers consume weights through `WeightMap`.

### `layers/`

Reusable neural network layers:

- `Embedding`
- `Linear`
- `RMSNorm`
- `RotaryEmbedding`

### `ops/`

Standalone operations and kernels:

- Attention and KV-cache attention.
- CUDA paged attention.
- Argmax sampling.
- Tensor copy.
- Elementwise add.
- SiLU-and-mul.

### `model/`

Qwen3 model implementation:

```text
Qwen3ForCausalLM
└── Qwen3Model
    ├── Embedding
    ├── Qwen3DecoderLayer x N
    │   ├── RMSNorm
    │   ├── Qwen3Attention
    │   ├── RMSNorm
    │   └── Qwen3MLP
    └── RMSNorm
```

The model supports both ordinary forward paths and decode-batch paths:

- `forward(...)`
- `forward_decode_batch(...)`

### `engine/`

Generation and request management:

- `GenerationRequest`: prompt, generated tokens, cached length, status, and block table index.
- `RequestManager`: request lifecycle and generated token accounting.
- `ModelKVCache`: continuous KV cache.
- `ModelPagedKVCache`: paged KV pool.
- `BlockTableManager`: maps logical request tokens to physical KV pages.
- `PagedKVCacheManager`: owns paged KV cache and block table manager.
- `PagedGenerateEngine`: prefill, single-request decode, and multi-request batch decode.

## Decode Data Flow

Single request paged decode:

```text
PagedGenerateEngine::decode_one_step(request_id)
  -> build input token and position tensor
  -> fill ForwardContext
  -> Qwen3ForCausalLM::forward
  -> Qwen3Attention writes K/V into ModelPagedKVCache
  -> paged attention reads K/V through BlockTableManager
  -> argmax_last_token
  -> RequestManager::mark_forward_done
  -> RequestManager::append_token
```

Batch decode:

```text
PagedGenerateEngine::decode_batch(active_request_ids)
  -> collect one uncached token per active request
  -> collect table_indices, past_lens, kv_seq_lens
  -> build BatchDecodeForwardContext
  -> Qwen3ForCausalLM::forward_decode_batch
  -> Qwen3Attention::forward_decode_batch
  -> write each request's new K/V into paged KV cache
  -> flash_attention_paged_kv_cache_batch_cuda
  -> argmax_each_row
  -> update each request independently
```

## Paged KV Cache

Each request owns a logical block table. The block table maps logical token pages to physical KV blocks in the global KV pool.

```text
logical token index
  -> logical page = token / page_size
  -> page offset = token % page_size
  -> physical block = block_table[logical_page]
  -> physical token = physical_block * page_size + page_offset
```

The CUDA batch paged attention kernel uses this mapping directly and avoids gathering every request's KV cache into a temporary contiguous buffer.

## CUDA Paged Attention Batch Kernel

Current kernel entry:

```cpp
flash_attention_paged_kv_cache_batch_cuda(...)
```

Kernel layout:

```text
grid.x = batch_size * num_q_heads
one CUDA block = one request row + one query head
threads reduce dot(q, k) over head_dim
softmax is streamed over kv_seq_len
value accumulation writes one output head
```

The host wrapper currently packs active block tables into a temporary CUDA `INT32` tensor each decode step. This keeps the implementation simple and correct, but it is a remaining optimization target.

# 项目架构

LiteLLMEngine 是一个小型 C++/CUDA 推理栈，项目主命名空间是 `lite_llm`。

## 目录结构

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

## 核心模块

### `core/`

底层运行时组件：

- `Tensor`：支持 CPU/CUDA 的 tensor 存储、内存分配、拷贝和清零。
- `Device` / `DType`：运行设备和数据类型元信息。
- CUDA 错误检查工具。
- Tensor 内存统计工具。

### `weights/`

负责加载转换后的模型权重，并通过 `WeightMap` 按 HuggingFace 风格的权重名提供给模型层。

### `layers/`

可复用神经网络层：

- `Embedding`
- `Linear`
- `RMSNorm`
- `RotaryEmbedding`

### `ops/`

独立算子和 CUDA kernel：

- Attention 和 KV-cache attention。
- CUDA paged attention。
- Argmax 采样。
- Tensor copy。
- Elementwise add。
- SiLU-and-mul。

### `model/`

Qwen3 模型实现：

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

模型同时支持普通 forward 路径和 batch decode 路径：

- `forward(...)`
- `forward_decode_batch(...)`

### `engine/`

生成和请求管理：

- `GenerationRequest`：保存 prompt、生成 token、cached length、状态和 block table 索引。
- `RequestManager`：管理 request 生命周期和生成 token 计数。
- `ModelKVCache`：连续 KV cache。
- `ModelPagedKVCache`：paged KV pool。
- `BlockTableManager`：把 request 的 logical token 映射到 physical KV page。
- `PagedKVCacheManager`：持有 paged KV cache 和 block table manager。
- `PagedGenerateEngine`：支持 prefill、单请求 decode 和多请求 batch decode。

## Decode 数据流

单请求 paged decode：

```text
PagedGenerateEngine::decode_one_step(request_id)
  -> 构造 input token 和 position tensor
  -> 填充 ForwardContext
  -> Qwen3ForCausalLM::forward
  -> Qwen3Attention 写入 K/V 到 ModelPagedKVCache
  -> paged attention 通过 BlockTableManager 读取 K/V
  -> argmax_last_token
  -> RequestManager::mark_forward_done
  -> RequestManager::append_token
```

Batch decode：

```text
PagedGenerateEngine::decode_batch(active_request_ids)
  -> 每个 active request 取一个未缓存 token
  -> 收集 table_indices、past_lens、kv_seq_lens
  -> 构造 BatchDecodeForwardContext
  -> Qwen3ForCausalLM::forward_decode_batch
  -> Qwen3Attention::forward_decode_batch
  -> 将每个 request 的新 K/V 写入 paged KV cache
  -> flash_attention_paged_kv_cache_batch_cuda
  -> argmax_each_row
  -> 分别更新每个 request 状态
```

## Paged KV Cache

每个 request 持有一张 logical block table。block table 将 logical token page 映射到全局 KV pool 中的 physical KV block。

```text
logical token index
  -> logical page = token / page_size
  -> page offset = token % page_size
  -> physical block = block_table[logical_page]
  -> physical token = physical_block * page_size + page_offset
```

CUDA batch paged attention kernel 直接使用这个映射读取 KV cache，避免为每个 request 先 gather 出连续 KV buffer。

## CUDA Paged Attention Batch Kernel

当前 kernel 入口：

```cpp
flash_attention_paged_kv_cache_batch_cuda(...)
```

kernel 组织方式：

```text
grid.x = batch_size * num_q_heads
一个 CUDA block = 一个 request row + 一个 query head
线程在 head_dim 维度上做 dot(q, k) reduce
softmax 沿 kv_seq_len 流式计算
value 累加后写出一个 output head
```

当前 host wrapper 会在每个 decode step 把 active block table 打包成临时 CUDA `INT32` tensor。这让实现保持简单和正确，但也是后续优化点。

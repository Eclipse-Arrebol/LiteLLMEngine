# LiteLLMEngine

[English README](readme.md)

LiteLLMEngine 是一个用 C++/CUDA 实现的轻量级 LLM 推理引擎，当前主要支持 Qwen3。项目覆盖了从 Tensor 存储、权重加载、模型算子，到 KV cache、Paged KV cache、多请求生成、batch decode，以及 CUDA paged attention batch kernel 的完整推理链路。

这个项目的目标是把现代 LLM decode serving 的关键机制拆开实现：request 如何映射到 paged KV cache block，decode 阶段如何把 active request 合批，以及 attention kernel 如何通过 block table 直接读取离散 KV page。

## 项目亮点

- 使用 C++/CUDA 实现 Qwen3 causal LM 推理。
- 实现核心算子：Embedding、Linear、RMSNorm、RoPE、Attention、MLP、Add、Argmax、SiLU-and-mul。
- 支持连续 KV cache 和 Paged KV cache。
- 实现多请求 `PagedGenerateEngine`，支持 prefill、decode 和 batch decode。
- 实现 CUDA paged attention batch kernel，用于 decode 阶段直接访问 paged KV cache。
- 提供 baseline、KV cache、paged interleaved decode、paged batch decode 的 benchmark 路径。

## 文档导航

- [项目架构](docs/architecture_zh.md) / [English](docs/architecture.md)
- [代码规范](docs/code_style_zh.md) / [English](docs/code_style.md)

## 构建

依赖：

```bash
sudo apt install -y nlohmann-json3-dev
```

配置和编译：

```bash
cmake -S . -B build
cmake --build build -j
```

运行测试：

```bash
ctest --test-dir build --output-on-failure
```

## 运行

```bash
./build/LiteLLMEngine \
  --model /root/rivermind-data/Qwen_Qwen3-0.6B \
  --prompt "Introduce CUDA briefly." \
  --max-tokens 64 \
  --device cuda \
  --temperature 0 \
  --eos-token-id 151645
```

## 性能

以下结果基于 Qwen3-0.6B 和 CUDA。主要 batch decode benchmark 使用 32 个请求，每个请求生成 64 个新 token。

| 阶段 | 模式 | 吞吐 | 主要优化 |
|---|---|---:|---|
| Baseline | 32 new tokens | 10.12 tok/s | 每个 decode step 都做全序列 forward |
| Baseline | 128 new tokens | 7.66 tok/s | 全序列 forward，序列越长开销越大 |
| KV cache | 128 new tokens | 14.03 tok/s | decode 阶段复用历史 key/value |
| Paged batch decode wrapper | 32 requests x 64 new tokens | 44.60 tok/s | active request 合批，但 attention 仍逐 request gather |
| Paged batch decode kernel | 32 requests x 64 new tokens | 189.06 tok/s | CUDA paged attention batch kernel 直接读取 KV page |

Baseline KV cache：

```bash
./build/LiteLLMEngine \
  --model /root/rivermind-data/Qwen_Qwen3-0.6B \
  --prompt "Introduce CUDA briefly." \
  --max-tokens 64 \
  --device cuda \
  --benchmark \
  --benchmark-requests 32 \
  --benchmark-warmup 1 \
  --use-kv-cache
```

Paged KV interleaved decode：

```bash
./build/LiteLLMEngine \
  --model /root/rivermind-data/Qwen_Qwen3-0.6B \
  --prompt "Introduce CUDA briefly." \
  --max-tokens 64 \
  --device cuda \
  --benchmark \
  --benchmark-requests 32 \
  --benchmark-warmup 1 \
  --use-paged-kv-cache \
  --benchmark-interleaved \
  --page-size 16
```

Paged KV batch decode：

```bash
./build/LiteLLMEngine \
  --model /root/rivermind-data/Qwen_Qwen3-0.6B \
  --prompt "Introduce CUDA briefly." \
  --max-tokens 64 \
  --device cuda \
  --benchmark \
  --benchmark-requests 32 \
  --benchmark-warmup 1 \
  --use-paged-kv-cache \
  --benchmark-batch-decode \
  --page-size 16
```

## 下一步

- 将 block table 常驻 GPU，避免每个 decode step 重新打包和拷贝。
- 减少 decode 阶段临时 Tensor 分配。
- 在 benchmark 中拆分 prefill 时间、decode 时间和内存流量。
- 将 paged attention kernel 从当前 FP32 实现扩展到更多数据类型。

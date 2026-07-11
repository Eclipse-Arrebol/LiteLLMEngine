# LiteLLMEngine

[中文 README](readme_zh.md)

LiteLLMEngine is a C++/CUDA learning-oriented LLM inference engine for Qwen3. It implements the core inference stack from tensor storage and weight loading up to KV cache, paged KV cache, multi-request generation, batch decode, and a CUDA paged attention batch kernel.

The project is intended to show how modern LLM decode serving works internally: requests are mapped to paged KV cache blocks, active decode tokens are batched, and attention reads KV pages directly through block tables.

## Highlights

- Qwen3 causal LM inference in C++/CUDA.
- Core operators: Embedding, Linear, RMSNorm, RoPE, Attention, MLP, Add, Argmax, SiLU-and-mul.
- Continuous KV cache and paged KV cache.
- Multi-request `PagedGenerateEngine` with prefill, decode, and batch decode.
- CUDA paged attention batch kernel for decode-time KV lookup.
- Benchmark path for baseline, KV cache, paged interleaved decode, and paged batch decode.

## Repository Guide

- [Architecture](docs/architecture.md) / [中文](docs/architecture_zh.md)
- [Code Style](docs/code_style.md) / [中文](docs/code_style_zh.md)

## Build

Dependencies:

```bash
sudo apt install -y nlohmann-json3-dev
```

Configure and build:

```bash
cmake -S . -B build
cmake --build build -j
```

Run tests:

```bash
ctest --test-dir build --output-on-failure
```

## Run

```bash
./build/LiteLLMEngine \
  --model /root/rivermind-data/Qwen_Qwen3-0.6B \
  --prompt "Introduce CUDA briefly." \
  --max-tokens 64 \
  --device cuda \
  --temperature 0 \
  --eos-token-id 151645
```

## Benchmarks

Measured on Qwen3-0.6B with CUDA. The main batch decode benchmark uses 32 requests and 64 new tokens per request.

| Stage | Mode | Throughput | Main Optimization |
|---|---|---:|---|
| Baseline | 32 new tokens | 10.12 tok/s | Full-sequence forward every decode step |
| Baseline | 128 new tokens | 7.66 tok/s | Full-sequence forward; cost grows with sequence length |
| KV cache | 128 new tokens | 14.03 tok/s | Reuse past key/value states during decode |
| Paged batch decode wrapper | 32 requests x 64 new tokens | 44.60 tok/s | Batch active requests; attention still gathers per request |
| Paged batch decode kernel | 32 requests x 64 new tokens | 189.06 tok/s | CUDA paged attention batch kernel reads KV pages directly |

Baseline KV cache:

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

Paged KV interleaved decode:

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

Paged KV batch decode:

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

## Next Steps

- Keep block tables resident on GPU instead of rebuilding/copying them each decode step.
- Reduce temporary tensor allocation in decode.
- Add more benchmark reporting for prefill time, decode time, and memory traffic.
- Extend paged attention kernels beyond the current FP32 implementation.

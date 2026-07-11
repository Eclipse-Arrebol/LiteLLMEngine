# Code Style

This project uses a small, direct C++ style. Prefer local consistency over introducing new abstractions.

## File Names

Use lowercase names with underscores:

```text
tensor.hpp
tensor.cpp
weight_loader.hpp
weight_loader.cpp
model_config.hpp
paged_generate_engine.hpp
paged_kv_cache.cpp
```

Headers live under `include/`, implementations live under `src/`, and tests live under `tests/`.

CUDA source files use `.cu`:

```text
attention.cu
argmax.cu
linear.cu
rms_norm.cu
rotary.cu
```

## Namespaces

All project code should live in:

```cpp
namespace lite_llm {

}  // namespace lite_llm
```

Internal helpers can use anonymous namespaces inside `.cpp` or `.cu` files.

## Type Names

Classes and structs use PascalCase:

```cpp
class Tensor;
class WeightMap;
class Qwen3ForCausalLM;
class PagedGenerateEngine;

struct ModelConfig;
struct ForwardContext;
struct BatchDecodeForwardContext;
```

## Function Names

Functions use lowercase with underscores:

```cpp
load_weights()
forward()
forward_decode_batch()
copy_from_cpu()
copy_to_cpu()
argmax_each_row()
```

CUDA kernels use a `_kernel` suffix:

```cpp
__global__ void flash_attention_paged_kv_cache_batch_kernel(...);
```

Public launcher/wrapper functions should describe the operation and backend:

```cpp
flash_attention_paged_kv_cache_cuda(...)
flash_attention_paged_kv_cache_batch_cuda(...)
```

## Variables

Local variables use lowercase with underscores:

```cpp
int64_t hidden_size;
int64_t num_layers;
int64_t max_new_tokens;
std::string model_path;
```

Private data members use a trailing underscore:

```cpp
class Tensor {
private:
    void* data_ = nullptr;
    std::vector<int64_t> shape_;
    DType dtype_ = DType::FP32;
    Device device_ = Device::CPU;
};
```

Constants can use a `k` prefix:

```cpp
constexpr int kBlockSize = 256;
```

## Include Order

Use this order:

```cpp
#include "ops/attention.hpp"

#include "core/cuda_utils.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <vector>
```

Order:

```text
1. Current module's header
2. Other project headers
3. CUDA / third-party headers
4. C++ standard library headers
```

## Design Rules

- Keep `forward` methods focused on computation.
- Keep weight loading in `load_weights`.
- Let `Tensor` own memory lifetime.
- Keep request lifecycle in `RequestManager`.
- Keep KV block allocation in `BlockTableManager` / `PagedKVCacheManager`.
- Avoid putting CUDA kernel details directly into high-level model code.
- Add abstractions only when they simplify a real repeated pattern.

## Tests

Each core module should have a focused test:

```text
test_tensor
test_linear
test_rms_norm
test_qwen3_attention
test_paged_kv_cache
test_paged_attention_cuda
test_paged_generate_engine
```

Run the full test suite with:

```bash
ctest --test-dir build --output-on-failure
```

# 代码规范

本项目使用直接、简单的 C++ 风格。优先保持项目内一致性，不为了抽象而抽象。

## 文件命名

文件名使用小写加下划线：

```text
tensor.hpp
tensor.cpp
weight_loader.hpp
weight_loader.cpp
model_config.hpp
paged_generate_engine.hpp
paged_kv_cache.cpp
```

头文件放在 `include/`，实现放在 `src/`，测试放在 `tests/`。

CUDA 源文件使用 `.cu`：

```text
attention.cu
argmax.cu
linear.cu
rms_norm.cu
rotary.cu
```

## 命名空间

项目代码统一放在：

```cpp
namespace lite_llm {

}  // namespace lite_llm
```

只在 `.cpp` 或 `.cu` 内部使用的 helper 可以放在匿名 namespace。

## 类型命名

类名和结构体名使用大驼峰：

```cpp
class Tensor;
class WeightMap;
class Qwen3ForCausalLM;
class PagedGenerateEngine;

struct ModelConfig;
struct ForwardContext;
struct BatchDecodeForwardContext;
```

## 函数命名

函数名使用小写加下划线：

```cpp
load_weights()
forward()
forward_decode_batch()
copy_from_cpu()
copy_to_cpu()
argmax_each_row()
```

CUDA kernel 使用 `_kernel` 后缀：

```cpp
__global__ void flash_attention_paged_kv_cache_batch_kernel(...);
```

公开的 launcher 或 wrapper 函数应说明操作和后端：

```cpp
flash_attention_paged_kv_cache_cuda(...)
flash_attention_paged_kv_cache_batch_cuda(...)
```

## 变量命名

局部变量使用小写加下划线：

```cpp
int64_t hidden_size;
int64_t num_layers;
int64_t max_new_tokens;
std::string model_path;
```

私有成员变量使用下划线后缀：

```cpp
class Tensor {
private:
    void* data_ = nullptr;
    std::vector<int64_t> shape_;
    DType dtype_ = DType::FP32;
    Device device_ = Device::CPU;
};
```

常量可以使用 `k` 前缀：

```cpp
constexpr int kBlockSize = 256;
```

## Include 顺序

推荐顺序：

```cpp
#include "ops/attention.hpp"

#include "core/cuda_utils.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <vector>
```

顺序说明：

```text
1. 当前模块自己的头文件
2. 其他项目内头文件
3. CUDA / 第三方头文件
4. C++ 标准库头文件
```

## 设计原则

- `forward` 方法只负责计算。
- 权重加载放在 `load_weights`。
- `Tensor` 负责内存生命周期。
- request 生命周期放在 `RequestManager`。
- KV block 分配放在 `BlockTableManager` / `PagedKVCacheManager`。
- 高层模型代码不要直接暴露 CUDA kernel 细节。
- 只有在能消除真实重复或简化复杂度时才新增抽象。

## 测试

每个核心模块都应有聚焦测试：

```text
test_tensor
test_linear
test_rms_norm
test_qwen3_attention
test_paged_kv_cache
test_paged_attention_cuda
test_paged_generate_engine
```

运行完整测试：

```bash
ctest --test-dir build --output-on-failure
```


## 目标架构

``` 
NebulaServe/
├── CMakeLists.txt
├── README.md
├── config/
│   └── model_config.json
├── include/
│   ├── core/
│   │   ├── tensor.hpp
│   │   ├── dtype.hpp
│   │   ├── device.hpp
│   │   ├── cuda_context.hpp
│   │   └── error.hpp
│   │
│   ├── weights/
│   │   ├── weight_loader.hpp
│   │   └── weight_map.hpp
│   │
│   ├── ops/
│   │   ├── embedding.hpp
│   │   ├── linear.hpp
│   │   ├── rmsnorm.hpp
│   │   ├── rope.hpp
│   │   ├── attention.hpp
│   │   ├── mlp.hpp
│   │   ├── activation.hpp
│   │   └── sampler.hpp
│   │
│   ├── model/
│   │   ├── model_config.hpp
│   │   ├── kv_cache.hpp
│   │   ├── decoder_layer.hpp
│   │   └── llama_model.hpp
│   │
│   └── runtime/
│       ├── request.hpp
│       ├── tokenizer.hpp
│       └── engine.hpp
│
├── src/
│   ├── core/
│   │   ├── tensor.cpp
│   │   ├── cuda_context.cpp
│   │   └── error.cpp
│   │
│   ├── weights/
│   │   ├── weight_loader.cpp
│   │   └── weight_map.cpp
│   │
│   ├── ops/
│   │   ├── embedding.cpp
│   │   ├── linear.cpp
│   │   ├── rmsnorm.cpp
│   │   ├── rope.cpp
│   │   ├── attention.cpp
│   │   ├── mlp.cpp
│   │   └── sampler.cpp
│   │
│   ├── model/
│   │   ├── kv_cache.cpp
│   │   ├── decoder_layer.cpp
│   │   └── llama_model.cpp
│   │
│   ├── runtime/
│   │   ├── tokenizer.cpp
│   │   └── engine.cpp
│   │
│   └── main.cpp
│
├── cuda/
│   ├── embedding_kernel.cu
│   ├── rmsnorm_kernel.cu
│   ├── rope_kernel.cu
│   ├── add_kernel.cu
│   ├── activation_kernel.cu
│   ├── attention_kernel.cu
│   ├── kv_cache_kernel.cu
│   └── sampler_kernel.cu
│
├── scripts/
│   ├── convert_weights.py
│   └── test_export_toy_model.py
│
├── tests/
│   ├── test_tensor.cpp
│   ├── test_weight_loader.cpp
│   ├── test_linear.cpp
│   ├── test_rmsnorm.cpp
│   ├── test_kv_cache.cpp
│   └── test_llama_model.cpp
│
└── benchmarks/
    ├── bench_linear.cpp
    ├── bench_rmsnorm.cpp
    └── bench_attention.cpp
```


## 开发规范
### 1. 文件命名

统一使用小写加下划线：

```text
tensor.hpp
tensor.cpp
weight_loader.hpp
weight_loader.cpp
cuda_context.hpp
model_config.hpp
llama_model.hpp
decoder_layer.hpp
kv_cache.hpp
```

CUDA 文件：

```text
rmsnorm_kernel.cu
rope_kernel.cu
attention_kernel.cu
kv_cache_kernel.cu
sampler_kernel.cu
```

头文件放 `include/`，实现放 `src/`，CUDA kernel 放 `cuda/`。

---

### 2. 类名命名

类名使用大驼峰：

```cpp
class Tensor;
class WeightLoader;
class LlamaModel;
class DecoderLayer;
class KVCache;
class CudaContext;
class SelfAttention;
```

结构体也用大驼峰：

```cpp
struct ModelConfig;
struct GenerateConfig;
struct Request;
```

---

### 3. 函数命名

函数使用小写加下划线：

```cpp
load_tensor()
load_config()
forward()
forward_one_token()
copy_from_cpu()
copy_to_cpu()
numel()
nbytes()
```

CUDA kernel launcher 也用小写加下划线：

```cpp
launch_rmsnorm_kernel()
launch_rope_kernel()
launch_embedding_kernel()
launch_kv_cache_append_kernel()
```

真正的 `__global__` kernel 可以加 `_kernel` 后缀：

```cpp
__global__ void rmsnorm_kernel(...);
__global__ void rope_kernel(...);
__global__ void embedding_kernel(...);
```

---

### 4. 变量命名

普通变量用小写加下划线：

```cpp
int hidden_size;
int num_layers;
int max_seq_len;
float rms_norm_eps;
std::string model_path;
```

成员变量加下划线后缀：

```cpp
class Tensor {
private:
    void* data_;
    std::vector<int> shape_;
    DType dtype_;
    Device device_;
};
```

常量可以用大写或 `k` 前缀。推荐 `k` 前缀：

```cpp
constexpr int kDefaultMaxTokens = 128;
constexpr float kDefaultTemperature = 0.7f;
```

---

### 5. 命名空间

项目统一放在：

```cpp
namespace nebula {
}
```

后面所有核心代码都写在里面：

```cpp
namespace nebula {

class Tensor {
};

} // namespace nebula
```

CUDA kernel 如果不想暴露，可以放匿名 namespace，launcher 暴露在 `nebula` 里。

---

### 6. include 顺序

统一这样：

```cpp
#include "ops/linear.hpp"

#include <cuda_runtime.h>

#include <iostream>
#include <memory>
#include <vector>
```

顺序是：

```text
1. 当前模块自己的头文件
2. 第三方库 / CUDA
3. C++ 标准库
```

---

### 7. 类设计原则

先定这几条：

```text
1. 一个类只做一件事
2. forward 只负责计算，不负责加载权重
3. 权重在构造函数中传入
4. Tensor 负责内存生命周期
5. Model 负责组织模块调用
6. CUDA kernel 不直接出现在业务逻辑里，统一通过 launch_xxx_kernel 调用
```

例如：

```cpp
class RMSNorm {
public:
    RMSNorm(Tensor weight, float eps);

    void forward(
        const Tensor& input,
        Tensor& output,
        cudaStream_t stream
    );

private:
    Tensor weight_;
    float eps_;
};
```


## 依赖


```
sudo apt install -y nlohmann-json3-dev
```


## 运行命令
```
./build/LiteLLMEngine \
  --model /root/rivermind-data/Qwen_Qwen3-0.6B \
  --prompt "Introduce CUDA briefly." \
  --max-tokens 64 \
  --device cuda \
  --temperature 0 \
  --eos-token-id 151645 \
  --verbose
```
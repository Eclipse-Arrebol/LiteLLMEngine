#include "layers/linear.hpp"
#include "core/cuda_utils.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>
#include <utility>

namespace lite_llm {

namespace{
void check_cublas(cublasStatus_t status, const char* expr) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuBLAS failed: ") + expr);
    }
}


void check_linear_forward_args(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& output,
    bool has_bias,
    const Tensor* bias
) {
    if (input.dtype() != DType::FP32) {
        throw std::runtime_error("Linear input must be FP32");
    }

    if (weight.dtype() != DType::FP32) {
        throw std::runtime_error("Linear weight must be FP32");
    }

    if (output.dtype() != DType::FP32) {
        throw std::runtime_error("Linear output must be FP32");
    }

    if (input.shape().size() != 2) {
        throw std::runtime_error("Linear input must be 2D: [M, in_features]");
    }

    if (weight.shape().size() != 2) {
        throw std::runtime_error("Linear weight must be 2D: [out_features, in_features]");
    }

    if (output.shape().size() != 2) {
        throw std::runtime_error("Linear output must be 2D: [M, out_features]");
    }

    const int64_t m = input.shape()[0];
    const int64_t in_features = input.shape()[1];
    const int64_t out_features = weight.shape()[0];

    if (m < 0 || in_features <= 0 || out_features <= 0) {
        throw std::runtime_error("Linear invalid shape");
    }

    if (weight.shape()[1] != in_features) {
        throw std::runtime_error("Linear weight shape mismatch");
    }

    if (output.shape()[0] != m || output.shape()[1] != out_features) {
        throw std::runtime_error("Linear output shape mismatch");
    }

    if (input.device() != weight.device() || input.device() != output.device()) {
        throw std::runtime_error("Linear input, weight and output must be on same device");
    }

    if (has_bias) {
        if (bias == nullptr) {
            throw std::runtime_error("Linear bias is null");
        }

        if (bias->dtype() != DType::FP32) {
            throw std::runtime_error("Linear bias must be FP32");
        }

        if (bias->shape().size() != 1 || bias->shape()[0] != out_features) {
            throw std::runtime_error("Linear bias shape mismatch");
        }

        if (bias->device() != input.device()) {
            throw std::runtime_error("Linear bias device mismatch");
        }
    }
}

void linear_forward_cpu(
    const Tensor& input,
    const Tensor& weight,
    Tensor& output,
    bool has_bias,
    const Tensor* bias
) {
    const float* x = input.ptr<float>();
    const float* w = weight.ptr<float>();
    float* out = output.ptr<float>();

    const int64_t m = input.shape()[0];
    const int64_t in_features = input.shape()[1];
    const int64_t out_features = weight.shape()[0];

    const float* b = nullptr;
    if (has_bias) {
        b = bias->ptr<float>();
    }

    for (int64_t row = 0; row < m; ++row) {
        for (int64_t out_col = 0; out_col < out_features; ++out_col) {
            float sum = has_bias ? b[out_col] : 0.0f;

            for (int64_t k = 0; k < in_features; ++k) {
                sum += x[row * in_features + k] * w[out_col * in_features + k];
            }

            out[row * out_features + out_col] = sum;
        }
    }
}


__global__ void linear_add_bias_kernel(
    float* output,
    const float* bias,
    int64_t m,
    int64_t out_features
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = m * out_features;

    if (idx >= total) {
        return;
    }

    const int64_t out_col = idx % out_features;
    output[idx] += bias[out_col];
}

void linear_forward_cuda(
    const Tensor& input,
    const Tensor& weight,
    Tensor& output,
    bool has_bias,
    const Tensor* bias
) {
    const int64_t m64 = input.shape()[0];
    const int64_t k64 = input.shape()[1];
    const int64_t n64 = weight.shape()[0];

    if (m64 == 0) {
        return;
    }

    if (
        m64 > std::numeric_limits<int>::max() ||
        k64 > std::numeric_limits<int>::max() ||
        n64 > std::numeric_limits<int>::max()
    ) {
        throw std::runtime_error("Linear CUDA shape too large for cuBLAS int API");
    }

    const int m = static_cast<int>(m64);
    const int k = static_cast<int>(k64);
    const int n = static_cast<int>(n64);

    cublasHandle_t handle = nullptr;
    check_cublas(cublasCreate(&handle), "cublasCreate");

    const float alpha = 1.0f;
    const float beta = 0.0f;

    /*
      Row-major view:
        input:  X [M, K]
        weight: W [N, K]
        output: Y [M, N]

      Need:
        Y = X * W^T

      cuBLAS uses column-major.
      Same memory can be viewed as:
        input_col  = X^T, shape [K, M]
        weight_col = W^T, shape [K, N]
        output_col = Y^T, shape [N, M]

      So:
        output_col = weight_col^T * input_col
                   = [N, K] * [K, M]
                   = [N, M]
    */
    cublasStatus_t status = cublasSgemm(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        weight.ptr<float>(),
        k,
        input.ptr<float>(),
        k,
        &beta,
        output.ptr<float>(),
        n
    );

    cublasStatus_t destroy_status = cublasDestroy(handle);

    check_cublas(status, "cublasSgemm");
    check_cublas(destroy_status, "cublasDestroy");

    if (has_bias) {
        const int64_t total = m64 * n64;

        dim3 block = cuda_make_1d_block(256);
        dim3 grid = cuda_make_1d_grid(total, block.x);

        linear_add_bias_kernel<<<grid, block>>>(
            output.ptr<float>(),
            bias->ptr<float>(),
            m64,
            n64
        );

        CUDA_KERNEL_CHECK();
    }
}



}


Linear::Linear(Tensor weight)
    : WeightedUnaryLayer(std::move(weight)) {
    if (weight_.shape().size() != 2) {
        throw std::runtime_error("Linear weight must be 2D");
    }

    if (weight_.dtype() != DType::FP32) {
        throw std::runtime_error("Linear weight must be FP32");
    }

    out_features_ = weight_.shape()[0];
    in_features_ = weight_.shape()[1];

    if (out_features_ <= 0 || in_features_ <= 0) {
        throw std::runtime_error("Linear got invalid weight shape");
    }
}

Linear::Linear(Tensor weight, Tensor bias)
    : Linear(std::move(weight)) {
    if (bias.shape().size() != 1) {
        throw std::runtime_error("Linear bias must be 1D");
    }

    if (bias.dtype() != DType::FP32) {
        throw std::runtime_error("Linear bias must be FP32");
    }

    if (bias.device() != weight_.device()) {
        throw std::runtime_error("Linear bias and weight must be on same device");
    }

    if (bias.shape()[0] != out_features_) {
        throw std::runtime_error("Linear bias shape mismatch");
    }

    bias_ = std::move(bias);
    has_bias_ = true;
}


void Linear::load_weight(Tensor weight) {
    if (weight.shape().size() != 2) {
        throw std::runtime_error("Linear weight must be 2D");
    }

    if (weight.dtype() != DType::FP32) {
        throw std::runtime_error("Linear weight must be FP32");
    }

    int64_t out_features = weight.shape()[0];
    int64_t in_features = weight.shape()[1];

    if (out_features <= 0 || in_features <= 0) {
        throw std::runtime_error("Linear got invalid weight shape");
    }

    weight_ = std::move(weight);
    out_features_ = out_features;
    in_features_ = in_features;

    bias_ = Tensor();
    has_bias_ = false;
}

void Linear::load_bias(Tensor bias) {
    if (!initialized()) {
        throw std::runtime_error("Linear::load_bias called before weight is initialized");
    }

    if (bias.shape().size() != 1) {
        throw std::runtime_error("Linear bias must be 1D");
    }

    if (bias.dtype() != DType::FP32) {
        throw std::runtime_error("Linear bias must be FP32");
    }

    if (bias.device() != weight_.device()) {
        throw std::runtime_error("Linear bias and weight must be on same device");
    }

    if (bias.shape()[0] != out_features_) {
        throw std::runtime_error("Linear bias shape mismatch");
    }

    bias_ = std::move(bias);
    has_bias_ = true;
}


void Linear::forward(const Tensor& input, Tensor& output) const {
    if (!initialized()) {
        throw std::runtime_error("Linear::forward called before weight is initialized");
    }

    const Tensor* bias_ptr = has_bias_ ? &bias_ : nullptr;

    check_linear_forward_args(
        input,
        weight_,
        output,
        has_bias_,
        bias_ptr
    );

    if (input.device() == Device::CPU) {
        linear_forward_cpu(input, weight_, output, has_bias_, bias_ptr);
        return;
    }

    if (input.device() == Device::CUDA) {
        linear_forward_cuda(input, weight_, output, has_bias_, bias_ptr);
        return;
    }

    throw std::runtime_error("Linear::forward not implemented yet");
}

} // namespace lite_llm
#include "ops/copy.hpp"

#include "core/cuda_utils.hpp"

#include <cuda_runtime.h>

#include <cstring>
#include <stdexcept>

namespace lite_llm {

namespace {

void check_tensor_copy_args(
    const Tensor& input,
    const Tensor& output
) {
    if (input.dtype() != output.dtype()) {
        throw std::runtime_error("tensor_copy dtype mismatch");
    }

    if (input.numel() != output.numel()) {
        throw std::runtime_error("tensor_copy numel mismatch");
    }

    if (input.device() != output.device()) {
        throw std::runtime_error("tensor_copy input and output must be on same device");
    }
}

void tensor_copy_cpu(
    const Tensor& input,
    Tensor& output
) {
    if (input.nbytes() == 0) {
        return;
    }

    std::memcpy(
        output.data(),
        input.data(),
        input.nbytes()
    );
}

void tensor_copy_cuda(
    const Tensor& input,
    Tensor& output
) {
    if (input.nbytes() == 0) {
        return;
    }

    CUDA_CHECK(cudaMemcpy(
        output.data(),
        input.data(),
        input.nbytes(),
        cudaMemcpyDeviceToDevice
    ));
}

}  // namespace

void tensor_copy(
    const Tensor& input,
    Tensor& output
) {
    check_tensor_copy_args(input, output);

    if (input.device() == Device::CPU) {
        tensor_copy_cpu(input, output);
        return;
    }

    if (input.device() == Device::CUDA) {
        tensor_copy_cuda(input, output);
        return;
    }

    throw std::runtime_error("Unsupported device for tensor_copy");
}

}  // namespace lite_llm
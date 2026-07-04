#include "core/tensor.hpp"

#include "core/cuda_utils.hpp"

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace lite_llm {

Tensor::Tensor(std::vector<int64_t> shape, DType dtype, Device device)
    : shape_(std::move(shape)),
      dtype_(dtype),
      device_(device) {
    allocate_();
}

Tensor::~Tensor() {
    release_();
}

Tensor::Tensor(Tensor&& other) noexcept {
    data_ = other.data_;
    shape_ = std::move(other.shape_);
    dtype_ = other.dtype_;
    device_ = other.device_;

    other.data_ = nullptr;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        release_();

        data_ = other.data_;
        shape_ = std::move(other.shape_);
        dtype_ = other.dtype_;
        device_ = other.device_;

        other.data_ = nullptr;
    }

    return *this;
}

size_t Tensor::numel() const {
    if (shape_.empty()) {
        return 0;
    }

    size_t result = 1;

    for (int64_t dim : shape_) {
        if (dim <= 0) {
            return 0;
        }

        result *= static_cast<size_t>(dim);
    }

    return result;
}

size_t Tensor::nbytes() const {
    return numel() * dtype_size(dtype_);
}

void Tensor::zero_() {
    if (data_ == nullptr) {
        return;
    }

    size_t bytes = nbytes();

    if (device_ == Device::CPU) {
        std::memset(data_, 0, bytes);
    } else if (device_ == Device::CUDA) {
        CUDA_CHECK(cudaMemset(data_, 0, bytes));
    } else {
        throw std::runtime_error("Unsupported device in Tensor::zero_()");
    }
}

void Tensor::copy_from_cpu(const void* src, size_t bytes) {
    if (src == nullptr) {
        throw std::runtime_error("copy_from_cpu got nullptr src");
    }

    if (bytes > nbytes()) {
        throw std::runtime_error("copy_from_cpu bytes exceed tensor storage size");
    }

    if (data_ == nullptr) {
        throw std::runtime_error("copy_from_cpu called on empty tensor");
    }

    if (device_ == Device::CPU) {
        std::memcpy(data_, src, bytes);
    } else if (device_ == Device::CUDA) {
        CUDA_CHECK(cudaMemcpy(data_, src, bytes, cudaMemcpyHostToDevice));
    } else {
        throw std::runtime_error("Unsupported device in Tensor::copy_from_cpu()");
    }
}

void Tensor::copy_to_cpu(void* dst, size_t bytes) const {
    if (dst == nullptr) {
        throw std::runtime_error("copy_to_cpu got nullptr dst");
    }

    if (bytes > nbytes()) {
        throw std::runtime_error("copy_to_cpu bytes exceed tensor storage size");
    }

    if (data_ == nullptr) {
        throw std::runtime_error("copy_to_cpu called on empty tensor");
    }

    if (device_ == Device::CPU) {
        std::memcpy(dst, data_, bytes);
    } else if (device_ == Device::CUDA) {
        CUDA_CHECK(cudaMemcpy(dst, data_, bytes, cudaMemcpyDeviceToHost));
    } else {
        throw std::runtime_error("Unsupported device in Tensor::copy_to_cpu()");
    }
}

void Tensor::allocate_() {
    size_t bytes = nbytes();

    if (bytes == 0) {
        data_ = nullptr;
        return;
    }

    if (device_ == Device::CPU) {
        data_ = std::malloc(bytes);

        if (data_ == nullptr) {
            throw std::bad_alloc();
        }
    } else if (device_ == Device::CUDA) {
        CUDA_CHECK(cudaMalloc(&data_, bytes));
    } else {
        throw std::runtime_error("Unsupported device in Tensor::allocate_()");
    }
}

void Tensor::release_() noexcept {
    if (data_ == nullptr) {
        return;
    }

    if (device_ == Device::CPU) {
        std::free(data_);
    } else if (device_ == Device::CUDA) {
        cudaFree(data_);
    }

    data_ = nullptr;
}

} // namespace lite_llm
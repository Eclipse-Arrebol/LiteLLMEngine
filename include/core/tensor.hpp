#pragma once

#include "core/device.hpp"
#include "core/dtype.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace lite_llm {

/**
 * @brief data_就是真实数据的指针，之所以使用void* 因为tensor可能接受不同类型的数据，所以不写死
 * 
 */
class Tensor {
public:
    Tensor() = default;

    Tensor(std::vector<int64_t> shape, DType dtype, Device device);

    ~Tensor();

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    void* data() const {
        return data_;
    }

    template <typename T>
    T* ptr() const {
        return static_cast<T*>(data_);
    }

    const std::vector<int64_t>& shape() const {
        return shape_;
    }

    DType dtype() const {
        return dtype_;
    }

    Device device() const {
        return device_;
    }

    size_t numel() const;
    size_t nbytes() const;

    bool empty() const {
        return data_ == nullptr;
    }

    void zero_();

    void copy_from_cpu(const void* src, size_t bytes);
    void copy_to_cpu(void* dst, size_t bytes) const;

private:
    void release_() noexcept;
    void allocate_();

private:
    void* data_ = nullptr;
    std::vector<int64_t> shape_;
    DType dtype_ = DType::FP32;
    Device device_ = Device::CPU;
};

} // namespace lite_llm
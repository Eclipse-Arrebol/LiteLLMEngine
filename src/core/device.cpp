#include "core/device.hpp"

#include <cuda_runtime.h>

#include <stdexcept>

namespace lite_llm {

std::string device_to_string(Device device) {
    switch (device) {
        case Device::CPU:
            return "cpu";
        case Device::CUDA:
            return "cuda";
        default:
            return "unknown";
    }
}

Device resolve_device(const std::string& device_arg) {
    if (device_arg == "cpu") {
        return Device::CPU;
    }

    if (device_arg == "cuda" || device_arg == "gpu") {
        return Device::CUDA;
    }

    if (device_arg == "auto") {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);

        if (err == cudaSuccess && device_count > 0) {
            return Device::CUDA;
        }

        return Device::CPU;
    }

    throw std::runtime_error("Unsupported device: " + device_arg);
}

} // namespace lite_llm
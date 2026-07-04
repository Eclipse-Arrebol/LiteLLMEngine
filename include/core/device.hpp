#pragma once

#include <string>

namespace lite_llm {

enum class Device {
    CPU,
    CUDA
};

std::string device_to_string(Device device);

Device resolve_device(const std::string& device_arg);

} // namespace lite_llm
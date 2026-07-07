#pragma once

#include "core/device.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace lite_llm {

struct TensorMemoryDeviceSnapshot {
    size_t current_bytes = 0;
    size_t peak_bytes = 0;

    size_t total_allocated_bytes = 0;
    size_t total_freed_bytes = 0;

    int64_t allocation_count = 0;
    int64_t free_count = 0;
};

struct TensorMemorySnapshot {
    TensorMemoryDeviceSnapshot cpu;
    TensorMemoryDeviceSnapshot cuda;
};

void tensor_memory_record_allocate(Device device, size_t bytes);
void tensor_memory_record_free(Device device, size_t bytes);

TensorMemorySnapshot tensor_memory_snapshot();

void print_tensor_memory_snapshot(
    const std::string& title,
    const TensorMemorySnapshot& snapshot
);

void print_tensor_memory_delta(
    const std::string& title,
    const TensorMemorySnapshot& before,
    const TensorMemorySnapshot& after
);

}  // namespace lite_llm
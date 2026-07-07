#include "core/tensor_memory_tracker.hpp"

#include <algorithm>
#include <iostream>
#include <mutex>

namespace lite_llm {
namespace {

struct TensorMemoryState {
    TensorMemoryDeviceSnapshot cpu;
    TensorMemoryDeviceSnapshot cuda;
    std::mutex mutex;
};

TensorMemoryState& global_tensor_memory_state() {
    static TensorMemoryState state;
    return state;
}

TensorMemoryDeviceSnapshot& pick_stats(
    TensorMemoryState& state,
    Device device
) {
    if (device == Device::CPU) {
        return state.cpu;
    }

    if (device == Device::CUDA) {
        return state.cuda;
    }

    return state.cpu;
}

const TensorMemoryDeviceSnapshot& pick_stats_const(
    const TensorMemorySnapshot& snapshot,
    Device device
) {
    if (device == Device::CPU) {
        return snapshot.cpu;
    }

    if (device == Device::CUDA) {
        return snapshot.cuda;
    }

    return snapshot.cpu;
}

void print_device_snapshot(
    const char* name,
    const TensorMemoryDeviceSnapshot& stats
) {
    const double current_mb =
        static_cast<double>(stats.current_bytes) / 1024.0 / 1024.0;

    const double peak_mb =
        static_cast<double>(stats.peak_bytes) / 1024.0 / 1024.0;

    const double total_alloc_mb =
        static_cast<double>(stats.total_allocated_bytes) / 1024.0 / 1024.0;

    const double total_free_mb =
        static_cast<double>(stats.total_freed_bytes) / 1024.0 / 1024.0;

    std::cout << "  " << name << ":\n";
    std::cout << "    current:       " << current_mb << " MB\n";
    std::cout << "    peak:          " << peak_mb << " MB\n";
    std::cout << "    total alloc:   " << total_alloc_mb << " MB\n";
    std::cout << "    total free:    " << total_free_mb << " MB\n";
    std::cout << "    alloc count:   " << stats.allocation_count << "\n";
    std::cout << "    free count:    " << stats.free_count << "\n";
}

TensorMemoryDeviceSnapshot diff_device_snapshot(
    const TensorMemoryDeviceSnapshot& before,
    const TensorMemoryDeviceSnapshot& after
) {
    TensorMemoryDeviceSnapshot diff;

    diff.current_bytes =
        after.current_bytes >= before.current_bytes
            ? after.current_bytes - before.current_bytes
            : 0;

    diff.peak_bytes =
        after.peak_bytes;

    diff.total_allocated_bytes =
        after.total_allocated_bytes - before.total_allocated_bytes;

    diff.total_freed_bytes =
        after.total_freed_bytes - before.total_freed_bytes;

    diff.allocation_count =
        after.allocation_count - before.allocation_count;

    diff.free_count =
        after.free_count - before.free_count;

    return diff;
}

}  // namespace

void tensor_memory_record_allocate(Device device, size_t bytes) {
    TensorMemoryState& state = global_tensor_memory_state();

    std::lock_guard<std::mutex> lock(state.mutex);

    TensorMemoryDeviceSnapshot& stats = pick_stats(state, device);

    stats.current_bytes += bytes;
    stats.peak_bytes = std::max(stats.peak_bytes, stats.current_bytes);
    stats.total_allocated_bytes += bytes;
    stats.allocation_count += 1;
}

void tensor_memory_record_free(Device device, size_t bytes) {
    TensorMemoryState& state = global_tensor_memory_state();

    std::lock_guard<std::mutex> lock(state.mutex);

    TensorMemoryDeviceSnapshot& stats = pick_stats(state, device);

    if (stats.current_bytes >= bytes) {
        stats.current_bytes -= bytes;
    } else {
        stats.current_bytes = 0;
    }

    stats.total_freed_bytes += bytes;
    stats.free_count += 1;
}

TensorMemorySnapshot tensor_memory_snapshot() {
    TensorMemoryState& state = global_tensor_memory_state();

    std::lock_guard<std::mutex> lock(state.mutex);

    TensorMemorySnapshot snapshot;
    snapshot.cpu = state.cpu;
    snapshot.cuda = state.cuda;

    return snapshot;
}

void print_tensor_memory_snapshot(
    const std::string& title,
    const TensorMemorySnapshot& snapshot
) {
    std::cout << "\nTensor memory snapshot: " << title << "\n";
    print_device_snapshot("CPU", snapshot.cpu);
    print_device_snapshot("CUDA", snapshot.cuda);
}

void print_tensor_memory_delta(
    const std::string& title,
    const TensorMemorySnapshot& before,
    const TensorMemorySnapshot& after
) {
    TensorMemorySnapshot diff;
    diff.cpu = diff_device_snapshot(before.cpu, after.cpu);
    diff.cuda = diff_device_snapshot(before.cuda, after.cuda);

    std::cout << "\nTensor memory delta: " << title << "\n";
    print_device_snapshot("CPU", diff.cpu);
    print_device_snapshot("CUDA", diff.cuda);
}

}  // namespace lite_llm
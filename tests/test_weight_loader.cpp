#include "core/device.hpp"
#include "core/dtype.hpp"
#include "core/tensor.hpp"
#include "weights/weight_loader.hpp"
#include "weights/weight_map.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error("Test failed: " + message);
    }
}

void write_binary_floats(
    const std::filesystem::path& path,
    const std::vector<float>& values
) {
    std::ofstream file(path, std::ios::binary);

    if (!file) {
        throw std::runtime_error("Failed to write binary file: " + path.string());
    }

    file.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float))
    );
}

void write_test_index(const std::filesystem::path& path) {
    std::ofstream file(path);

    if (!file) {
        throw std::runtime_error("Failed to write index file: " + path.string());
    }

    file << R"({
  "format": "litellm_engine_weights_v1",
  "tensors": [
    {
      "name": "a.weight",
      "filename": "a.weight.bin",
      "dtype": "fp32",
      "shape": [2, 3]
    },
    {
      "name": "b.weight",
      "filename": "b.weight.bin",
      "dtype": "fp32",
      "shape": [3]
    }
  ]
})";
}

} // namespace

int main() {
    try {
        std::cout << "[test_weight_loader] prepare files..." << std::endl;

        std::filesystem::path test_dir =
            std::filesystem::temp_directory_path() / "litellm_engine_weight_loader_test";

        std::filesystem::remove_all(test_dir);
        std::filesystem::create_directories(test_dir);

        write_test_index(test_dir / "weights_index.json");

        write_binary_floats(
            test_dir / "a.weight.bin",
            {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}
        );

        write_binary_floats(
            test_dir / "b.weight.bin",
            {10.0f, 20.0f, 30.0f}
        );

        lite_llm::WeightLoaderOptions options;
        options.device = lite_llm::Device::CPU;

        std::cout << "[test_weight_loader] load weights..." << std::endl;

        lite_llm::WeightMap weights =
            lite_llm::load_weight_map_from_directory(test_dir.string(), options);

        expect(weights.size() == 2, "WeightMap size mismatch");
        expect(weights.contains("a.weight"), "missing a.weight");
        expect(weights.contains("b.weight"), "missing b.weight");

        lite_llm::Tensor a = weights.take("a.weight");

        expect(a.shape().size() == 2, "a.weight rank mismatch");
        expect(a.shape()[0] == 2, "a.weight dim0 mismatch");
        expect(a.shape()[1] == 3, "a.weight dim1 mismatch");
        expect(a.dtype() == lite_llm::DType::FP32, "a.weight dtype mismatch");
        expect(a.device() == lite_llm::Device::CPU, "a.weight device mismatch");

        std::vector<float> host_a(6, 0.0f);
        a.copy_to_cpu(host_a.data(), host_a.size() * sizeof(float));

        for (size_t i = 0; i < host_a.size(); ++i) {
            expect(host_a[i] == static_cast<float>(i + 1), "a.weight value mismatch");
        }

        lite_llm::Tensor b = weights.take("b.weight");

        expect(b.shape().size() == 1, "b.weight rank mismatch");
        expect(b.shape()[0] == 3, "b.weight dim mismatch");

        std::vector<float> host_b(3, 0.0f);
        b.copy_to_cpu(host_b.data(), host_b.size() * sizeof(float));

        expect(host_b[0] == 10.0f, "b.weight value0 mismatch");
        expect(host_b[1] == 20.0f, "b.weight value1 mismatch");
        expect(host_b[2] == 30.0f, "b.weight value2 mismatch");

        expect(weights.empty(), "WeightMap should be empty after taking all weights");

        std::filesystem::remove_all(test_dir);

        std::cout << "[test_weight_loader] passed" << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
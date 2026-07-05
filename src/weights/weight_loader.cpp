#include "weights/weight_loader.hpp"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace lite_llm {

namespace {

DType parse_dtype(const std::string& dtype) {
    if (dtype == "fp32") {
        return DType::FP32;
    }

    if (dtype == "fp16") {
        return DType::FP16;
    }

    if (dtype == "int32") {
        return DType::INT32;
    }

    throw std::runtime_error("Unsupported weight dtype: " + dtype);
}

std::vector<int64_t> parse_shape(const nlohmann::json& shape_json) {
    if (!shape_json.is_array()) {
        throw std::runtime_error("Weight shape must be an array");
    }

    std::vector<int64_t> shape;
    shape.reserve(shape_json.size());

    for (const auto& dim_json : shape_json) {
        if (!dim_json.is_number_integer()) {
            throw std::runtime_error("Weight shape dim must be integer");
        }

        int64_t dim = dim_json.get<int64_t>();

        if (dim <= 0) {
            throw std::runtime_error("Weight shape dim must be positive");
        }

        shape.push_back(dim);
    }

    return shape;
}

std::vector<uint8_t> read_binary_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);

    std::string abs_path = std::filesystem::absolute(path).string();

    if (!file) {
        throw std::runtime_error("Failed to open weight binary file: " + abs_path);
    }

    std::streamsize size = file.tellg();

    if (size < 0) {
        throw std::runtime_error("Failed to get weight binary file size: " + abs_path);
    }

    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(static_cast<size_t>(size));

    if (!buffer.empty()) {
        file.read(reinterpret_cast<char*>(buffer.data()), size);

        if (!file) {
            throw std::runtime_error("Failed to read weight binary file: " + abs_path);
        }
    }

    return buffer;
}

nlohmann::json read_json_file(const std::filesystem::path& path) {
    std::ifstream file(path);

    std::string abs_path = std::filesystem::absolute(path).string();

    if (!file) {
        throw std::runtime_error("Failed to open weight index file: " + abs_path);
    }

    nlohmann::json json;
    file >> json;

    return json;
}

} // namespace


/**
 * @brief 这里定义了一个中间存储json的格式
 {
  "format": "litellm_engine_weights_v1",
  "tensors": [
    {
      "name": "model.embed_tokens.weight",
      "filename": "model.embed_tokens.weight.bin",
      "dtype": "fp32",
      "shape": [100, 16]
    }
  ]
}
 * 
 * @param directory 
 * @param options 
 * @return WeightMap 
 */
WeightMap load_weight_map_from_directory(
    const std::string& directory,
    const WeightLoaderOptions& options
) {
    std::filesystem::path dir_path(directory);
    std::filesystem::path index_path = dir_path / options.index_filename;

    nlohmann::json index = read_json_file(index_path);

    if (!index.contains("format") ||
        index.at("format").get<std::string>() != "litellm_engine_weights_v1") {
        throw std::runtime_error("Unsupported or missing weight format");
    }

    if (!index.contains("tensors") || !index.at("tensors").is_array()) {
        throw std::runtime_error("weights_index.json must contain tensors array");
    }

    WeightMap weights;

    for (const auto& item : index.at("tensors")) {
        if (!item.contains("name") ||
            !item.contains("filename") ||
            !item.contains("dtype") ||
            !item.contains("shape")) {
            throw std::runtime_error("Invalid tensor entry in weights_index.json");
        }

        std::string name = item.at("name").get<std::string>();
        std::string filename = item.at("filename").get<std::string>();
        std::string dtype_string = item.at("dtype").get<std::string>();

        DType dtype = parse_dtype(dtype_string);
        std::vector<int64_t> shape = parse_shape(item.at("shape"));

        Tensor tensor(shape, dtype, options.device);

        std::filesystem::path bin_path = dir_path / filename;
        std::vector<uint8_t> buffer = read_binary_file(bin_path);

        if (buffer.size() != tensor.nbytes()) {
            throw std::runtime_error(
                "Weight binary size mismatch\n"
                "  name: " + name + "\n" +
                "  file: " + std::filesystem::absolute(bin_path).string() + "\n" +
                "  expected bytes: " + std::to_string(tensor.nbytes()) + "\n" +
                "  actual bytes: " + std::to_string(buffer.size())
            );
        }

        tensor.copy_from_cpu(buffer.data(), buffer.size());

        weights.add(std::move(name), std::move(tensor));
    }

    return weights;
}

} // namespace lite_llm
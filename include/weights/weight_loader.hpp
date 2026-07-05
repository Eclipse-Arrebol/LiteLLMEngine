#pragma once

#include "core/device.hpp"
#include "core/dtype.hpp"
#include "weights/weight_map.hpp"

#include <string>

namespace lite_llm {

struct WeightLoaderOptions {
    Device device = Device::CPU;
    std::string index_filename = "weights_index.json";
};

WeightMap load_weight_map_from_directory(
    const std::string& directory,
    const WeightLoaderOptions& options
);

} // namespace lite_llm
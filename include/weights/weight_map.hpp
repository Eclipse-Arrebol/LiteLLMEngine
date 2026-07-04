#pragma once

#include "core/tensor.hpp"

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace lite_llm {

class WeightMap {
public:
    WeightMap() = default;

    WeightMap(const WeightMap&) = delete;
    WeightMap& operator=(const WeightMap&) = delete;

    WeightMap(WeightMap&&) noexcept = default;
    WeightMap& operator=(WeightMap&&) noexcept = default;

    void add(std::string name, Tensor tensor) {
        auto result = weights_.emplace(std::move(name), std::move(tensor));

        if (!result.second) {
            throw std::runtime_error("Duplicate weight name in WeightMap");
        }
    }

    bool contains(const std::string& name) const {
        return weights_.find(name) != weights_.end();
    }

    Tensor take(const std::string& name) {
        auto it = weights_.find(name);

        if (it == weights_.end()) {
            throw std::runtime_error("Weight not found: " + name);
        }

        Tensor tensor = std::move(it->second);
        weights_.erase(it);

        return tensor;
    }

    size_t size() const {
        return weights_.size();
    }

    bool empty() const {
        return weights_.empty();
    }

private:
    std::unordered_map<std::string, Tensor> weights_;
};

} // namespace lite_llm
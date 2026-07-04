#pragma once

#include "core/tensor.hpp"

#include <utility>

/*
    Layer              : 所有模型层的基础类
    UnaryLayer         : 单输入、单输出层，forward(input, output)
    WeightedUnaryLayer : 带一个主权重 weight_ 的单输入单输出层
 */
namespace lite_llm {

class Layer {
public:
    Layer() = default;
    virtual ~Layer() = default;

    Layer(const Layer&) = delete;
    Layer& operator=(const Layer&) = delete;

    Layer(Layer&&) noexcept = default;
    Layer& operator=(Layer&&) noexcept = default;

    virtual const char* name() const = 0;

    virtual bool initialized() const {
        return true;
    }
};

class UnaryLayer : public Layer {
public:
    UnaryLayer() = default;
    ~UnaryLayer() override = default;

    UnaryLayer(const UnaryLayer&) = delete;
    UnaryLayer& operator=(const UnaryLayer&) = delete;

    UnaryLayer(UnaryLayer&&) noexcept = default;
    UnaryLayer& operator=(UnaryLayer&&) noexcept = default;

    virtual void forward(const Tensor& input, Tensor& output) const = 0;
};

class WeightedUnaryLayer : public UnaryLayer {
public:
    WeightedUnaryLayer() = default;

    explicit WeightedUnaryLayer(Tensor weight)
        : weight_(std::move(weight)) {
    }

    ~WeightedUnaryLayer() override = default;

    WeightedUnaryLayer(const WeightedUnaryLayer&) = delete;
    WeightedUnaryLayer& operator=(const WeightedUnaryLayer&) = delete;

    WeightedUnaryLayer(WeightedUnaryLayer&&) noexcept = default;
    WeightedUnaryLayer& operator=(WeightedUnaryLayer&&) noexcept = default;

    bool initialized() const override {
        return !weight_.empty();
    }

    const Tensor& weight() const {
        return weight_;
    }

protected:
    Tensor weight_;
};

} // namespace lite_llm
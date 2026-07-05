#include "runtime/args.hpp"
#include "runtime/model_downloader.hpp"
#include "model/model_config.hpp"
#include "core/device.hpp"
#include "core/tensor.hpp"
#include "core/dtype.hpp"
#include "ops/add.hpp"

#include <exception>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    try {
        lite_llm::Args args = lite_llm::parse_args(argc, argv);
        
        std::string device_arg = args.device.value_or("auto");
        lite_llm::Device device = lite_llm::resolve_device(device_arg);
        std::cout << "Device: " << lite_llm::device_to_string(device) << std::endl;

        std::vector<float> host_input = {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f
        };

        lite_llm::Tensor tensor({2, 3}, lite_llm::DType::FP32, device);

        tensor.copy_from_cpu(
            host_input.data(),
            host_input.size() * sizeof(float)
        );

        std::vector<float> host_output(host_input.size(), 0.0f);

        tensor.copy_to_cpu(
            host_output.data(),
            host_output.size() * sizeof(float)
        );

        std::cout << "Tensor copy test:" << std::endl;

        for (float value : host_output) {
            std::cout << value << " ";
        }

        std::cout << std::endl;

        std::vector<float> host_a = {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f
        };

        std::vector<float> host_b = {
            10.0f, 20.0f, 30.0f,
            40.0f, 50.0f, 60.0f
        };

        lite_llm::Tensor a({2, 3}, lite_llm::DType::FP32, device);
        lite_llm::Tensor b({2, 3}, lite_llm::DType::FP32, device);
        lite_llm::Tensor out({2, 3}, lite_llm::DType::FP32, device);

        a.copy_from_cpu(host_a.data(), host_a.size() * sizeof(float));
        b.copy_from_cpu(host_b.data(), host_b.size() * sizeof(float));

        lite_llm::tensor_add(a, b, out);

        std::vector<float> host_out(host_a.size(), 0.0f);
        out.copy_to_cpu(host_out.data(), host_out.size() * sizeof(float));

        std::cout << "Add test:" << std::endl;

        for (float value : host_out) {
            std::cout << value << " ";
        }

        std::cout << std::endl;

        std::cout << "lite_llm config:\n";
        std::cout << "  Model:       " << args.model << "\n";
        std::cout << "  Prompt:      " << args.prompt << "\n";
        std::cout << "  Max tokens:  " << args.max_tokens << "\n";
        std::cout << "  Temperature: " << args.temperature << "\n";
        std::cout << "  Top-k:       " << args.top_k << "\n";
        std::cout << "  Top-p:       " << args.top_p << "\n";
        std::cout << "  Device:      " << device_arg << "\n";

        auto model_files = lite_llm::ensure_model_files(args.model);
        auto model_config = lite_llm::load_model_config(model_files.config_path);
        lite_llm::print_model_config(model_config);
        // TODO:
        // Engine engine(args);
        // engine.generate();

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n\n";
        lite_llm::print_usage(argv[0]);
        return 1;
    }



}
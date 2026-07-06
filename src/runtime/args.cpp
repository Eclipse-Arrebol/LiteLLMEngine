#include "runtime/args.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>


namespace lite_llm {

namespace {

/**
 * @brief 这个函数主要是判断str中是否存在字串为prefix
 * 
 * @param str 
 * @param prefix 
 * @return true 
 * @return false 
 */
bool starts_with(const std::string& str, const std::string& prefix) {
    return str.rfind(prefix, 0) == 0;
}


/**
 * @brief 这个函数实现了两种解析方式，分别是 --model=xxx，或者是--model xxx
 * 
 * @param index 
 * @param argc 
 * @param argv 
 * @param arg 
 * @return std::string 
 */
std::string get_arg_value(
    int& index,
    int argc,
    char** argv,
    const std::string& arg
) {
    auto equal_pos = arg.find('=');

    if (equal_pos != std::string::npos) {
        return arg.substr(equal_pos + 1);
    }

    if (index + 1 >= argc) {
        throw std::runtime_error("Missing value for argument: " + arg);
    }

    return argv[++index];
}

} // namespace

void print_usage(const char* program) {
    std::cout
        << "Usage:\n"
        << "  " << program << " --model <model_path_or_name> [options]\n\n"
        << "Options:\n"
        << "  --model <str>          HuggingFace model name or local path\n"
        << "  --prompt <str>         Input prompt\n"
        << "  --input-ids <str>      Comma-separated token ids, e.g. \"1,2,3\"\n"
        << "  --max-tokens <int>     Maximum generated tokens, default: 128\n"
        << "  --temperature <float>  Sampling temperature, default: 0.7\n"
        << "  --top-k <int>          Top-k sampling, default: 50\n"
        << "  --top-p <float>        Top-p sampling, default: 0.9\n"
        << "  --device <str>         Device, e.g. cpu, cuda, cuda:0\n"
        << "  -h, --help             Show this help message\n"
        << "  --eos-token-id <int>  Stop generation when this token id is generated\n";
}

/**
 * @brief 解析输入参数
 * 
 * @param argc 
 * @param argv 
 * @return Args 
 */
Args parse_args(int argc, char** argv) {
    Args args;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (arg == "--model" || starts_with(arg, "--model=")) {
            args.model = get_arg_value(i, argc, argv, arg);
        } else if (arg == "--prompt" || starts_with(arg, "--prompt=")) {
            args.prompt = get_arg_value(i, argc, argv, arg);
        } else if(arg == "--input-ids" || starts_with(arg, "--input-ids=")){
            args.input_ids = get_arg_value(i, argc, argv, arg);
        } else if (arg == "--max-tokens" || starts_with(arg, "--max-tokens=")) {
            args.max_tokens = std::stoi(get_arg_value(i, argc, argv, arg));
        } else if (arg == "--eos-token-id" || starts_with(arg, "--eos-token-id=")) {
            args.eos_token_id = std::stoi(get_arg_value(i, argc, argv, arg));
        }else if (arg == "--temperature" || starts_with(arg, "--temperature=")) {
            args.temperature = std::stof(get_arg_value(i, argc, argv, arg));
        } else if (arg == "--top-k" || starts_with(arg, "--top-k=")) {
            args.top_k = std::stoi(get_arg_value(i, argc, argv, arg));
        } else if (arg == "--top-p" || starts_with(arg, "--top-p=")) {
            args.top_p = std::stof(get_arg_value(i, argc, argv, arg));
        } else if (arg == "--device" || starts_with(arg, "--device=")) {
            args.device = get_arg_value(i, argc, argv, arg);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (args.model.empty()) {
        throw std::runtime_error("Missing required argument: --model");
    }

    if (args.max_tokens <= 0) {
        throw std::runtime_error("--max-tokens must be greater than 0");
    }

    if (args.temperature < 0.0f) {
        throw std::runtime_error("--temperature must be >= 0");
    }

    if (args.top_k < 0) {
        throw std::runtime_error("--top-k must be >= 0");
    }

    if (args.top_p <= 0.0f || args.top_p > 1.0f) {
        throw std::runtime_error("--top-p must be in range (0, 1]");
    }

    if (args.eos_token_id.has_value() && args.eos_token_id.value() < 0) {
        throw std::runtime_error("--eos-token-id must be >= 0");
    }

    return args;
}

} // namespace lite_llm
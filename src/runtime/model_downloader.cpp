#include "runtime/model_downloader.hpp"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

namespace lite_llm {

namespace fs = std::filesystem;

namespace {

/**
 * @brief 将/换成_,主要还是确保目录的正确hf中经常会有/
 * 
 * @param model 
 * @return std::string 
 */
std::string sanitize_model_name(const std::string& model) {
    std::string result = model;

    for (char& c : result) {
        if (c == '/') {
            c = '_';
        }
    }

    return result;
}

int run_command(const std::vector<std::string>& args) {
    std::vector<char*> argv;
    argv.reserve(args.size() + 1);

    for (const auto& arg : args) {
        argv.push_back(const_cast<char*>(arg.c_str()));
    }

    argv.push_back(nullptr);

    pid_t pid = fork();

    if (pid < 0) {
        throw std::runtime_error("Failed to fork process");
    }

    if (pid == 0) {
        execvp(argv[0], argv.data());
        _exit(127);
    }

    int status = 0;
    waitpid(pid, &status, 0);

    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }

    return -1;
}

/**
 * @brief 下载文件必要文件，require为true才会报错
 * 
 * @param model 
 * @param filename 
 * @param output_dir 
 * @param required 
 */
void download_file(
    const std::string& model,
    const std::string& filename,
    const std::string& output_dir,
    bool required
) {
    std::vector<std::string> cmd = {
        "hf",
        "download",
        model,
        filename,
        "--local-dir",
        output_dir
    };

    int ret = run_command(cmd);

    if (ret != 0 && required) {
        throw std::runtime_error("Failed to download required file: " + filename);
    }

    if (ret != 0) {
        std::cerr << "Warning: optional file not found: " << filename << "\n";
    }
}

} // namespace

ModelFiles ensure_model_files(const std::string& model) {
    fs::path model_dir;

    if (fs::exists(model)) {
        model_dir = fs::path(model);
    } else {
        model_dir = fs::path("models") / sanitize_model_name(model);

        fs::create_directories(model_dir);

        std::cout << "Downloading model metadata from HuggingFace: "
                  << model << "\n";

        download_file(model, "config.json", model_dir.string(), true);
        download_file(model, "tokenizer.json", model_dir.string(), true);
        download_file(model, "tokenizer_config.json", model_dir.string(), true);

        download_file(model, "generation_config.json", model_dir.string(), false);
        download_file(model, "special_tokens_map.json", model_dir.string(), false);
    }

    ModelFiles files;
    files.model_dir = model_dir.string();
    files.config_path = (model_dir / "config.json").string();
    files.tokenizer_path = (model_dir / "tokenizer.json").string();
    files.tokenizer_config_path = (model_dir / "tokenizer_config.json").string();
    files.generation_config_path = (model_dir / "generation_config.json").string();

    if (!fs::exists(files.config_path)) {
        throw std::runtime_error("Missing config.json in model directory: " + files.model_dir);
    }

    return files;
}

} // namespace lite_llm
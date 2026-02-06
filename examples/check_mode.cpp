#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "core/config_loader.h"
#include "runtime/iruntime.h"
#include "runtime/runtime_setup.h"

namespace {

std::vector<int> parse_tokens(const std::string& text) {
    std::vector<int> tokens;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        tokens.push_back(std::stoi(item));
    }
    return tokens;
}

int getenv_int(const char* name, int def) {
    const char* val = std::getenv(name);
    if (!val || !*val) return def;
    try {
        return std::stoi(val);
    } catch (...) {
        return def;
    }
}

std::string getenv_str(const char* name, const std::string& def) {
    const char* val = std::getenv(name);
    if (!val || !*val) return def;
    return std::string(val);
}

bool write_tokens(const std::filesystem::path& path, const std::vector<int>& tokens) {
    std::ofstream out(path);
    if (!out.is_open()) return false;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) out << " ";
        out << tokens[i];
    }
    out << "\n";
    return true;
}

bool write_logits(const std::filesystem::path& path, const std::vector<float>& logits) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) return false;
    if (!logits.empty()) {
        out.write(reinterpret_cast<const char*>(logits.data()),
                  static_cast<std::streamsize>(logits.size() * sizeof(float)));
    }
    return true;
}

bool write_meta(const std::filesystem::path& path,
                const std::string& model_path,
                int vocab_size,
                int hidden_size,
                int num_layers,
                int token_count) {
    std::ofstream out(path);
    if (!out.is_open()) return false;
    out << "{\n";
    out << "  \"model_path\": \"" << model_path << "\",\n";
    out << "  \"prompt\": \"\",\n";
    out << "  \"vocab_size\": " << vocab_size << ",\n";
    out << "  \"hidden_size\": " << hidden_size << ",\n";
    out << "  \"num_layers\": " << num_layers << ",\n";
    out << "  \"token_count\": " << token_count << "\n";
    out << "}\n";
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const char* model_path = nullptr;
    if (argc > 1) {
        model_path = argv[1];
    } else {
        model_path = std::getenv("MODEL_PATH");
    }

    if (!model_path || !*model_path) {
        std::cerr << "Usage: " << argv[0] << " /path/to/model\n";
        std::cerr << "Or set MODEL_PATH\n";
        return 1;
    }

    std::string config_path = std::string(model_path) + "/config.json";
    ember::ModelConfig model_config;
    try {
        model_config = ember::parse_model_config(config_path);
    } catch (const std::exception& ex) {
        std::cerr << "Failed to parse config.json: " << ex.what() << "\n";
        return 1;
    }

    auto runtime = ember::RuntimeFactory::create_cuda();
    if (!runtime || !runtime->available()) {
        std::cerr << "CUDA runtime not available\n";
        return 1;
    }

    ember::RuntimeConfig runtime_config;
    runtime_config.max_ctx_len = getenv_int("MAX_CTX_LEN", 64);
    runtime_config.batch_size = 1;
    runtime_config.check_correctness = true;
    runtime_config.dump_layer = getenv_int("DUMP_LAYER", 2);
    runtime_config.dump_dir = getenv_str("DUMP_DIR", "debug/example_check");

    ember::RuntimeSetup setup;
    ember::DeviceMap device_map = ember::DeviceMap::single_device(model_config.num_layers, 0);

    ember::Error err = ember::load_runtime(*runtime, model_path, model_config, device_map, setup);
    if (err) {
        std::cerr << "load failed: " << err.message() << "\n";
        ember::shutdown_runtime(*runtime, setup);
        return 1;
    }

    err = ember::init_session_and_kv(*runtime, model_config, runtime_config, setup);
    if (err) {
        std::cerr << "kv cache alloc failed: " << err.message() << "\n";
        ember::shutdown_runtime(*runtime, setup);
        return 1;
    }

    std::vector<int> tokens;
    const char* token_env = std::getenv("PROMPT_TOKENS");
    if (token_env && *token_env) {
        tokens = parse_tokens(token_env);
    }
    if (tokens.empty()) {
        tokens = {1, 2, 3};
    }
    int max_token = std::max(0, static_cast<int>(model_config.vocab_size - 1));
    for (int& t : tokens) {
        if (t < 0) t = 0;
        if (t > max_token) t = max_token;
    }

    ember::Session& session = setup.session;
    std::vector<float> logits;
    err = runtime->prefill_with_logits(tokens, session, logits);
    if (err) {
        std::cerr << "prefill failed: " << err.message() << "\n";
        ember::shutdown_runtime(*runtime, setup);
        return 1;
    }

    std::filesystem::path out_dir(runtime_config.dump_dir);
    std::error_code ec;
    std::filesystem::create_directories(out_dir, ec);
    if (ec) {
        std::cerr << "Failed to create dump dir: " << out_dir.string() << "\n";
        ember::shutdown_runtime(*runtime, setup);
        return 1;
    }

    bool ok = true;
    ok &= write_tokens(out_dir / "tokens.txt", tokens);
    ok &= write_logits(out_dir / "logits.bin", logits);
    ok &= write_meta(out_dir / "meta.json", model_path,
                     static_cast<int>(model_config.vocab_size),
                     static_cast<int>(model_config.hidden_size),
                     static_cast<int>(model_config.num_layers),
                     static_cast<int>(tokens.size()));

    if (!ok) {
        std::cerr << "Failed to write debug outputs in " << out_dir.string() << "\n";
        ember::shutdown_runtime(*runtime, setup);
        return 1;
    }

    std::cout << "Saved check outputs to " << out_dir.string() << "\n";
    ember::shutdown_runtime(*runtime, setup);
    return 0;
}

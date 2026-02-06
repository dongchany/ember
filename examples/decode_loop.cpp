#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "backends/cuda/cuda_runtime.h"
#include "core/config_loader.h"
#include "core/sampler.h"
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

float getenv_float(const char* name, float def) {
    const char* val = std::getenv(name);
    if (!val || !*val) return def;
    try {
        return std::stof(val);
    } catch (...) {
        return def;
    }
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
    runtime_config.temperature = getenv_float("TEMPERATURE", 0.0f);
    runtime_config.top_p = getenv_float("TOP_P", 1.0f);
    runtime_config.top_k = getenv_int("TOP_K", 1);

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
        tokens = {1};
    }

    ember::Sampler sampler;
    if (const char* seed_env = std::getenv("SEED")) {
        try {
            sampler.set_seed(static_cast<uint64_t>(std::stoull(seed_env)));
        } catch (...) {
        }
    }

    ember::Session& session = setup.session;
    std::vector<float> logits;
    err = runtime->prefill_with_logits(tokens, session, logits);
    if (err) {
        std::cerr << "prefill failed: " << err.message() << "\n";
        ember::shutdown_runtime(*runtime, setup);
        return 1;
    }

    int max_new_tokens = getenv_int("MAX_NEW_TOKENS", 8);
    std::vector<int> history = tokens;
    int last_token = sampler.sample(logits, runtime_config, history);
    history.push_back(last_token);

    for (int i = 0; i < max_new_tokens - 1; ++i) {
        std::vector<float> step_logits;
        err = runtime->decode(last_token, session, step_logits);
        if (err) {
            std::cerr << "decode failed: " << err.message() << "\n";
            ember::shutdown_runtime(*runtime, setup);
            return 1;
        }
        last_token = sampler.sample(step_logits, runtime_config, history);
        history.push_back(last_token);
    }

    std::cout << "Generated token ids:";
    for (int tok : history) {
        std::cout << " " << tok;
    }
    std::cout << "\n";

    ember::shutdown_runtime(*runtime, setup);
    return 0;
}

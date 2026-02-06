#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "core/config_loader.h"
#include "runtime/runtime_setup.h"
#include "runtime/iruntime.h"
#include "backends/cuda/cuda_runtime.h"

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
    runtime_config.max_ctx_len = 32;
    runtime_config.batch_size = 1;

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

    ember::Session& session = setup.session;
    std::vector<int> tokens = {1};
    std::vector<float> logits;
    err = runtime->prefill_with_logits(tokens, session, logits);
    if (err) {
        std::cerr << "prefill failed: " << err.message() << "\n";
        ember::shutdown_runtime(*runtime, setup);
        return 1;
    }

    if (logits.empty()) {
        std::cerr << "logits empty\n";
        ember::shutdown_runtime(*runtime, setup);
        return 1;
    }

    auto it = std::max_element(logits.begin(), logits.end());
    int top_id = static_cast<int>(std::distance(logits.begin(), it));
    std::cout << "Top1 token id: " << top_id << " logit=" << *it << "\n";

    ember::shutdown_runtime(*runtime, setup);
    return 0;
}

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "backends/cuda/cuda_runtime.h"
#include "core/config_loader.h"
#include "runtime/runtime_setup.h"

namespace {

std::vector<int> make_tokens(int64_t vocab_size, int n, int seed) {
    std::vector<int> tokens;
    tokens.reserve(static_cast<size_t>(n));
    if (vocab_size <= 0) return {0};
    int max_token = static_cast<int>(vocab_size - 1);
    for (int i = 0; i < n; ++i) {
        int t = (seed + i * 131) % (max_token + 1);
        if (t < 0) t = 0;
        tokens.push_back(t);
    }
    if (tokens.empty()) tokens.push_back(0);
    return tokens;
}

}  // namespace

int main() {
    const char* model_path = std::getenv("MODEL_PATH");
    if (!model_path || !*model_path) {
        std::cout << "[skip] MODEL_PATH not set\n";
        return 0;
    }

    auto runtime = ember::RuntimeFactory::create_cuda();
    if (!runtime || !runtime->available()) {
        std::cout << "[skip] No CUDA device available\n";
        return 0;
    }

    ember::ModelConfig config;
    try {
        config = ember::parse_model_config(std::string(model_path) + "/config.json");
    } catch (const std::exception& ex) {
        std::cerr << "[FAIL] parse_model_config failed: " << ex.what() << "\n";
        return 1;
    }

    ember::RuntimeConfig runtime_config;
    runtime_config.max_ctx_len = 64;
    runtime_config.batch_size = 4;
    runtime_config.device_ids = {0};
    runtime_config.kv_cache_dtype = ember::dtype_from_string(config.torch_dtype);
    if (runtime_config.kv_cache_dtype == ember::DType::UNKNOWN) runtime_config.kv_cache_dtype = ember::DType::F16;

    ember::DeviceMap device_map = ember::DeviceMap::single_device(config.num_layers, 0);

    ember::RuntimeSetup setup;
    ember::Error err = ember::load_runtime(*runtime, model_path, config, device_map, setup);
    if (err) {
        std::cerr << "[FAIL] load_runtime failed: " << err.to_string() << "\n";
        ember::shutdown_runtime(*runtime, setup);
        return 1;
    }
    err = ember::init_session_and_kv(*runtime, config, runtime_config, setup);
    if (err) {
        std::cerr << "[FAIL] init_session_and_kv failed: " << err.to_string() << "\n";
        ember::shutdown_runtime(*runtime, setup);
        return 1;
    }

    auto* cuda_rt = dynamic_cast<ember::cuda::CudaRuntime*>(runtime.get());
    if (!cuda_rt) {
        std::cerr << "[FAIL] expected CUDA runtime\n";
        ember::shutdown_runtime(*runtime, setup);
        return 1;
    }

    // Prefill three slots with different prompt lengths; keep one slot inactive.
    const int slot0 = 0;
    const int slot1 = 1;
    const int slot2 = 2;
    const int slot3 = 3;

    std::vector<int> p0 = make_tokens(config.vocab_size, 4, 17);
    std::vector<int> p1 = make_tokens(config.vocab_size, 8, 23);
    std::vector<int> p3 = make_tokens(config.vocab_size, 6, 31);

    err = cuda_rt->prefill_into_slot(p0, slot0, setup.session, nullptr);
    if (err) {
        std::cerr << "[FAIL] prefill_into_slot(slot0) failed: " << err.to_string() << "\n";
        ember::shutdown_runtime(*runtime, setup);
        return 1;
    }
    err = cuda_rt->prefill_into_slot(p1, slot1, setup.session, nullptr);
    if (err) {
        std::cerr << "[FAIL] prefill_into_slot(slot1) failed: " << err.to_string() << "\n";
        ember::shutdown_runtime(*runtime, setup);
        return 1;
    }
    err = cuda_rt->prefill_into_slot(p3, slot3, setup.session, nullptr);
    if (err) {
        std::cerr << "[FAIL] prefill_into_slot(slot3) failed: " << err.to_string() << "\n";
        ember::shutdown_runtime(*runtime, setup);
        return 1;
    }
    setup.session.set_inactive(slot2);

    const int pos0 = setup.session.cur_pos(slot0);
    const int pos1 = setup.session.cur_pos(slot1);
    const int pos3 = setup.session.cur_pos(slot3);

    std::vector<int> last_tokens = {p0.back(), p1.back(), 0, p3.back()};
    std::vector<int> next_tokens;
    err = cuda_rt->decode_batch_greedy(last_tokens, setup.session, next_tokens);
    if (err) {
        std::cerr << "[FAIL] decode_batch_greedy failed: " << err.to_string() << "\n";
        ember::shutdown_runtime(*runtime, setup);
        return 1;
    }
    if (static_cast<int>(next_tokens.size()) != runtime_config.batch_size) {
        std::cerr << "[FAIL] unexpected next_tokens size\n";
        ember::shutdown_runtime(*runtime, setup);
        return 1;
    }

    if (setup.session.cur_pos(slot0) != pos0 + 1) {
        std::cerr << "[FAIL] slot0 cur_pos not advanced\n";
        ember::shutdown_runtime(*runtime, setup);
        return 1;
    }
    if (setup.session.cur_pos(slot1) != pos1 + 1) {
        std::cerr << "[FAIL] slot1 cur_pos not advanced\n";
        ember::shutdown_runtime(*runtime, setup);
        return 1;
    }
    if (setup.session.cur_pos(slot3) != pos3 + 1) {
        std::cerr << "[FAIL] slot3 cur_pos not advanced\n";
        ember::shutdown_runtime(*runtime, setup);
        return 1;
    }
    if (setup.session.cur_pos(slot2) != -1) {
        std::cerr << "[FAIL] slot2 (inactive) cur_pos changed\n";
        ember::shutdown_runtime(*runtime, setup);
        return 1;
    }

    std::cout << "[ok] varpos batch decode updates per-slot positions\n";
    ember::shutdown_runtime(*runtime, setup);
    return 0;
}


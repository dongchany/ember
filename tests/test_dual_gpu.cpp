#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "backends/cuda/cuda_runtime.h"
#include "core/config_loader.h"
#include "runtime/runtime_setup.h"

namespace {

void fail(const std::string& msg) {
    std::cerr << "[FAIL] " << msg << "\n";
}

bool has_nonfinite(const std::vector<float>& values, size_t& idx) {
    for (size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i])) {
            idx = i;
            return true;
        }
    }
    return false;
}

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        m = std::max(m, std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    }
    return m;
}

std::vector<int> make_tokens(int64_t vocab_size, int n) {
    std::vector<int> tokens;
    tokens.reserve(static_cast<size_t>(n));
    if (vocab_size <= 0) return {0};
    int max_token = static_cast<int>(vocab_size - 1);
    for (int i = 0; i < n; ++i) {
        int t = 1 + (i * 131) % (max_token + 1);
        if (t < 0) t = 0;
        if (t > max_token) t = max_token;
        tokens.push_back(t);
    }
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
    if (ember::cuda::get_device_count() < 2) {
        std::cout << "[skip] Need 2 GPUs\n";
        return 0;
    }

    ember::ModelConfig config;
    try {
        config = ember::parse_model_config(std::string(model_path) + "/config.json");
    } catch (const std::exception& ex) {
        fail(std::string("parse_model_config failed: ") + ex.what());
        return 1;
    }

    ember::RuntimeConfig runtime_config;
    runtime_config.max_ctx_len = 64;
    runtime_config.batch_size = 1;

    ember::DeviceMap device_map;
    device_map.num_devices = 2;
    device_map.embedding_device = 0;
    device_map.lm_head_device = 1;
    device_map.layer_to_device.resize(static_cast<size_t>(config.num_layers));
    int boundary = static_cast<int>(config.num_layers / 2);
    for (int i = 0; i < config.num_layers; ++i) {
        device_map.layer_to_device[static_cast<size_t>(i)] = (i < boundary) ? 0 : 1;
    }

    ember::RuntimeSetup setup;
    ember::Error err = ember::load_runtime(*runtime, model_path, config, device_map, setup);
    if (err) {
        fail("load_runtime failed: " + err.to_string());
        ember::shutdown_runtime(*runtime, setup);
        return 1;
    }

    auto* cuda_rt = dynamic_cast<ember::cuda::CudaRuntime*>(runtime.get());
    if (!cuda_rt) {
        fail("expected CUDA runtime");
        ember::shutdown_runtime(*runtime, setup);
        return 1;
    }

    std::vector<int> prompt = make_tokens(config.vocab_size, 32);
    const int last = prompt.back();

    auto run_path = [&](bool use_chunked_overlap, std::vector<float>& out_logits) -> bool {
        setup.session.init(config, runtime_config);
        ember::Error e = runtime->allocate_kv_cache(setup.session);
        if (e) {
            fail("allocate_kv_cache failed: " + e.to_string());
            return false;
        }

        if (use_chunked_overlap) {
            e = cuda_rt->prefill_chunked_pipeline(prompt, setup.session, /*chunk_len=*/8, /*overlap=*/true, nullptr);
        } else {
            e = runtime->prefill(prompt, setup.session);
        }
        if (e) {
            fail(std::string("prefill failed: ") + e.to_string());
            runtime->free_kv_cache(setup.session);
            return false;
        }

        e = runtime->decode(last, setup.session, out_logits);
        if (e) {
            fail(std::string("decode failed: ") + e.to_string());
            runtime->free_kv_cache(setup.session);
            return false;
        }

        runtime->free_kv_cache(setup.session);
        return true;
    };

    std::vector<float> logits_naive;
    std::vector<float> logits_overlap;
    if (!run_path(false, logits_naive)) {
        ember::shutdown_runtime(*runtime, setup);
        return 1;
    }
    if (!run_path(true, logits_overlap)) {
        ember::shutdown_runtime(*runtime, setup);
        return 1;
    }

    if (logits_naive.size() != logits_overlap.size() ||
        logits_naive.size() != static_cast<size_t>(config.vocab_size)) {
        std::ostringstream oss;
        oss << "logits size mismatch: naive=" << logits_naive.size()
            << " overlap=" << logits_overlap.size()
            << " vocab=" << config.vocab_size;
        fail(oss.str());
        ember::shutdown_runtime(*runtime, setup);
        return 1;
    }

    size_t bad_idx = 0;
    if (has_nonfinite(logits_naive, bad_idx) || has_nonfinite(logits_overlap, bad_idx)) {
        fail("logits contains NaN/Inf");
        ember::shutdown_runtime(*runtime, setup);
        return 1;
    }

    double diff = max_abs_diff(logits_naive, logits_overlap);
    if (diff > 1e-3) {
        std::ostringstream oss;
        oss << "max abs diff too large: " << diff;
        fail(oss.str());
        ember::shutdown_runtime(*runtime, setup);
        return 1;
    }

    std::cout << "[ok] dual-gpu chunked overlap matches naive (max diff " << diff << ")\n";
    ember::shutdown_runtime(*runtime, setup);
    return 0;
}

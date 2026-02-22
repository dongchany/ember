#include <cmath>
#include <cstdlib>
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

bool validate_stage_profile(const ember::cuda::CudaRuntime::StageProfileMs& profile,
                            std::string& reason) {
    auto check = [&](const char* name, float value) {
        if (!std::isfinite(value) || value < 0.0f) {
            std::ostringstream oss;
            oss << name << " invalid: " << value;
            reason = oss.str();
            return false;
        }
        return true;
    };
    return check("embedding_ms", profile.embedding_ms) &&
           check("rmsnorm_ms", profile.rmsnorm_ms) &&
           check("attention_ms", profile.attention_ms) &&
           check("ffn_ms", profile.ffn_ms) &&
           check("p2p_ms", profile.p2p_ms) &&
           check("memcpy_h2d_ms", profile.memcpy_h2d_ms) &&
           check("memcpy_d2h_ms", profile.memcpy_d2h_ms) &&
           check("lm_head_ms", profile.lm_head_ms) &&
           check("total_ms", profile.total_ms);
}

std::vector<int> make_tokens(int64_t vocab_size) {
    std::vector<int> tokens = {1, 2, 3, 4};
    if (vocab_size <= 0) {
        return {0};
    }
    int max_token = static_cast<int>(vocab_size - 1);
    for (int& t : tokens) {
        if (t < 0) t = 0;
        if (t > max_token) t = max_token;
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
    auto* cuda_rt = dynamic_cast<ember::cuda::CudaRuntime*>(runtime.get());
    if (!cuda_rt) {
        fail("expected CUDA runtime implementation");
        return 1;
    }

    std::string config_path = std::string(model_path) + "/config.json";
    ember::ModelConfig config;
    try {
        config = ember::parse_model_config(config_path);
    } catch (const std::exception& ex) {
        fail(std::string("parse_model_config failed: ") + ex.what());
        return 1;
    }

    ember::RuntimeConfig runtime_config;
    runtime_config.max_ctx_len = 32;
    runtime_config.batch_size = 1;

    ember::RuntimeSetup setup;

    ember::DeviceMap device_map = ember::DeviceMap::single_device(config.num_layers, 0);

    auto cleanup = [&]() {
        ember::shutdown_runtime(*runtime, setup);
    };

    ember::Error err = ember::load_runtime(*runtime, model_path, config, device_map, setup);
    if (err) {
        fail("runtime.load failed: " + err.to_string());
        cleanup();
        return 1;
    }
    err = ember::init_session_and_kv(*runtime, config, runtime_config, setup);
    if (err) {
        fail("allocate_kv_cache failed: " + err.to_string());
        cleanup();
        return 1;
    }
    ember::Session& session = setup.session;
    cuda_rt->set_stage_profiling(true);

    std::vector<int> tokens = make_tokens(config.vocab_size);
    std::vector<float> logits;
    err = runtime->prefill_with_logits(tokens, session, logits);
    if (err) {
        fail("prefill_with_logits failed: " + err.to_string());
        cleanup();
        return 1;
    }
    {
        std::string reason;
        auto sp = cuda_rt->take_last_stage_profile_ms();
        if (!validate_stage_profile(sp, reason)) {
            fail("prefill stage profile invalid: " + reason);
            cleanup();
            return 1;
        }
        if (!(sp.total_ms > 0.0f)) {
            fail("prefill stage profile total_ms should be > 0");
            cleanup();
            return 1;
        }
    }

    if (logits.size() != static_cast<size_t>(config.vocab_size)) {
        std::ostringstream oss;
        oss << "logits size mismatch: got " << logits.size()
            << ", expected " << config.vocab_size;
        fail(oss.str());
        cleanup();
        return 1;
    }

    size_t bad_idx = 0;
    if (has_nonfinite(logits, bad_idx)) {
        std::ostringstream oss;
        oss << "logits contains NaN/Inf at index " << bad_idx;
        fail(oss.str());
        cleanup();
        return 1;
    }

    std::vector<float> logits2;
    err = runtime->decode(tokens.back(), session, logits2);
    if (err) {
        fail("decode failed: " + err.to_string());
        cleanup();
        return 1;
    }
    if (logits2.size() != static_cast<size_t>(config.vocab_size)) {
        std::ostringstream oss;
        oss << "decode logits size mismatch: got " << logits2.size()
            << ", expected " << config.vocab_size;
        fail(oss.str());
        cleanup();
        return 1;
    }
    if (has_nonfinite(logits2, bad_idx)) {
        std::ostringstream oss;
        oss << "decode logits contains NaN/Inf at index " << bad_idx;
        fail(oss.str());
        cleanup();
        return 1;
    }
    {
        std::string reason;
        auto sp = cuda_rt->take_last_stage_profile_ms();
        if (!validate_stage_profile(sp, reason)) {
            fail("decode stage profile invalid: " + reason);
            cleanup();
            return 1;
        }
        if (!(sp.total_ms > 0.0f)) {
            fail("decode stage profile total_ms should be > 0");
            cleanup();
            return 1;
        }
    }

    cleanup();
    std::cout << "[PASS] cuda_runtime smoke test\n";
    return 0;
}

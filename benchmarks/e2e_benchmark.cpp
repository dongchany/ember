#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "backends/cuda/cuda_runtime.h"
#include "core/config_loader.h"
#include "runtime/scheduler.h"
#include "runtime/runtime_setup.h"

namespace {

[[noreturn]] void die(const std::string& msg) {
    std::cerr << "error: " << msg << "\n";
    std::exit(1);
}

std::vector<int> split_ints(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) out.push_back(std::stoi(tok));
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_dir;
    std::vector<int> gpus = {0, 1};
    std::vector<int> split = {};
    int prompt_len = 1024;
    int gen_len = 100;
    int chunk_len = 128;
    int iters = 3;
    bool overlap = false;
    bool force_pipeline = false;
    int decode_batch = 1;
    bool phase_aware = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) die(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (arg == "-h" || arg == "--help") {
            std::cout
                << "Ember E2E Benchmark (prefill+decode)\n\n"
                << "Usage:\n"
                << "  " << argv[0] << " --model <dir> [options]\n\n"
                << "Options:\n"
                << "  --gpus 0,1          (default: 0,1)\n"
                << "  --split A,B         layer split for 2 GPUs (e.g. 20,20); default: even split\n"
                << "  --prompt-len N      (default: 1024)\n"
                << "  --gen-len N         (default: 100)\n"
                << "  --chunk-len N       prefill chunk length (default: 128)\n"
                << "  --iters N           (default: 3)\n"
                << "  --decode-batch N    decode batch size (default: 1)\n"
                << "  --phase-aware       run PhaseAwareScheduler for prefill\n"
                << "  --pipeline          force 2-GPU chunked prefill pipeline even when --no-overlap\n"
                << "  --no-pipeline       disable forced pipeline (default)\n"
                << "  --overlap           enable chunked prefill overlap\n"
                << "  --no-overlap        disable overlap (default)\n"
                << "\nOutput CSV columns:\n"
                << "  mode,prompt_len,gen_len,chunk_len,batch_size,ttft_ms,prefill_ms,decode_ms,decode_tok_s\n";
            return 0;
        } else if (arg == "--model") {
            model_dir = need("--model");
        } else if (arg == "--gpus") {
            gpus = split_ints(need("--gpus"));
        } else if (arg == "--split") {
            split = split_ints(need("--split"));
        } else if (arg == "--prompt-len") {
            prompt_len = std::stoi(need("--prompt-len"));
        } else if (arg == "--gen-len") {
            gen_len = std::stoi(need("--gen-len"));
        } else if (arg == "--chunk-len") {
            chunk_len = std::stoi(need("--chunk-len"));
        } else if (arg == "--iters") {
            iters = std::stoi(need("--iters"));
        } else if (arg == "--decode-batch") {
            decode_batch = std::stoi(need("--decode-batch"));
        } else if (arg == "--phase-aware") {
            phase_aware = true;
        } else if (arg == "--pipeline") {
            force_pipeline = true;
        } else if (arg == "--no-pipeline") {
            force_pipeline = false;
        } else if (arg == "--overlap") {
            overlap = true;
        } else if (arg == "--no-overlap") {
            overlap = false;
        } else {
            die("unknown argument: " + arg);
        }
    }

    if (model_dir.empty()) die("--model is required");
    if (prompt_len <= 0) die("--prompt-len must be > 0");
    if (gen_len < 0) die("--gen-len must be >= 0");
    if (chunk_len <= 0) die("--chunk-len must be > 0");
    if (iters <= 0) die("--iters must be > 0");
    if (gpus.empty()) die("--gpus is empty");
    if (decode_batch <= 0) die("--decode-batch must be > 0");

    ember::ModelConfig config;
    try {
        config = ember::parse_model_config(model_dir + "/config.json");
    } catch (const std::exception& ex) {
        die(std::string("parse_model_config failed: ") + ex.what());
    }

    auto runtime = ember::RuntimeFactory::create_cuda();
    if (!runtime || !runtime->available()) die("CUDA runtime not available");

    ember::DeviceMap device_map;
    if (gpus.size() == 1) {
        device_map = ember::DeviceMap::single_device(config.num_layers, gpus[0]);
    } else if (gpus.size() == 2) {
        if (!split.empty() && split.size() != 2) die("--split expects A,B");
        int a = split.empty() ? (static_cast<int>(config.num_layers) / 2) : split[0];
        int b = split.empty() ? (static_cast<int>(config.num_layers) - a) : split[1];
        if (a <= 0 || b <= 0 || a + b != config.num_layers) die("invalid --split");
        device_map.num_devices = 2;
        device_map.embedding_device = gpus[0];
        device_map.lm_head_device = gpus[1];
        device_map.layer_to_device.resize(static_cast<size_t>(config.num_layers));
        for (int i = 0; i < config.num_layers; ++i) {
            device_map.layer_to_device[static_cast<size_t>(i)] = (i < a) ? gpus[0] : gpus[1];
        }
    } else {
        die("benchmark currently supports 1 or 2 GPUs");
    }

    ember::RuntimeConfig runtime_config;
    runtime_config.max_ctx_len = prompt_len + gen_len + 8;
    runtime_config.batch_size = decode_batch;
    runtime_config.device_ids = gpus;
    runtime_config.kv_cache_dtype = ember::dtype_from_string(config.torch_dtype);
    if (runtime_config.kv_cache_dtype == ember::DType::UNKNOWN) runtime_config.kv_cache_dtype = ember::DType::F16;

    ember::RuntimeSetup setup;
    ember::Error err = ember::load_runtime(*runtime, model_dir, config, device_map, setup);
    if (err) die("load_runtime failed: " + err.to_string());
    err = ember::init_session_and_kv(*runtime, config, runtime_config, setup);
    if (err) die("init_session_and_kv failed: " + err.to_string());

    auto* cuda_rt = dynamic_cast<ember::cuda::CudaRuntime*>(runtime.get());
    if (!cuda_rt) die("expected CUDA runtime");

    ember::PhaseAwareScheduler scheduler(
        *runtime,
        ember::PhaseAwareSchedulerConfig{
            .prefill_chunk_len = chunk_len,
            .prefill_overlap = overlap,
            .decode_batch_size = decode_batch,
        }
    );

    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> dist(0, static_cast<int>(config.vocab_size - 1));

    double prefill_ms_sum = 0.0;
    double decode_ms_sum = 0.0;
    double ttft_ms_sum = 0.0;
    const bool can_chunked_pipeline = (gpus.size() == 2 && decode_batch == 1);
    const bool use_chunked_pipeline = (can_chunked_pipeline && (force_pipeline || overlap));

    for (int it = 0; it < iters; ++it) {
        setup.session.reset();
        std::vector<int> prompt(static_cast<size_t>(prompt_len));
        for (int& t : prompt) t = dist(rng);

        auto t0 = std::chrono::high_resolution_clock::now();
        if (decode_batch == 1) {
            if (phase_aware) {
                err = scheduler.prefill(prompt, setup.session);
            } else if (use_chunked_pipeline) {
                err = cuda_rt->prefill_chunked_pipeline(prompt, setup.session, chunk_len, /*overlap=*/overlap, nullptr);
            } else {
                err = runtime->prefill(prompt, setup.session);
            }
        } else {
            // For batch experiments, replicate prompt into a flat [batch, seq] buffer.
            std::vector<int> flat(static_cast<size_t>(decode_batch) * static_cast<size_t>(prompt_len));
            for (int b = 0; b < decode_batch; ++b) {
                std::copy(prompt.begin(), prompt.end(), flat.begin() + static_cast<size_t>(b) * static_cast<size_t>(prompt_len));
            }
            err = cuda_rt->prefill_batch_flat(flat, prompt_len, setup.session);
        }
        if (err) die("prefill failed: " + err.to_string());
        auto t1 = std::chrono::high_resolution_clock::now();
        prefill_ms_sum += std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (gen_len > 0) {
            auto d0 = std::chrono::high_resolution_clock::now();
            if (decode_batch == 1) {
                int last = prompt.back();
                for (int i = 0; i < gen_len; ++i) {
                    err = cuda_rt->decode_to_device(last, setup.session);
                    if (err) die("decode failed: " + err.to_string());
                    last = (last + 1) % static_cast<int>(config.vocab_size);
                    if (i == 0) {
                        for (int dev : gpus) {
                            ember::Error se = ember::cuda::cuda_sync(dev);
                            if (se) die("cuda_sync failed: " + se.to_string());
                        }
                        auto tt = std::chrono::high_resolution_clock::now();
                        ttft_ms_sum += std::chrono::duration<double, std::milli>(tt - t0).count();
                    }
                }
            } else {
                std::vector<int> last_tokens(static_cast<size_t>(decode_batch), prompt.back());
                for (int i = 0; i < gen_len; ++i) {
                    err = cuda_rt->decode_batch_to_device(last_tokens, setup.session);
                    if (err) die("decode failed: " + err.to_string());
                    for (int& t : last_tokens) t = (t + 1) % static_cast<int>(config.vocab_size);
                }
            }
            for (int dev : gpus) {
                ember::Error se = ember::cuda::cuda_sync(dev);
                if (se) die("cuda_sync failed: " + se.to_string());
            }
            auto d1 = std::chrono::high_resolution_clock::now();
            decode_ms_sum += std::chrono::duration<double, std::milli>(d1 - d0).count();
            if (decode_batch != 1) {
                // TTFT is defined for single-request decode in this benchmark.
                ttft_ms_sum += 0.0;
            }
        }
    }

    const double ttft_ms = (gen_len > 0 && decode_batch == 1) ? (ttft_ms_sum / static_cast<double>(iters)) : 0.0;
    const double prefill_ms = prefill_ms_sum / static_cast<double>(iters);
    const double decode_ms = decode_ms_sum / static_cast<double>(iters);
    const double decode_tokens = static_cast<double>(gen_len) * static_cast<double>(decode_batch);
    const double decode_tok_s = (gen_len > 0 && decode_ms > 0.0) ? (decode_tokens * 1000.0 / decode_ms) : 0.0;

    std::cout << (overlap ? "overlap" : "no_overlap") << ","
              << prompt_len << "," << gen_len << "," << chunk_len << ","
              << decode_batch << ","
              << std::fixed << std::setprecision(3) << ttft_ms << ","
              << std::fixed << std::setprecision(3) << prefill_ms << ","
              << std::fixed << std::setprecision(3) << decode_ms << ","
              << std::fixed << std::setprecision(3) << decode_tok_s << "\n";

    ember::shutdown_runtime(*runtime, setup);
    return 0;
}

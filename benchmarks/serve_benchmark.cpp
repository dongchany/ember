#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "core/config_loader.h"
#include "runtime/runtime_setup.h"
#include "runtime/scheduler.h"

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
    int batch_size = 8;
    int num_reqs = 32;
    int prompt_len = 1024;
    int gen_len = 64;
    bool vary_gen = true;
    int seed = 1234;
    int prefill_chunk_len = 128;
    bool prefill_overlap = true;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) die(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (arg == "-h" || arg == "--help") {
            std::cout
                << "Ember Serve Benchmark (continuous batching simulation)\n\n"
                << "Usage:\n"
                << "  " << argv[0] << " --model <dir> [options]\n\n"
                << "Options:\n"
                << "  --gpus 0,1          (default: 0,1)\n"
                << "  --split A,B         layer split for 2 GPUs (e.g. 20,20); default: even split\n"
                << "  --batch-size N      max in-flight requests (default: 8)\n"
                << "  --num-req N         total requests to simulate (default: 32)\n"
                << "  --prompt-len N      (default: 1024)\n"
                << "  --gen-len N         max new tokens per request (default: 64)\n"
                << "  --no-vary-gen       disable per-request gen length variation\n"
                << "  --prefill-chunk-len N   chunk length for prefill pipeline (default: 128)\n"
                << "  --no-prefill-overlap    disable prefill overlap (default: enabled)\n"
                << "  --seed N            RNG seed (default: 1234)\n"
                << "\nOutput CSV columns:\n"
                << "  mode,num_reqs,batch_size,prompt_len,gen_len,vary_gen,prefill_ms,decode_ms,gen_tokens,decode_tok_s\n";
            return 0;
        } else if (arg == "--model") {
            model_dir = need("--model");
        } else if (arg == "--gpus") {
            gpus = split_ints(need("--gpus"));
        } else if (arg == "--split") {
            split = split_ints(need("--split"));
        } else if (arg == "--batch-size") {
            batch_size = std::stoi(need("--batch-size"));
        } else if (arg == "--num-req") {
            num_reqs = std::stoi(need("--num-req"));
        } else if (arg == "--prompt-len") {
            prompt_len = std::stoi(need("--prompt-len"));
        } else if (arg == "--gen-len") {
            gen_len = std::stoi(need("--gen-len"));
        } else if (arg == "--no-vary-gen") {
            vary_gen = false;
        } else if (arg == "--prefill-chunk-len") {
            prefill_chunk_len = std::stoi(need("--prefill-chunk-len"));
        } else if (arg == "--no-prefill-overlap") {
            prefill_overlap = false;
        } else if (arg == "--seed") {
            seed = std::stoi(need("--seed"));
        } else {
            die("unknown argument: " + arg);
        }
    }

    if (model_dir.empty()) die("--model is required");
    if (gpus.empty()) die("--gpus is empty");
    if (batch_size <= 0) die("--batch-size must be > 0");
    if (num_reqs <= 0) die("--num-req must be > 0");
    if (prompt_len <= 0) die("--prompt-len must be > 0");
    if (gen_len < 0) die("--gen-len must be >= 0");

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
        die("serve benchmark currently supports 1 or 2 GPUs");
    }

    ember::RuntimeConfig runtime_config;
    runtime_config.max_ctx_len = prompt_len + gen_len + 8;
    runtime_config.batch_size = batch_size;
    runtime_config.device_ids = gpus;
    runtime_config.kv_cache_dtype = ember::dtype_from_string(config.torch_dtype);
    if (runtime_config.kv_cache_dtype == ember::DType::UNKNOWN) runtime_config.kv_cache_dtype = ember::DType::F16;

    ember::RuntimeSetup setup;
    ember::Error err = ember::load_runtime(*runtime, model_dir, config, device_map, setup);
    if (err) die("load_runtime failed: " + err.to_string());
    err = ember::init_session_and_kv(*runtime, config, runtime_config, setup);
    if (err) die("init_session_and_kv failed: " + err.to_string());

    ember::PhaseAwareBatchScheduler sched(
        *runtime,
        setup.session,
        ember::PhaseAwareBatchSchedulerConfig{
            .refill_on_step = true,
            .prefill_chunk_len = prefill_chunk_len,
            .prefill_overlap = prefill_overlap,
        }
    );

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, static_cast<int>(config.vocab_size - 1));

    int submitted = 0;
    int completed = 0;
    int64_t gen_tokens = 0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;

    auto make_prompt = [&]() -> std::vector<int> {
        std::vector<int> prompt(static_cast<size_t>(prompt_len));
        for (int& t : prompt) t = dist(rng);
        return prompt;
    };
    auto req_gen_len = [&](int req_idx) -> int {
        if (!vary_gen) return gen_len;
        if (gen_len <= 0) return 0;
        int d = req_idx % 4;
        int out = gen_len - d;
        return out > 0 ? out : 1;
    };

    auto try_submit_one = [&]() -> bool {
        if (submitted >= num_reqs) return false;
        std::vector<int> prompt = make_prompt();
        int req_len = req_gen_len(submitted);
        auto t0 = std::chrono::high_resolution_clock::now();
        ember::Result<int> r = sched.submit(prompt, req_len);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (!r.ok()) {
            if (r.error().code() == ember::ErrorCode::OUT_OF_MEMORY) {
                return false;  // no free slot
            }
            die("submit failed: " + r.error().to_string());
        }
        prefill_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        ++submitted;
        if (req_len == 0) ++completed;
        return true;
    };

    // Initial fill.
    while (try_submit_one()) {}

    // Decode loop.
    while (completed < num_reqs) {
        if (!sched.has_active()) {
            // No active slots; try to admit more.
            if (!try_submit_one()) break;
            continue;
        }

        auto d0 = std::chrono::high_resolution_clock::now();
        ember::Result<int> r = sched.step();
        auto d1 = std::chrono::high_resolution_clock::now();
        if (!r.ok()) die("step failed: " + r.error().to_string());
        decode_ms += std::chrono::duration<double, std::milli>(d1 - d0).count();
        gen_tokens += r.value();

        // Refill any freed slots.
        while (try_submit_one()) {}

        // Count completions by scanning (cheap).
        int active = sched.active_slots();
        completed = submitted - active;
    }

    double decode_tok_s = (decode_ms > 0.0) ? (static_cast<double>(gen_tokens) / (decode_ms / 1000.0)) : 0.0;

    std::cout << std::fixed << std::setprecision(3)
              << "phase_aware_batch,"
              << num_reqs << ","
              << batch_size << ","
              << prompt_len << ","
              << gen_len << ","
              << (vary_gen ? 1 : 0) << ","
              << prefill_ms << ","
              << decode_ms << ","
              << gen_tokens << ","
              << decode_tok_s
              << "\n";

    return 0;
}

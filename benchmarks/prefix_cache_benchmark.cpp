#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "backends/cuda/cuda_runtime.h"
#include "core/config_loader.h"
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

std::string join_with_plus(const std::vector<int>& v) {
    std::ostringstream oss;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) oss << "+";
        oss << v[i];
    }
    return oss.str();
}

double ms_since(std::chrono::high_resolution_clock::time_point t0,
                std::chrono::high_resolution_clock::time_point t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_dir;
    std::vector<int> gpus = {0, 1};
    std::vector<int> split = {};
    int prompt_len = 2048;
    int prefix_len = 1024;
    int num_docs = 100;
    int chunk_len = 512;
    int iters = 1;
    int warmup = 0;
    int seed = 1234;
    bool overlap = true;
    bool force_pipeline = false;
    std::string csv_path;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) die(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (arg == "-h" || arg == "--help") {
            std::cout
                << "Ember Prefix Cache Benchmark\n\n"
                << "Usage:\n"
                << "  " << argv[0] << " --model <dir> [options]\n\n"
                << "Options:\n"
                << "  --model <dir>       model directory\n"
                << "  --gpus LIST         e.g. 0 or 0,1 (default: 0,1)\n"
                << "  --split A,B         layer split for 2 GPUs (default: even)\n"
                << "  --prompt-len N      full prompt length (default: 2048)\n"
                << "  --prefix-len N      shared prefix length (default: 1024)\n"
                << "  --num-docs N        number of requests/docs (default: 100)\n"
                << "  --chunk-len N       prefill chunk length for pipeline (default: 512)\n"
                << "  --iters N           measured iterations (default: 1)\n"
                << "  --warmup N          warmup iterations (default: 0)\n"
                << "  --seed N            RNG seed (default: 1234)\n"
                << "  --overlap           use 2-GPU overlap (default: on)\n"
                << "  --no-overlap        disable overlap\n"
                << "  --pipeline          force chunked pipeline even when --no-overlap\n"
                << "  --no-pipeline       disable forced pipeline (default)\n"
                << "  --csv PATH          write CSV row (default: stdout)\n";
            return 0;
        } else if (arg == "--model") {
            model_dir = need("--model");
        } else if (arg == "--gpus") {
            gpus = split_ints(need("--gpus"));
        } else if (arg == "--split") {
            split = split_ints(need("--split"));
        } else if (arg == "--prompt-len") {
            prompt_len = std::stoi(need("--prompt-len"));
        } else if (arg == "--prefix-len") {
            prefix_len = std::stoi(need("--prefix-len"));
        } else if (arg == "--num-docs") {
            num_docs = std::stoi(need("--num-docs"));
        } else if (arg == "--chunk-len") {
            chunk_len = std::stoi(need("--chunk-len"));
        } else if (arg == "--iters") {
            iters = std::stoi(need("--iters"));
        } else if (arg == "--warmup") {
            warmup = std::stoi(need("--warmup"));
        } else if (arg == "--seed") {
            seed = std::stoi(need("--seed"));
        } else if (arg == "--overlap") {
            overlap = true;
        } else if (arg == "--no-overlap") {
            overlap = false;
        } else if (arg == "--pipeline") {
            force_pipeline = true;
        } else if (arg == "--no-pipeline") {
            force_pipeline = false;
        } else if (arg == "--csv") {
            csv_path = need("--csv");
        } else {
            die("unknown argument: " + arg);
        }
    }

    if (model_dir.empty()) die("--model is required");
    if (prompt_len <= 0) die("--prompt-len must be > 0");
    if (prefix_len < 0 || prefix_len > prompt_len) die("--prefix-len must satisfy 0 <= prefix_len <= prompt_len");
    if (num_docs <= 0) die("--num-docs must be > 0");
    if (chunk_len <= 0) die("--chunk-len must be > 0");
    if (iters <= 0) die("--iters must be > 0");
    if (warmup < 0) die("--warmup must be >= 0");
    if (gpus.empty()) die("--gpus is empty");
    if (!split.empty() && split.size() != 2) die("--split expects A,B");
    if (gpus.size() > 2) die("benchmark supports only 1 or 2 GPUs");

    const int suffix_len = prompt_len - prefix_len;

    ember::ModelConfig config;
    try {
        config = ember::parse_model_config(model_dir + "/config.json");
    } catch (const std::exception& ex) {
        die(std::string("parse_model_config failed: ") + ex.what());
    }

    auto runtime = ember::RuntimeFactory::create_cuda();
    if (!runtime || !runtime->available()) die("CUDA runtime not available");
    auto* cuda_rt = dynamic_cast<ember::cuda::CudaRuntime*>(runtime.get());
    if (!cuda_rt) die("expected CUDA runtime");

    ember::DeviceMap device_map;
    if (gpus.size() == 1) {
        device_map = ember::DeviceMap::single_device(config.num_layers, gpus[0]);
    } else {
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
    }

    ember::RuntimeConfig runtime_config;
    runtime_config.max_ctx_len = prompt_len + 8;
    runtime_config.batch_size = 1;
    runtime_config.device_ids = gpus;
    runtime_config.kv_cache_dtype = ember::dtype_from_string(config.torch_dtype);
    if (runtime_config.kv_cache_dtype == ember::DType::UNKNOWN) runtime_config.kv_cache_dtype = ember::DType::F16;

    ember::RuntimeSetup setup;
    ember::Error err = ember::load_runtime(*runtime, model_dir, config, device_map, setup);
    if (err) die("load_runtime failed: " + err.to_string());
    err = ember::init_session_and_kv(*runtime, config, runtime_config, setup);
    if (err) die("init_session_and_kv failed: " + err.to_string());

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, static_cast<int>(config.vocab_size - 1));

    std::vector<int> prefix_tokens(static_cast<size_t>(prefix_len));
    for (int& t : prefix_tokens) t = dist(rng);

    std::vector<std::vector<int>> suffix_tokens(static_cast<size_t>(num_docs),
                                                std::vector<int>(static_cast<size_t>(suffix_len)));
    for (auto& s : suffix_tokens) {
        for (int& t : s) t = dist(rng);
    }

    std::vector<std::vector<int>> full_prompts(static_cast<size_t>(num_docs),
                                               std::vector<int>(static_cast<size_t>(prompt_len)));
    for (int i = 0; i < num_docs; ++i) {
        if (prefix_len > 0) {
            std::copy(prefix_tokens.begin(), prefix_tokens.end(), full_prompts[static_cast<size_t>(i)].begin());
        }
        if (suffix_len > 0) {
            std::copy(suffix_tokens[static_cast<size_t>(i)].begin(),
                      suffix_tokens[static_cast<size_t>(i)].end(),
                      full_prompts[static_cast<size_t>(i)].begin() + prefix_len);
        }
    }

    const bool can_pipeline = (gpus.size() == 2);
    const bool use_pipeline = (can_pipeline && (force_pipeline || overlap));

    auto run_prefill = [&](const std::vector<int>& tokens) -> double {
        auto t0 = std::chrono::high_resolution_clock::now();
        if (use_pipeline) {
            err = cuda_rt->prefill_chunked_pipeline(tokens, setup.session, chunk_len, /*overlap=*/overlap, nullptr);
        } else {
            err = runtime->prefill(tokens, setup.session);
        }
        if (err) die("prefill failed: " + err.to_string());
        auto t1 = std::chrono::high_resolution_clock::now();
        return ms_since(t0, t1);
    };

    double no_cache_total_acc = 0.0;
    double cache_total_acc = 0.0;
    double cache_prefix_once_acc = 0.0;
    double cache_suffix_acc = 0.0;
    int measured = 0;

    for (int it = 0; it < warmup + iters; ++it) {
        double no_cache_total = 0.0;
        for (int i = 0; i < num_docs; ++i) {
            setup.session.set_cur_pos(0);
            no_cache_total += run_prefill(full_prompts[static_cast<size_t>(i)]);
        }

        setup.session.set_cur_pos(0);
        double prefix_once = 0.0;
        if (prefix_len > 0) {
            prefix_once = run_prefill(prefix_tokens);
        }
        const int prefix_pos = setup.session.cur_pos();

        double cache_suffix_total = 0.0;
        for (int i = 0; i < num_docs; ++i) {
            setup.session.set_cur_pos(prefix_pos);
            if (suffix_len > 0) {
                cache_suffix_total += run_prefill(suffix_tokens[static_cast<size_t>(i)]);
            }
        }
        const double cache_total = prefix_once + cache_suffix_total;

        if (it >= warmup) {
            no_cache_total_acc += no_cache_total;
            cache_total_acc += cache_total;
            cache_prefix_once_acc += prefix_once;
            cache_suffix_acc += cache_suffix_total;
            measured += 1;
        }
    }

    if (measured <= 0) die("no measured iterations");

    const double no_cache_total_ms = no_cache_total_acc / static_cast<double>(measured);
    const double with_cache_total_ms = cache_total_acc / static_cast<double>(measured);
    const double cache_prefix_once_ms = cache_prefix_once_acc / static_cast<double>(measured);
    const double cache_suffix_total_ms = cache_suffix_acc / static_cast<double>(measured);
    const double no_cache_per_doc_ms = no_cache_total_ms / static_cast<double>(num_docs);
    const double with_cache_per_doc_ms = with_cache_total_ms / static_cast<double>(num_docs);
    const double speedup_x = (with_cache_total_ms > 0.0) ? (no_cache_total_ms / with_cache_total_ms) : 0.0;
    const double savings_pct = (no_cache_total_ms > 0.0)
                                   ? ((no_cache_total_ms - with_cache_total_ms) / no_cache_total_ms * 100.0)
                                   : 0.0;
    const double theoretical_savings_pct =
        (prompt_len > 0)
            ? (static_cast<double>(prefix_len) * static_cast<double>(num_docs - 1) /
               (static_cast<double>(num_docs) * static_cast<double>(prompt_len)) * 100.0)
            : 0.0;

    std::ostream* out = &std::cout;
    std::ofstream f;
    if (!csv_path.empty()) {
        f.open(csv_path);
        if (!f.is_open()) die("failed to open output: " + csv_path);
        out = &f;
    }

    const std::string gpus_str = join_with_plus(gpus);
    std::string split_str;
    if (gpus.size() == 2) {
        int a = split.empty() ? (static_cast<int>(config.num_layers) / 2) : split[0];
        int b = split.empty() ? (static_cast<int>(config.num_layers) - a) : split[1];
        split_str = std::to_string(a) + "+" + std::to_string(b);
    } else {
        split_str = std::to_string(config.num_layers);
    }

    *out << "gpus,split,mode,prompt_len,prefix_len,suffix_len,num_docs,iters,warmup,"
         << "no_cache_total_ms,with_cache_total_ms,cache_prefix_once_ms,cache_suffix_total_ms,"
         << "no_cache_per_doc_ms,with_cache_per_doc_ms,speedup_x,savings_pct,theoretical_savings_pct\n";
    *out << gpus_str << "," << split_str << "," << (overlap ? "overlap" : "no_overlap") << ","
         << prompt_len << "," << prefix_len << "," << suffix_len << "," << num_docs << ","
         << iters << "," << warmup << ","
         << std::fixed << std::setprecision(3)
         << no_cache_total_ms << ","
         << with_cache_total_ms << ","
         << cache_prefix_once_ms << ","
         << cache_suffix_total_ms << ","
         << no_cache_per_doc_ms << ","
         << with_cache_per_doc_ms << ","
         << speedup_x << ","
         << savings_pct << ","
         << theoretical_savings_pct << "\n";

    ember::shutdown_runtime(*runtime, setup);
    return 0;
}

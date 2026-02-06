#include <algorithm>
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

std::string join_ints(const std::vector<int>& v) {
    std::ostringstream oss;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) oss << "+";
        oss << v[i];
    }
    return oss.str();
}

struct Accum {
    double wall_ms = 0.0;
    double embedding_ms = 0.0;
    double rmsnorm_ms = 0.0;
    double attention_ms = 0.0;
    double ffn_ms = 0.0;
    double p2p_ms = 0.0;
    double lm_head_ms = 0.0;
    double prof_total_ms = 0.0;
};

void add(Accum& a, double wall_ms, const ember::cuda::CudaRuntime::StageProfileMs& sp) {
    a.wall_ms += wall_ms;
    a.embedding_ms += sp.embedding_ms;
    a.rmsnorm_ms += sp.rmsnorm_ms;
    a.attention_ms += sp.attention_ms;
    a.ffn_ms += sp.ffn_ms;
    a.p2p_ms += sp.p2p_ms;
    a.lm_head_ms += sp.lm_head_ms;
    a.prof_total_ms += sp.total_ms;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_dir;
    std::vector<int> gpus = {0};
    std::vector<int> split = {};
    int prompt_len = 2048;
    int decode_steps = 256;
    int chunk_len = 512;
    int iters = 3;
    int warmup = 1;
    bool overlap = false;
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
                << "Ember Stage Breakdown Benchmark (prefill + decode stage sums)\n\n"
                << "Usage:\n"
                << "  " << argv[0] << " --model <dir> [options]\n\n"
                << "Options:\n"
                << "  --model <dir>         model directory\n"
                << "  --gpus LIST           e.g. 0 or 0,1 (default: 0)\n"
                << "  --split A,B           layer split for 2 GPUs (default: even)\n"
                << "  --prompt-len N        (default: 2048)\n"
                << "  --decode-steps N      (default: 256)\n"
                << "  --chunk-len N         prefill chunk length for pipeline (default: 512)\n"
                << "  --overlap             enable 2-GPU chunked prefill overlap\n"
                << "  --no-overlap          disable overlap (default)\n"
                << "  --pipeline            force 2-GPU chunked prefill pipeline even when --no-overlap\n"
                << "  --no-pipeline         disable forced pipeline (default)\n"
                << "  --iters N             (default: 3)\n"
                << "  --warmup N            (default: 1)\n"
                << "  --csv PATH            write CSV (default: stdout)\n";
            return 0;
        } else if (arg == "--model") {
            model_dir = need("--model");
        } else if (arg == "--gpus") {
            gpus = split_ints(need("--gpus"));
        } else if (arg == "--split") {
            split = split_ints(need("--split"));
        } else if (arg == "--prompt-len") {
            prompt_len = std::stoi(need("--prompt-len"));
        } else if (arg == "--decode-steps") {
            decode_steps = std::stoi(need("--decode-steps"));
        } else if (arg == "--chunk-len") {
            chunk_len = std::stoi(need("--chunk-len"));
        } else if (arg == "--iters") {
            iters = std::stoi(need("--iters"));
        } else if (arg == "--warmup") {
            warmup = std::stoi(need("--warmup"));
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
    if (decode_steps < 0) die("--decode-steps must be >= 0");
    if (chunk_len <= 0) die("--chunk-len must be > 0");
    if (iters <= 0) die("--iters must be > 0");
    if (warmup < 0) die("--warmup must be >= 0");
    if (gpus.empty()) die("--gpus is empty");
    if (!split.empty() && split.size() != 2) die("--split expects A,B");
    if (gpus.size() > 2) die("benchmark supports 1 or 2 GPUs");

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
    runtime_config.max_ctx_len = prompt_len + decode_steps + 8;
    runtime_config.batch_size = 1;
    runtime_config.device_ids = gpus;
    runtime_config.kv_cache_dtype = ember::dtype_from_string(config.torch_dtype);
    if (runtime_config.kv_cache_dtype == ember::DType::UNKNOWN) runtime_config.kv_cache_dtype = ember::DType::F16;

    ember::RuntimeSetup setup;
    ember::Error err = ember::load_runtime(*runtime, model_dir, config, device_map, setup);
    if (err) die("load_runtime failed: " + err.to_string());
    err = ember::init_session_and_kv(*runtime, config, runtime_config, setup);
    if (err) die("init_session_and_kv failed: " + err.to_string());

    std::ostream* out = &std::cout;
    std::ofstream f;
    if (!csv_path.empty()) {
        f.open(csv_path);
        if (!f.is_open()) die("failed to open output: " + csv_path);
        out = &f;
    }

    *out << "phase,mode,gpus,split,prompt_len,decode_steps,chunk_len,overlap,wall_ms,"
         << "embedding_ms,rmsnorm_ms,attention_ms,ffn_ms,p2p_ms,lm_head_ms,profile_total_ms\n";

    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> dist(0, static_cast<int>(config.vocab_size - 1));

    std::vector<int> prompt(static_cast<size_t>(prompt_len));
    for (int& t : prompt) t = dist(rng);

    auto run_prefill = [&](Accum& acc) {
        cuda_rt->set_stage_profiling(true);
        setup.session.reset();
        auto t0 = std::chrono::high_resolution_clock::now();
        const bool can_pipeline = (gpus.size() == 2);
        const bool use_pipeline = (can_pipeline && (force_pipeline || overlap));
        if (use_pipeline) {
            ember::Error e = cuda_rt->prefill_chunked_pipeline(prompt, setup.session, chunk_len, /*overlap=*/overlap, nullptr);
            if (e) die("prefill_chunked_pipeline failed: " + e.to_string());
        } else {
            ember::Error e = runtime->prefill(prompt, setup.session);
            if (e) die("prefill failed: " + e.to_string());
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        const double wall = std::chrono::duration<double, std::milli>(t1 - t0).count();
        add(acc, wall, cuda_rt->take_last_stage_profile_ms());
    };

    auto run_decode = [&](Accum& acc) {
        if (decode_steps == 0) return;
        cuda_rt->set_stage_profiling(true);
        std::vector<float> tmp;
        int last = prompt.back();
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < decode_steps; ++i) {
            ember::Error e = cuda_rt->decode_to_device(last, setup.session);
            if (e) die("decode_to_device failed: " + e.to_string());
            add(acc, 0.0, cuda_rt->take_last_stage_profile_ms());
            last = (last + 1) % static_cast<int>(config.vocab_size);
        }
        for (int dev : gpus) {
            ember::Error se = ember::cuda::cuda_sync(dev);
            if (se) die("cuda_sync failed: " + se.to_string());
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        const double wall = std::chrono::duration<double, std::milli>(t1 - t0).count();
        acc.wall_ms += wall;
    };

    // Warmup.
    for (int w = 0; w < warmup; ++w) {
        Accum dummy;
        run_prefill(dummy);
        run_decode(dummy);
    }

    Accum pre_acc;
    Accum dec_acc;
    for (int it = 0; it < iters; ++it) {
        run_prefill(pre_acc);
        run_decode(dec_acc);
    }

    auto print_row = [&](const char* phase, const Accum& a, double denom) {
        auto div = [&](double x) { return x / denom; };
        const std::string mode = (gpus.size() == 2 && overlap) ? "overlap" : "no_overlap";
        const std::string gstr = join_ints(gpus);
        std::string sstr;
        if (gpus.size() == 2) {
            int a_split = split.empty() ? (static_cast<int>(config.num_layers) / 2) : split[0];
            int b_split = split.empty() ? (static_cast<int>(config.num_layers) - a_split) : split[1];
            sstr = std::to_string(a_split) + "+" + std::to_string(b_split);
        } else {
            sstr = std::to_string(config.num_layers);
        }
        *out << phase << "," << mode << "," << gstr << "," << sstr << ","
             << prompt_len << "," << decode_steps << "," << chunk_len << "," << (overlap ? 1 : 0) << ","
             << std::fixed << std::setprecision(3) << div(a.wall_ms) << ","
             << std::fixed << std::setprecision(3) << div(a.embedding_ms) << ","
             << std::fixed << std::setprecision(3) << div(a.rmsnorm_ms) << ","
             << std::fixed << std::setprecision(3) << div(a.attention_ms) << ","
             << std::fixed << std::setprecision(3) << div(a.ffn_ms) << ","
             << std::fixed << std::setprecision(3) << div(a.p2p_ms) << ","
             << std::fixed << std::setprecision(3) << div(a.lm_head_ms) << ","
             << std::fixed << std::setprecision(3) << div(a.prof_total_ms) << "\n";
    };

    print_row("prefill", pre_acc, static_cast<double>(iters));
    if (decode_steps > 0) {
        const double denom = static_cast<double>(iters) * static_cast<double>(decode_steps);
        print_row("decode_per_token", dec_acc, denom);
    }

    ember::shutdown_runtime(*runtime, setup);
    return 0;
}

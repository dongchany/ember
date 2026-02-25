#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
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
    std::string adapter_dir;
    std::vector<int> gpus = {0, 1};
    std::vector<int> split = {};
    float scale = 1.0f;
    int iters = 1;
    int warmup = 0;
    std::string csv_path;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) die(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (arg == "-h" || arg == "--help") {
            std::cout
                << "Ember LoRA Hot Update Benchmark\n\n"
                << "Usage:\n"
                << "  " << argv[0] << " --model <dir> --adapter <dir> [options]\n\n"
                << "Options:\n"
                << "  --model <dir>       model directory\n"
                << "  --adapter <dir>     LoRA adapter directory (contains adapter_model.safetensors)\n"
                << "  --gpus LIST         e.g. 0 or 0,1 (default: 0,1)\n"
                << "  --split A,B         layer split for 2 GPUs (default: even)\n"
                << "  --scale X           user scale before alpha/r (default: 1.0)\n"
                << "  --iters N           measured iterations (default: 1)\n"
                << "  --warmup N          warmup iterations (default: 0)\n"
                << "  --csv PATH          write CSV row (default: stdout)\n";
            return 0;
        } else if (arg == "--model") {
            model_dir = need("--model");
        } else if (arg == "--adapter") {
            adapter_dir = need("--adapter");
        } else if (arg == "--gpus") {
            gpus = split_ints(need("--gpus"));
        } else if (arg == "--split") {
            split = split_ints(need("--split"));
        } else if (arg == "--scale") {
            scale = std::stof(need("--scale"));
        } else if (arg == "--iters") {
            iters = std::stoi(need("--iters"));
        } else if (arg == "--warmup") {
            warmup = std::stoi(need("--warmup"));
        } else if (arg == "--csv") {
            csv_path = need("--csv");
        } else {
            die("unknown argument: " + arg);
        }
    }

    if (model_dir.empty()) die("--model is required");
    if (adapter_dir.empty()) die("--adapter is required");
    if (iters <= 0) die("--iters must be > 0");
    if (warmup < 0) die("--warmup must be >= 0");
    if (gpus.empty()) die("--gpus is empty");
    if (!split.empty() && split.size() != 2) die("--split expects A,B");
    if (gpus.size() > 2) die("benchmark supports only 1 or 2 GPUs");

    namespace fs = std::filesystem;
    fs::path adapter_path = fs::path(adapter_dir);
    if (fs::is_regular_file(adapter_path)) {
        adapter_path = adapter_path.parent_path();
    }
    if (!fs::exists(adapter_path)) {
        die("adapter path does not exist: " + adapter_path.string());
    }
    adapter_dir = adapter_path.string();

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
    runtime_config.max_ctx_len = 128;
    runtime_config.batch_size = 1;
    runtime_config.device_ids = gpus;
    runtime_config.kv_cache_dtype = ember::dtype_from_string(config.torch_dtype);
    if (runtime_config.kv_cache_dtype == ember::DType::UNKNOWN) runtime_config.kv_cache_dtype = ember::DType::F16;

    ember::RuntimeSetup setup;
    ember::Error err = ember::load_runtime(*runtime, model_dir, config, device_map, setup);
    if (err) die("load_runtime failed: " + err.to_string());
    err = ember::init_session_and_kv(*runtime, config, runtime_config, setup);
    if (err) die("init_session_and_kv failed: " + err.to_string());

    double ext_ms_sum = 0.0;
    double inner_ms_sum = 0.0;
    int updated_sum = 0;
    int skipped_sum = 0;
    float effective_scale = 0.0f;
    int measured = 0;

    for (int i = 0; i < warmup + iters; ++i) {
        ember::cuda::CudaRuntime::LoraApplyStats st{};
        auto t0 = std::chrono::high_resolution_clock::now();
        err = cuda_rt->apply_lora_adapter(adapter_dir, scale, &st);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (err) die("apply_lora_adapter failed: " + err.to_string());

        if (i >= warmup) {
            ext_ms_sum += ms_since(t0, t1);
            inner_ms_sum += st.wall_ms;
            updated_sum += st.updated_matrices;
            skipped_sum += st.skipped_matrices;
            effective_scale = st.scale_used;
            measured++;
        }
    }

    if (measured <= 0) die("no measured iterations");
    const double ext_ms_avg = ext_ms_sum / static_cast<double>(measured);
    const double inner_ms_avg = inner_ms_sum / static_cast<double>(measured);
    const int updated_avg = static_cast<int>(std::lround(
        static_cast<double>(updated_sum) / static_cast<double>(measured)));
    const int skipped_avg = static_cast<int>(std::lround(
        static_cast<double>(skipped_sum) / static_cast<double>(measured)));

    std::ostringstream row;
    row << std::fixed << std::setprecision(3)
        << "lora_hot_update" << ","
        << model_dir << ","
        << adapter_dir << ","
        << join_with_plus(gpus) << ","
        << join_with_plus(split) << ","
        << scale << ","
        << effective_scale << ","
        << iters << ","
        << warmup << ","
        << updated_avg << ","
        << skipped_avg << ","
        << ext_ms_avg << ","
        << inner_ms_avg;

    const std::string header =
        "mode,model_dir,adapter_dir,gpus,split,scale,effective_scale,iters,warmup,"
        "updated_matrices,skipped_matrices,apply_ms_ext,apply_ms_inner";

    if (!csv_path.empty()) {
        std::ofstream out(csv_path, std::ios::binary);
        if (!out.is_open()) die("failed to open csv: " + csv_path);
        out << header << "\n" << row.str() << "\n";
    } else {
        std::cout << header << "\n" << row.str() << "\n";
    }

    return 0;
}

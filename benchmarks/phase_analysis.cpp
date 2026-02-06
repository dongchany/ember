#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include "backends/cuda/cuda_runtime.h"
#include "backends/cuda/cuda_utils.h"
#include "core/config.h"
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
        if (tok.empty()) continue;
        out.push_back(std::stoi(tok));
    }
    return out;
}

struct LayerStats {
    double flops = 0.0;
    double bytes = 0.0;
};

static LayerStats estimate_layer_stats_prefill(const ember::ModelConfig& c, int prompt_len, ember::DType dtype) {
    const double elem = static_cast<double>(ember::dtype_size(dtype));
    const double H = static_cast<double>(c.hidden_size);
    const double I = static_cast<double>(c.intermediate_size);
    const double Nh = static_cast<double>(c.num_heads);
    const double Hd = static_cast<double>(c.head_dim);
    const double Nk = static_cast<double>(c.num_kv_heads);

    const double L = static_cast<double>(prompt_len);
    const double M = L;  // batch=1

    // GEMMs: FLOPs ~= 2 * M * in * out
    const double flops_q = 2.0 * M * H * (Nh * Hd);
    const double flops_k = 2.0 * M * H * (Nk * Hd);
    const double flops_v = 2.0 * M * H * (Nk * Hd);
    const double flops_o = 2.0 * M * (Nh * Hd) * H;

    const double flops_gate = 2.0 * M * H * I;
    const double flops_up = 2.0 * M * H * I;
    const double flops_down = 2.0 * M * I * H;

    // Causal attention: roughly triangular.
    const double tri = L * (L + 1.0) / 2.0;
    const double flops_qk = 2.0 * Nh * tri * Hd;  // QK^T
    const double flops_pv = 2.0 * Nh * tri * Hd;  // P*V

    LayerStats s;
    s.flops = flops_q + flops_k + flops_v + flops_o + flops_gate + flops_up + flops_down + flops_qk + flops_pv;

    // Bandwidth is an estimate: read weights + KV read/write (full prefill writes all positions).
    const double weight_bytes =
        (H * (Nh * Hd) + H * (Nk * Hd) + H * (Nk * Hd) + (Nh * Hd) * H + 3.0 * H * I) * elem;
    const double kv_write_bytes = 2.0 * Nk * L * Hd * elem;
    const double kv_read_bytes = 0.0;
    s.bytes = weight_bytes + kv_write_bytes + kv_read_bytes;
    return s;
}

static LayerStats estimate_layer_stats_decode(const ember::ModelConfig& c, int seq_k, ember::DType dtype) {
    const double elem = static_cast<double>(ember::dtype_size(dtype));
    const double H = static_cast<double>(c.hidden_size);
    const double I = static_cast<double>(c.intermediate_size);
    const double Nh = static_cast<double>(c.num_heads);
    const double Hd = static_cast<double>(c.head_dim);
    const double Nk = static_cast<double>(c.num_kv_heads);

    const double M = 1.0;  // batch=1, seq_len=1 token
    const double K = static_cast<double>(seq_k);

    const double flops_q = 2.0 * M * H * (Nh * Hd);
    const double flops_k = 2.0 * M * H * (Nk * Hd);
    const double flops_v = 2.0 * M * H * (Nk * Hd);
    const double flops_o = 2.0 * M * (Nh * Hd) * H;

    const double flops_gate = 2.0 * M * H * I;
    const double flops_up = 2.0 * M * H * I;
    const double flops_down = 2.0 * M * I * H;

    // Decode: QK^T and P*V with seq_q=1 => Nh * K dot products.
    const double flops_qk = 2.0 * Nh * K * Hd;
    const double flops_pv = 2.0 * Nh * K * Hd;

    LayerStats s;
    s.flops = flops_q + flops_k + flops_v + flops_o + flops_gate + flops_up + flops_down + flops_qk + flops_pv;

    const double weight_bytes =
        (H * (Nh * Hd) + H * (Nk * Hd) + H * (Nk * Hd) + (Nh * Hd) * H + 3.0 * H * I) * elem;
    const double kv_read_bytes = 2.0 * Nk * K * Hd * elem;
    const double kv_write_bytes = 2.0 * Nk * 1.0 * Hd * elem;
    s.bytes = weight_bytes + kv_read_bytes + kv_write_bytes;
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_dir;
    std::vector<int> prompt_lens;
    int decode_steps = 100;
    int device_id = 0;
    std::string out_csv;
    int warmup = 1;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) die(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (arg == "-h" || arg == "--help") {
            std::cout
                << "Ember Phase Analysis Benchmark\n\n"
                << "Usage:\n"
                << "  " << argv[0] << " --model <dir> --prompt-lens 128,512,1024,2048 [options]\n\n"
                << "Options:\n"
                << "  --decode-steps N   (default: 100)\n"
                << "  --device ID        (default: 0)\n"
                << "  --warmup N         warmup runs per prompt length (default: 1)\n"
                << "  --output PATH      write CSV (default: stdout)\n";
            return 0;
        } else if (arg == "--model") {
            model_dir = need("--model");
        } else if (arg == "--prompt-lens") {
            prompt_lens = split_ints(need("--prompt-lens"));
        } else if (arg == "--decode-steps") {
            decode_steps = std::stoi(need("--decode-steps"));
        } else if (arg == "--device") {
            device_id = std::stoi(need("--device"));
        } else if (arg == "--warmup") {
            warmup = std::stoi(need("--warmup"));
        } else if (arg == "--output") {
            out_csv = need("--output");
        } else {
            die("unknown argument: " + arg);
        }
    }

    if (model_dir.empty()) die("--model is required");
    if (prompt_lens.empty()) die("--prompt-lens is required");
    if (decode_steps <= 0) die("--decode-steps must be > 0");

    ember::ModelConfig config;
    try {
        config = ember::parse_model_config(model_dir + "/config.json");
    } catch (const std::exception& ex) {
        die(std::string("parse_model_config failed: ") + ex.what());
    }

    auto runtime = ember::RuntimeFactory::create_cuda();
    if (!runtime || !runtime->available()) die("CUDA runtime not available");

    ember::DeviceMap dm = ember::DeviceMap::single_device(config.num_layers, device_id);

    ember::RuntimeConfig runtime_config;
    int max_prompt = *std::max_element(prompt_lens.begin(), prompt_lens.end());
    runtime_config.max_ctx_len = max_prompt + decode_steps + 8;
    runtime_config.batch_size = 1;
    runtime_config.device_ids = {device_id};
    runtime_config.kv_cache_dtype = ember::dtype_from_string(config.torch_dtype);
    if (runtime_config.kv_cache_dtype == ember::DType::UNKNOWN) runtime_config.kv_cache_dtype = ember::DType::F16;

    ember::RuntimeSetup setup;
    ember::Error err = ember::load_runtime(*runtime, model_dir, config, dm, setup);
    if (err) die("load_runtime failed: " + err.to_string());
    err = ember::init_session_and_kv(*runtime, config, runtime_config, setup);
    if (err) die("init_session_and_kv failed: " + err.to_string());

    auto* cuda_rt = dynamic_cast<ember::cuda::CudaRuntime*>(runtime.get());
    if (!cuda_rt) die("phase_analysis requires CUDA runtime implementation");

    std::ostream* out = &std::cout;
    std::ofstream f;
    if (!out_csv.empty()) {
        f.open(out_csv);
        if (!f.is_open()) die("failed to open output: " + out_csv);
        out = &f;
    }

    *out << "prompt_len,layer_id,prefill_time_ms,decode_time_ms,prefill_tflops,decode_tflops,prefill_bandwidth,decode_bandwidth\n";

    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(0, static_cast<int>(config.vocab_size - 1));

    for (int prompt_len : prompt_lens) {
        if (prompt_len <= 0) continue;
        std::vector<int> tokens(static_cast<size_t>(prompt_len));
        for (int& t : tokens) t = dist(rng);

        // Warmup runs (ignore output).
        for (int w = 0; w < warmup; ++w) {
            setup.session.reset();
            cuda_rt->set_layer_profiling(true);
            std::vector<float> tmp;
            runtime->prefill(tokens, setup.session);
            int last = tokens.back();
            for (int i = 0; i < std::min(4, decode_steps); ++i) {
                runtime->decode(last, setup.session, tmp);
                last = (last + 1) % static_cast<int>(config.vocab_size);
            }
        }

        setup.session.reset();

        cuda_rt->set_layer_profiling(true);
        err = runtime->prefill(tokens, setup.session);
        if (err) die("prefill failed: " + err.to_string());
        std::vector<float> prefill_ms = cuda_rt->take_last_layer_profile_ms();
        if (prefill_ms.size() != static_cast<size_t>(config.num_layers)) {
            die("unexpected prefill profile size");
        }

        std::vector<double> decode_sum_ms(static_cast<size_t>(config.num_layers), 0.0);
        std::vector<double> decode_sum_flops(static_cast<size_t>(config.num_layers), 0.0);
        std::vector<double> decode_sum_bytes(static_cast<size_t>(config.num_layers), 0.0);

        std::vector<float> logits;
        int last = tokens.back();
        for (int step = 0; step < decode_steps; ++step) {
            cuda_rt->set_layer_profiling(true);
            err = runtime->decode(last, setup.session, logits);
            if (err) die("decode failed: " + err.to_string());
            std::vector<float> ms = cuda_rt->take_last_layer_profile_ms();
            if (ms.size() != static_cast<size_t>(config.num_layers)) die("unexpected decode profile size");
            const int seq_k = prompt_len + step + 1;
            for (int layer = 0; layer < config.num_layers; ++layer) {
                decode_sum_ms[static_cast<size_t>(layer)] += static_cast<double>(ms[static_cast<size_t>(layer)]);
                LayerStats s = estimate_layer_stats_decode(config, seq_k, runtime_config.kv_cache_dtype);
                decode_sum_flops[static_cast<size_t>(layer)] += s.flops;
                decode_sum_bytes[static_cast<size_t>(layer)] += s.bytes;
            }
            last = (last + 1) % static_cast<int>(config.vocab_size);
        }

        for (int layer = 0; layer < config.num_layers; ++layer) {
            const double pre_ms = static_cast<double>(prefill_ms[static_cast<size_t>(layer)]);
            const double dec_ms = decode_sum_ms[static_cast<size_t>(layer)] / static_cast<double>(decode_steps);

            LayerStats pre_s = estimate_layer_stats_prefill(config, prompt_len, runtime_config.kv_cache_dtype);
            const double pre_tflops = (pre_s.flops / (pre_ms * 1e-3)) / 1e12;
            const double pre_bw = (pre_s.bytes / (pre_ms * 1e-3)) / 1e9;

            const double dec_total_ms = decode_sum_ms[static_cast<size_t>(layer)];
            const double dec_tflops = (decode_sum_flops[static_cast<size_t>(layer)] / (dec_total_ms * 1e-3)) / 1e12;
            const double dec_bw = (decode_sum_bytes[static_cast<size_t>(layer)] / (dec_total_ms * 1e-3)) / 1e9;

            *out << prompt_len << "," << layer << ","
                 << std::fixed << std::setprecision(4) << pre_ms << ","
                 << std::fixed << std::setprecision(4) << dec_ms << ","
                 << std::fixed << std::setprecision(3) << pre_tflops << ","
                 << std::fixed << std::setprecision(3) << dec_tflops << ","
                 << std::fixed << std::setprecision(1) << pre_bw << ","
                 << std::fixed << std::setprecision(1) << dec_bw << "\n";
        }
    }

    ember::shutdown_runtime(*runtime, setup);
    return 0;
}

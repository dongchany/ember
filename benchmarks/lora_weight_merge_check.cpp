#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_fp16.h>

#include "backends/cuda/cuda_runtime.h"
#include "core/config_loader.h"
#include "formats/safetensors.h"
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

float bf16_to_f32(uint16_t v) {
    uint32_t bits = static_cast<uint32_t>(v) << 16;
    float f = 0.0f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

std::vector<float> tensor_to_float(const ember::Tensor& t) {
    const size_t n = t.numel();
    std::vector<float> out(n, 0.0f);
    if (t.dtype == ember::DType::F32) {
        const float* p = static_cast<const float*>(t.data);
        std::copy(p, p + n, out.begin());
    } else if (t.dtype == ember::DType::F16) {
        const uint16_t* p = static_cast<const uint16_t*>(t.data);
        for (size_t i = 0; i < n; ++i) {
            __half_raw raw{};
            raw.x = p[i];
            out[i] = __half2float(static_cast<__half>(raw));
        }
    } else if (t.dtype == ember::DType::BF16) {
        const uint16_t* p = static_cast<const uint16_t*>(t.data);
        for (size_t i = 0; i < n; ++i) {
            out[i] = bf16_to_f32(p[i]);
        }
    } else {
        die("unsupported tensor dtype for conversion");
    }
    return out;
}

struct DiffStats {
    float max_abs = 0.0f;
    float mean_abs = 0.0f;
};

DiffStats diff_stats(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) die("diff_stats size mismatch");
    DiffStats s{};
    if (a.empty()) return s;
    double acc = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const float d = std::fabs(a[i] - b[i]);
        s.max_abs = std::max(s.max_abs, d);
        acc += static_cast<double>(d);
    }
    s.mean_abs = static_cast<float>(acc / static_cast<double>(a.size()));
    return s;
}

float read_alpha_over_r(const std::string& adapter_dir) {
    namespace fs = std::filesystem;
    const fs::path cfg = fs::path(adapter_dir) / "adapter_config.json";
    if (!fs::exists(cfg)) return 1.0f;
    std::ifstream in(cfg);
    if (!in.is_open()) return 1.0f;
    std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (s.empty()) return 1.0f;

    std::smatch m_alpha;
    std::smatch m_r;
    static const std::regex kAlpha(R"("lora_alpha"\s*:\s*([0-9]+(?:\.[0-9]+)?))");
    static const std::regex kR(R"("r"\s*:\s*([0-9]+(?:\.[0-9]+)?))");
    if (!std::regex_search(s, m_alpha, kAlpha)) return 1.0f;
    if (!std::regex_search(s, m_r, kR)) return 1.0f;
    const float alpha = std::stof(m_alpha[1].str());
    const float r = std::stof(m_r[1].str());
    if (alpha <= 0.0f || r <= 0.0f) return 1.0f;
    return alpha / r;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_dir;
    std::string adapter_dir;
    std::vector<int> gpus = {0};
    std::vector<int> split = {};
    int layer_idx = 0;
    std::string proj = "q_proj";
    float scale = 1.0f;
    std::string csv_path;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) die(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (arg == "-h" || arg == "--help") {
            std::cout
                << "Ember LoRA Weight Merge Check\n\n"
                << "Usage:\n"
                << "  " << argv[0] << " --model <dir> --adapter <dir> --layer N --proj NAME [options]\n\n"
                << "Options:\n"
                << "  --model <dir>       model directory\n"
                << "  --adapter <dir>     LoRA adapter directory\n"
                << "  --gpus LIST         e.g. 0 or 0,1 (default: 0)\n"
                << "  --split A,B         layer split for 2 GPUs\n"
                << "  --layer N           target layer index\n"
                << "  --proj NAME         q_proj|k_proj|v_proj|o_proj\n"
                << "  --scale X           user scale before alpha/r (default: 1.0)\n"
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
        } else if (arg == "--layer") {
            layer_idx = std::stoi(need("--layer"));
        } else if (arg == "--proj") {
            proj = need("--proj");
        } else if (arg == "--scale") {
            scale = std::stof(need("--scale"));
        } else if (arg == "--csv") {
            csv_path = need("--csv");
        } else {
            die("unknown argument: " + arg);
        }
    }

    if (model_dir.empty()) die("--model is required");
    if (adapter_dir.empty()) die("--adapter is required");
    if (gpus.empty()) die("--gpus is empty");
    if (!split.empty() && split.size() != 2) die("--split expects A,B");
    if (gpus.size() > 2) die("only 1 or 2 GPUs supported");
    if (proj != "q_proj" && proj != "k_proj" && proj != "v_proj" && proj != "o_proj") {
        die("--proj must be q_proj|k_proj|v_proj|o_proj");
    }

    ember::ModelConfig config{};
    try {
        config = ember::parse_model_config(model_dir + "/config.json");
    } catch (const std::exception& ex) {
        die(std::string("parse_model_config failed: ") + ex.what());
    }
    if (layer_idx < 0 || layer_idx >= config.num_layers) {
        die("--layer out of range");
    }

    ember::DeviceMap device_map;
    if (gpus.size() == 1) {
        device_map = ember::DeviceMap::single_device(config.num_layers, gpus[0]);
    } else {
        const int a = split.empty() ? (config.num_layers / 2) : split[0];
        const int b = split.empty() ? (config.num_layers - a) : split[1];
        if (a <= 0 || b <= 0 || a + b != config.num_layers) die("invalid --split");
        device_map.num_devices = 2;
        device_map.embedding_device = gpus[0];
        device_map.lm_head_device = gpus[1];
        device_map.layer_to_device.resize(static_cast<size_t>(config.num_layers));
        for (int i = 0; i < config.num_layers; ++i) {
            device_map.layer_to_device[static_cast<size_t>(i)] = (i < a) ? gpus[0] : gpus[1];
        }
    }

    auto runtime = ember::RuntimeFactory::create_cuda();
    if (!runtime || !runtime->available()) die("CUDA runtime not available");
    auto* cuda_rt = dynamic_cast<ember::cuda::CudaRuntime*>(runtime.get());
    if (!cuda_rt) die("expected CUDA runtime");

    ember::RuntimeSetup setup;
    ember::Error err = ember::load_runtime(*runtime, model_dir, config, device_map, setup);
    if (err) die("load_runtime failed: " + err.to_string());

    std::vector<float> w_before;
    int out_dim = 0;
    int in_dim = 0;
    err = cuda_rt->debug_copy_attention_weight(layer_idx, proj, w_before, &out_dim, &in_dim);
    if (err) die("debug_copy_attention_weight(before) failed: " + err.to_string());

    ember::ModelWeightLoader loader;
    err = loader.open(adapter_dir);
    if (err) die("open adapter failed: " + err.to_string());

    std::string a_name;
    std::string b_name;
    std::regex pat("layers\\.(" + std::to_string(layer_idx) + ")\\.self_attn\\.(" + proj +
                   ")\\.lora_([AB])(?:\\.default)?\\.weight$");
    for (const auto& name : loader.tensor_names()) {
        std::smatch m;
        if (!std::regex_search(name, m, pat)) continue;
        if (m[3].str() == "A") a_name = name;
        if (m[3].str() == "B") b_name = name;
    }
    if (a_name.empty() || b_name.empty()) {
        die("failed to find LoRA A/B tensors for target layer/proj");
    }

    ember::Tensor a_t;
    ember::Tensor b_t;
    err = loader.read_tensor(a_name, a_t);
    if (err) die("read A failed: " + err.to_string());
    err = loader.read_tensor(b_name, b_t);
    if (err) die("read B failed: " + err.to_string());
    if (a_t.shape.size() != 2 || b_t.shape.size() != 2) {
        die("A/B tensors must be rank-2");
    }
    const int r = static_cast<int>(a_t.shape[0]);
    const int a_in = static_cast<int>(a_t.shape[1]);
    const int b_out = static_cast<int>(b_t.shape[0]);
    const int b_r = static_cast<int>(b_t.shape[1]);
    if (a_in != in_dim || b_out != out_dim || b_r != r) {
        die("LoRA A/B shapes mismatch target weight shape");
    }

    const std::vector<float> a = tensor_to_float(a_t);  // [r, in]
    const std::vector<float> b = tensor_to_float(b_t);  // [out, r]
    std::free(a_t.data);
    std::free(b_t.data);

    const float effective_scale = scale * read_alpha_over_r(adapter_dir);
    std::vector<float> expected_delta(static_cast<size_t>(out_dim) * static_cast<size_t>(in_dim), 0.0f);
    for (int o = 0; o < out_dim; ++o) {
        for (int i = 0; i < in_dim; ++i) {
            float sum = 0.0f;
            for (int k = 0; k < r; ++k) {
                sum += b[static_cast<size_t>(o) * static_cast<size_t>(r) + static_cast<size_t>(k)] *
                       a[static_cast<size_t>(k) * static_cast<size_t>(in_dim) + static_cast<size_t>(i)];
            }
            expected_delta[static_cast<size_t>(o) * static_cast<size_t>(in_dim) + static_cast<size_t>(i)] =
                sum * effective_scale;
        }
    }

    ember::cuda::CudaRuntime::LoraApplyStats st{};
    err = cuda_rt->apply_lora_adapter(adapter_dir, scale, false, &st);
    if (err) die("apply_lora_adapter(+scale) failed: " + err.to_string());

    std::vector<float> w_after;
    err = cuda_rt->debug_copy_attention_weight(layer_idx, proj, w_after, nullptr, nullptr);
    if (err) die("debug_copy_attention_weight(after) failed: " + err.to_string());

    std::vector<float> observed_delta(w_before.size(), 0.0f);
    for (size_t i = 0; i < w_before.size(); ++i) {
        observed_delta[i] = w_after[i] - w_before[i];
    }
    DiffStats merge_diff = diff_stats(observed_delta, expected_delta);

    err = cuda_rt->apply_lora_adapter(adapter_dir, -scale, false, nullptr);
    if (err) die("apply_lora_adapter(-scale) failed: " + err.to_string());
    std::vector<float> w_back;
    err = cuda_rt->debug_copy_attention_weight(layer_idx, proj, w_back, nullptr, nullptr);
    if (err) die("debug_copy_attention_weight(back) failed: " + err.to_string());
    DiffStats rollback_diff = diff_stats(w_back, w_before);

    std::ostringstream row;
    row << std::fixed << std::setprecision(8)
        << "lora_weight_merge_check" << ","
        << model_dir << ","
        << adapter_dir << ","
        << join_with_plus(gpus) << ","
        << join_with_plus(split) << ","
        << layer_idx << ","
        << proj << ","
        << scale << ","
        << effective_scale << ","
        << out_dim << ","
        << in_dim << ","
        << r << ","
        << merge_diff.max_abs << ","
        << merge_diff.mean_abs << ","
        << rollback_diff.max_abs << ","
        << rollback_diff.mean_abs;

    const std::string header =
        "mode,model_dir,adapter_dir,gpus,split,layer_idx,proj,scale,effective_scale,out_dim,in_dim,rank,"
        "delta_max_abs_diff,delta_mean_abs_diff,rollback_max_abs_diff,rollback_mean_abs_diff";

    if (!csv_path.empty()) {
        std::ofstream out(csv_path, std::ios::binary);
        if (!out.is_open()) die("failed to open csv: " + csv_path);
        out << header << "\n" << row.str() << "\n";
    } else {
        std::cout << header << "\n" << row.str() << "\n";
    }
    return 0;
}

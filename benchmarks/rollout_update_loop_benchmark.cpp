#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "backends/cuda/cuda_runtime.h"
#include "core/config_loader.h"
#include "core/sampler.h"
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

std::vector<int> sample_random_prompt(int prompt_len, int vocab_size, int seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, std::max(1, vocab_size - 1));
    std::vector<int> tokens(static_cast<size_t>(prompt_len));
    for (int& t : tokens) t = dist(rng);
    return tokens;
}

enum class UpdateMode {
    APPLY,
    SKIP,
};

UpdateMode parse_update_mode(const std::string& s) {
    if (s == "apply") return UpdateMode::APPLY;
    if (s == "skip") return UpdateMode::SKIP;
    die("invalid --update-mode: " + s + " (expect apply|skip)");
    return UpdateMode::APPLY;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_dir;
    std::string adapter_dir;
    std::vector<int> gpus = {0, 1};
    std::vector<int> split = {};
    int prompt_len = 1024;
    int gen_len = 128;
    int num_candidates = 8;
    int rounds = 10;
    int warmup = 2;
    int chunk_len = 512;
    bool overlap = true;
    float temperature = 0.7f;
    float top_p = 0.9f;
    int top_k = 40;
    int seed = 1234;
    float scale = 1.0f;
    bool replace_existing = true;
    UpdateMode update_mode = UpdateMode::APPLY;
    double simulate_sync_ms = 0.0;
    std::string csv_path;
    std::string per_round_csv_path;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) die(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (arg == "-h" || arg == "--help") {
            std::cout
                << "Ember Rollout+Update Loop Benchmark\n\n"
                << "Usage:\n"
                << "  " << argv[0] << " --model <dir> [options]\n\n"
                << "Options:\n"
                << "  --model <dir>          model directory\n"
                << "  --adapter <dir>        LoRA adapter dir (required when --update-mode apply)\n"
                << "  --update-mode MODE     apply|skip (default: apply)\n"
                << "  --simulate-sync-ms X   extra per-round sleep to emulate dual-stack sync (default: 0)\n"
                << "  --gpus LIST            e.g. 0 or 0,1 (default: 0,1)\n"
                << "  --split A,B            layer split for 2 GPUs (default: even)\n"
                << "  --prompt-len N         prompt length (default: 1024)\n"
                << "  --gen-len N            generation length per candidate (default: 128)\n"
                << "  --num-candidates N     candidates per round (default: 8)\n"
                << "  --rounds N             measured rounds (default: 10)\n"
                << "  --warmup N             warmup rounds (default: 2)\n"
                << "  --chunk-len N          chunk len for 2-GPU prefill (default: 512)\n"
                << "  --overlap / --no-overlap\n"
                << "  --temperature F        (default: 0.7)\n"
                << "  --top-p F              (default: 0.9)\n"
                << "  --top-k N              (default: 40)\n"
                << "  --seed N               RNG seed (default: 1234)\n"
                << "  --scale X              LoRA user scale (default: 1.0)\n"
                << "  --no-replace-existing  keep previous adapter merged (default: replace)\n"
                << "  --csv PATH             summary CSV row\n"
                << "  --per-round-csv PATH   optional per-round CSV\n";
            return 0;
        } else if (arg == "--model") {
            model_dir = need("--model");
        } else if (arg == "--adapter") {
            adapter_dir = need("--adapter");
        } else if (arg == "--update-mode") {
            update_mode = parse_update_mode(need("--update-mode"));
        } else if (arg == "--simulate-sync-ms") {
            simulate_sync_ms = std::stod(need("--simulate-sync-ms"));
        } else if (arg == "--gpus") {
            gpus = split_ints(need("--gpus"));
        } else if (arg == "--split") {
            split = split_ints(need("--split"));
        } else if (arg == "--prompt-len") {
            prompt_len = std::stoi(need("--prompt-len"));
        } else if (arg == "--gen-len") {
            gen_len = std::stoi(need("--gen-len"));
        } else if (arg == "--num-candidates") {
            num_candidates = std::stoi(need("--num-candidates"));
        } else if (arg == "--rounds") {
            rounds = std::stoi(need("--rounds"));
        } else if (arg == "--warmup") {
            warmup = std::stoi(need("--warmup"));
        } else if (arg == "--chunk-len") {
            chunk_len = std::stoi(need("--chunk-len"));
        } else if (arg == "--overlap") {
            overlap = true;
        } else if (arg == "--no-overlap") {
            overlap = false;
        } else if (arg == "--temperature") {
            temperature = std::stof(need("--temperature"));
        } else if (arg == "--top-p") {
            top_p = std::stof(need("--top-p"));
        } else if (arg == "--top-k") {
            top_k = std::stoi(need("--top-k"));
        } else if (arg == "--seed") {
            seed = std::stoi(need("--seed"));
        } else if (arg == "--scale") {
            scale = std::stof(need("--scale"));
        } else if (arg == "--no-replace-existing") {
            replace_existing = false;
        } else if (arg == "--csv") {
            csv_path = need("--csv");
        } else if (arg == "--per-round-csv") {
            per_round_csv_path = need("--per-round-csv");
        } else {
            die("unknown argument: " + arg);
        }
    }

    if (model_dir.empty()) die("--model is required");
    if (prompt_len <= 0) die("--prompt-len must be > 0");
    if (gen_len <= 0) die("--gen-len must be > 0");
    if (num_candidates <= 0) die("--num-candidates must be > 0");
    if (rounds <= 0) die("--rounds must be > 0");
    if (warmup < 0) die("--warmup must be >= 0");
    if (simulate_sync_ms < 0.0) die("--simulate-sync-ms must be >= 0");
    if (gpus.empty()) die("--gpus is empty");
    if (!split.empty() && split.size() != 2) die("--split expects A,B");
    if (gpus.size() > 2) die("benchmark supports only 1 or 2 GPUs");
    if (update_mode == UpdateMode::APPLY) {
        if (adapter_dir.empty()) die("--adapter is required when --update-mode apply");
        namespace fs = std::filesystem;
        fs::path ap = fs::path(adapter_dir);
        if (fs::is_regular_file(ap)) ap = ap.parent_path();
        if (!fs::exists(ap)) die("adapter path not found: " + ap.string());
        adapter_dir = ap.string();
    }

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
    std::vector<int> split_used;
    if (gpus.size() == 1) {
        device_map = ember::DeviceMap::single_device(config.num_layers, gpus[0]);
        split_used = {static_cast<int>(config.num_layers), 0};
    } else {
        int a = split.empty() ? (static_cast<int>(config.num_layers) / 2) : split[0];
        int b = split.empty() ? (static_cast<int>(config.num_layers) - a) : split[1];
        if (a <= 0 || b <= 0 || a + b != config.num_layers) die("invalid --split");
        device_map.num_devices = 2;
        device_map.embedding_device = gpus[0];
        device_map.lm_head_device = gpus[1];
        device_map.layer_to_device.resize(static_cast<size_t>(config.num_layers));
        for (int li = 0; li < config.num_layers; ++li) {
            device_map.layer_to_device[static_cast<size_t>(li)] = (li < a) ? gpus[0] : gpus[1];
        }
        split_used = {a, b};
    }

    ember::RuntimeConfig runtime_config;
    runtime_config.max_ctx_len = prompt_len + gen_len + 8;
    runtime_config.batch_size = num_candidates;
    runtime_config.device_ids = gpus;
    runtime_config.kv_cache_dtype = ember::dtype_from_string(config.torch_dtype);
    if (runtime_config.kv_cache_dtype == ember::DType::UNKNOWN) runtime_config.kv_cache_dtype = ember::DType::F16;
    runtime_config.temperature = temperature;
    runtime_config.top_p = top_p;
    runtime_config.top_k = top_k;

    ember::RuntimeSetup setup;
    ember::Error err = ember::load_runtime(*runtime, model_dir, config, device_map, setup);
    if (err) die("load_runtime failed: " + err.to_string());
    err = ember::init_session_and_kv(*runtime, config, runtime_config, setup);
    if (err) die("init_session_and_kv failed: " + err.to_string());

    const std::vector<int> prompt_tokens = sample_random_prompt(prompt_len, config.vocab_size, seed);
    const size_t tokens_per_round = static_cast<size_t>(gen_len) * static_cast<size_t>(num_candidates);

    double sum_update_ext_ms = 0.0;
    double sum_update_inner_ms = 0.0;
    double sum_sync_ms = 0.0;
    double sum_prefill_ms = 0.0;
    double sum_decode_ms = 0.0;
    double sum_rollout_ms = 0.0;
    double sum_round_ms = 0.0;
    int measured = 0;
    int updated_mats_last = 0;
    int skipped_mats_last = 0;
    float effective_scale_last = 0.0f;

    std::vector<std::string> per_round_lines;
    per_round_lines.push_back(
        "round,phase,update_ms_ext,update_ms_inner,sync_ms,prefill_ms,decode_ms,rollout_ms,round_ms,gen_tokens");

    for (int r = 0; r < warmup + rounds; ++r) {
        double update_ext_ms = 0.0;
        double update_inner_ms = 0.0;

        if (update_mode == UpdateMode::APPLY) {
            ember::cuda::CudaRuntime::LoraApplyStats st{};
            auto t_up0 = std::chrono::high_resolution_clock::now();
            err = cuda_rt->apply_lora_adapter(adapter_dir, scale, replace_existing, &st);
            auto t_up1 = std::chrono::high_resolution_clock::now();
            if (err) die("apply_lora_adapter failed at round " + std::to_string(r) + ": " + err.to_string());
            update_ext_ms = ms_since(t_up0, t_up1);
            update_inner_ms = st.wall_ms;
            updated_mats_last = st.updated_matrices;
            skipped_mats_last = st.skipped_matrices;
            effective_scale_last = st.scale_used;
        }

        if (simulate_sync_ms > 0.0) {
            std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(simulate_sync_ms));
        }

        setup.session.reset();
        std::vector<std::vector<int>> histories(static_cast<size_t>(num_candidates), prompt_tokens);
        std::vector<int> last_tokens(static_cast<size_t>(num_candidates), 0);

        ember::Sampler sampler(temperature, top_k, top_p);
        sampler.set_seed(static_cast<uint64_t>(seed + r));

        auto t_pf0 = std::chrono::high_resolution_clock::now();
        for (int slot = 0; slot < num_candidates; ++slot) {
            std::vector<float> logits;
            if (gpus.size() == 2) {
                err = cuda_rt->prefill_into_slot_pipeline(prompt_tokens, slot, setup.session, chunk_len, overlap, &logits);
            } else {
                err = cuda_rt->prefill_into_slot(prompt_tokens, slot, setup.session, &logits);
            }
            if (err) die("prefill_into_slot failed at round " + std::to_string(r) + ": " + err.to_string());
            const int tok = sampler.sample(logits, runtime_config, histories[static_cast<size_t>(slot)]);
            histories[static_cast<size_t>(slot)].push_back(tok);
            last_tokens[static_cast<size_t>(slot)] = tok;
        }
        auto t_pf1 = std::chrono::high_resolution_clock::now();

        auto t_dec0 = std::chrono::high_resolution_clock::now();
        const size_t vocab = static_cast<size_t>(config.vocab_size);
        for (int step = 1; step < gen_len; ++step) {
            std::vector<float> logits_flat;
            err = cuda_rt->decode_batch(last_tokens, setup.session, logits_flat);
            if (err) die("decode_batch failed at round " + std::to_string(r) + ": " + err.to_string());
            if (logits_flat.size() != static_cast<size_t>(num_candidates) * vocab) {
                die("unexpected logits_flat size");
            }
            for (int slot = 0; slot < num_candidates; ++slot) {
                const size_t off = static_cast<size_t>(slot) * vocab;
                std::vector<float> row(
                    logits_flat.begin() + static_cast<std::ptrdiff_t>(off),
                    logits_flat.begin() + static_cast<std::ptrdiff_t>(off + vocab));
                const int tok = sampler.sample(row, runtime_config, histories[static_cast<size_t>(slot)]);
                histories[static_cast<size_t>(slot)].push_back(tok);
                last_tokens[static_cast<size_t>(slot)] = tok;
            }
        }
        auto t_dec1 = std::chrono::high_resolution_clock::now();

        const double prefill_ms = ms_since(t_pf0, t_pf1);
        const double decode_ms = ms_since(t_dec0, t_dec1);
        const double rollout_ms = prefill_ms + decode_ms;
        const double round_ms = update_ext_ms + simulate_sync_ms + rollout_ms;

        per_round_lines.push_back(
            std::to_string(r) + "," + (r < warmup ? "warmup" : "measured") + "," +
            std::to_string(update_ext_ms) + "," + std::to_string(update_inner_ms) + "," +
            std::to_string(simulate_sync_ms) + "," + std::to_string(prefill_ms) + "," +
            std::to_string(decode_ms) + "," + std::to_string(rollout_ms) + "," +
            std::to_string(round_ms) + "," + std::to_string(tokens_per_round));

        if (r >= warmup) {
            sum_update_ext_ms += update_ext_ms;
            sum_update_inner_ms += update_inner_ms;
            sum_sync_ms += simulate_sync_ms;
            sum_prefill_ms += prefill_ms;
            sum_decode_ms += decode_ms;
            sum_rollout_ms += rollout_ms;
            sum_round_ms += round_ms;
            measured++;
        }
    }

    if (measured <= 0) die("no measured rounds");
    const double measured_tokens = static_cast<double>(tokens_per_round) * static_cast<double>(measured);
    const double rollout_tok_s = sum_rollout_ms > 0.0 ? (measured_tokens * 1000.0 / sum_rollout_ms) : 0.0;
    const double e2e_tok_s = sum_round_ms > 0.0 ? (measured_tokens * 1000.0 / sum_round_ms) : 0.0;

    const std::string header =
        "mode,update_mode,rounds,warmup,prompt_len,gen_len,num_candidates,gpus,split,overlap,chunk_len,"
        "temperature,top_p,top_k,scale,replace_existing,simulate_sync_ms,"
        "updated_matrices,skipped_matrices,effective_scale,"
        "update_ms_ext_avg,update_ms_inner_avg,sync_ms_avg,prefill_ms_avg,decode_ms_avg,rollout_ms_avg,round_ms_avg,"
        "tokens_per_round,total_tokens_measured,rollout_tok_s,e2e_tok_s";

    std::ostringstream row;
    row << std::fixed << std::setprecision(6)
        << "rollout_update_loop,"
        << (update_mode == UpdateMode::APPLY ? "apply" : "skip") << ","
        << rounds << ","
        << warmup << ","
        << prompt_len << ","
        << gen_len << ","
        << num_candidates << ","
        << join_with_plus(gpus) << ","
        << join_with_plus(split_used) << ","
        << (overlap ? 1 : 0) << ","
        << chunk_len << ","
        << temperature << ","
        << top_p << ","
        << top_k << ","
        << scale << ","
        << (replace_existing ? 1 : 0) << ","
        << simulate_sync_ms << ","
        << updated_mats_last << ","
        << skipped_mats_last << ","
        << effective_scale_last << ","
        << (sum_update_ext_ms / static_cast<double>(measured)) << ","
        << (sum_update_inner_ms / static_cast<double>(measured)) << ","
        << (sum_sync_ms / static_cast<double>(measured)) << ","
        << (sum_prefill_ms / static_cast<double>(measured)) << ","
        << (sum_decode_ms / static_cast<double>(measured)) << ","
        << (sum_rollout_ms / static_cast<double>(measured)) << ","
        << (sum_round_ms / static_cast<double>(measured)) << ","
        << tokens_per_round << ","
        << static_cast<size_t>(measured_tokens) << ","
        << rollout_tok_s << ","
        << e2e_tok_s;

    if (!csv_path.empty()) {
        std::ofstream out(csv_path);
        if (!out.is_open()) die("failed to open csv: " + csv_path);
        out << header << "\n" << row.str() << "\n";
    } else {
        std::cout << header << "\n" << row.str() << "\n";
    }

    if (!per_round_csv_path.empty()) {
        std::ofstream out(per_round_csv_path);
        if (!out.is_open()) die("failed to open per-round csv: " + per_round_csv_path);
        for (const std::string& ln : per_round_lines) out << ln << "\n";
    }

    return 0;
}

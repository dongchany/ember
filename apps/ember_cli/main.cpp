// Ember - Qwen3 CUDA Inference Engine
// Main CLI Entry Point

#include <iostream>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <unordered_set>
#include <cctype>

#include "cli/args.h"
#include "core/types.h"
#include "core/error.h"
#include "core/config.h"
#include "core/config_loader.h"
#include "core/session.h"
#include "core/sampler.h"
#include "core/tokenizer.h"
#include "runtime/iruntime.h"
#include "runtime/runtime_setup.h"
#include "runtime/scheduler.h"
#include "backends/cuda/cuda_runtime.h"
#include "backends/cuda/cuda_utils.h"

namespace fs = std::filesystem;

// ANSI color helpers for the startup banner.
#define C_RESET "\033[0m"
#define C_ORANGE "\033[38;5;208m"
#define C_YELLOW "\033[33m"
#define C_RED "\033[31m"
#define C_DIM "\033[2m"
#define C_BOLD "\033[1m"

static inline void ember_banner() {
    std::printf("\n");
    std::printf(C_ORANGE  "    ███████╗███╗   ███╗██████╗ ███████╗██████╗ \n" C_RESET);
    std::printf(C_ORANGE  "    ██╔════╝████╗ ████║██╔══██╗██╔════╝██╔══██╗\n" C_RESET);
    std::printf(C_YELLOW  "    █████╗  ██╔████╔██║██████╔╝█████╗  ██████╔╝\n" C_RESET);
    std::printf(C_YELLOW  "    ██╔══╝  ██║╚██╔╝██║██╔══██╗██╔══╝  ██╔══██╗\n" C_RESET);
    std::printf(C_RED     "    ███████╗██║ ╚═╝ ██║██████╔╝███████╗██║  ██║\n" C_RESET);
    std::printf(C_RED     "    ╚══════╝╚═╝     ╚═╝╚═════╝ ╚══════╝╚═╝  ╚═╝\n" C_RESET);
    std::printf("\n");
    std::printf(C_DIM     "    ─────────────────────────────────────────────\n" C_RESET);
    std::printf(C_BOLD    "      日拱一卒，功不唐捐；蹄疾步稳，如临深渊。\n" C_RESET);
    std::printf(C_DIM     "    ─────────────────────────────────────────────\n" C_RESET);
    std::printf(C_DIM     "    Lightweight CUDA Inference Engine for Qwen3\n" C_RESET);
    std::printf("\n");
}

static std::string read_text_file(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) return "";
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

static std::string find_json_string_value(const std::string& content, const std::string& key) {
    size_t pos = content.find("\"" + key + "\"");
    if (pos == std::string::npos) return "";
    pos = content.find(":", pos);
    if (pos == std::string::npos) return "";
    pos = content.find("\"", pos);
    if (pos == std::string::npos) return "";
    std::string out;
    bool escape = false;
    for (size_t i = pos + 1; i < content.size(); ++i) {
        char c = content[i];
        if (escape) {
            switch (c) {
                case 'n': out += '\n'; break;
                case 't': out += '\t'; break;
                case 'r': out += '\r'; break;
                case '\\': out += '\\'; break;
                case '"': out += '"'; break;
                default: out += c; break;
            }
            escape = false;
        } else if (c == '\\') {
            escape = true;
        } else if (c == '"') {
            break;
        } else {
            out += c;
        }
    }
    return out;
}

static std::string load_chat_template(const std::string& model_dir) {
    std::string path = model_dir + "/tokenizer_config.json";
    if (!fs::exists(path)) return "";
    std::string content = read_text_file(path);
    if (content.empty()) return "";
    return find_json_string_value(content, "chat_template");
}

static std::string sanitize_name(std::string name) {
    for (char& c : name) {
        if (!(isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_')) {
            c = '_';
        }
    }
    if (name.empty()) name = "model";
    return name;
}

static std::string model_name_from_path(const std::string& model_path) {
    fs::path p(model_path);
    if (p.has_parent_path() && p.parent_path().filename() == "snapshots") {
        fs::path model_root = p.parent_path().parent_path().filename();
        if (!model_root.empty()) {
            return sanitize_name(model_root.string());
        }
    }
    return sanitize_name(p.filename().string());
}

static bool write_text_file(const fs::path& path, const std::string& content) {
    std::ofstream out(path);
    if (!out.is_open()) return false;
    out << content;
    return true;
}

static bool write_ints_file(const fs::path& path, const std::vector<int>& values) {
    std::ofstream out(path);
    if (!out.is_open()) return false;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) out << " ";
        out << values[i];
    }
    out << "\n";
    return true;
}

static bool write_f32_binary(const fs::path& path, const std::vector<float>& values) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) return false;
    if (!values.empty()) {
        out.write(reinterpret_cast<const char*>(values.data()),
                  static_cast<std::streamsize>(values.size() * sizeof(float)));
    }
    return true;
}

static bool write_check_meta(const fs::path& path, const std::string& model_path,
                             const std::string& prompt, int vocab_size,
                             int hidden_size, int num_layers, int token_count) {
    std::ofstream out(path);
    if (!out.is_open()) return false;
    out << "{\n";
    out << "  \"model_path\": \"" << model_path << "\",\n";
    out << "  \"prompt\": \"";
    for (char c : prompt) {
        if (c == '\\') out << "\\\\";
        else if (c == '\"') out << "\\\"";
        else if (c == '\n') out << "\\n";
        else if (c == '\t') out << "\\t";
        else out << c;
    }
    out << "\",\n";
    out << "  \"vocab_size\": " << vocab_size << ",\n";
    out << "  \"hidden_size\": " << hidden_size << ",\n";
    out << "  \"num_layers\": " << num_layers << ",\n";
    out << "  \"token_count\": " << token_count << "\n";
    out << "}\n";
    return true;
}

static std::vector<int> parse_eos_token_ids(const std::string& content) {
    std::vector<int> ids;
    size_t pos = content.find("\"eos_token_id\"");
    if (pos == std::string::npos) return ids;
    pos = content.find(":", pos);
    if (pos == std::string::npos) return ids;
    pos++;
    while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t' || content[pos] == '\n' || content[pos] == '\r')) pos++;
    if (pos >= content.size()) return ids;
    if (content[pos] == '[') {
        pos++;
        while (pos < content.size()) {
            while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t' || content[pos] == '\n' || content[pos] == '\r')) pos++;
            if (pos < content.size() && content[pos] == ']') break;
            size_t end = pos;
            while (end < content.size() && (isdigit(content[end]) || content[end] == '-')) end++;
            if (end > pos) {
                ids.push_back(std::stoi(content.substr(pos, end - pos)));
                pos = end;
            }
            while (pos < content.size() && content[pos] != ',' && content[pos] != ']') pos++;
            if (pos < content.size() && content[pos] == ',') pos++;
        }
    } else {
        size_t end = pos;
        while (end < content.size() && (isdigit(content[end]) || content[end] == '-')) end++;
        if (end > pos) {
            ids.push_back(std::stoi(content.substr(pos, end - pos)));
        }
    }
    return ids;
}

static std::vector<int> load_eos_token_ids(const std::string& model_dir) {
    std::string path = model_dir + "/generation_config.json";
    if (!fs::exists(path)) return {};
    std::string content = read_text_file(path);
    if (content.empty()) return {};
    return parse_eos_token_ids(content);
}

static bool find_json_number_value(const std::string& content, const std::string& key, double& out) {
    size_t pos = content.find("\"" + key + "\"");
    if (pos == std::string::npos) return false;
    pos = content.find(":", pos);
    if (pos == std::string::npos) return false;
    pos++;
    while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t' || content[pos] == '\n')) {
        pos++;
    }
    if (pos >= content.size()) return false;
    size_t end = pos;
    while (end < content.size()) {
        char c = content[end];
        if (!(isdigit(static_cast<unsigned char>(c)) || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E')) {
            break;
        }
        end++;
    }
    if (end <= pos) return false;
    try {
        out = std::stod(content.substr(pos, end - pos));
    } catch (...) {
        return false;
    }
    return true;
}

static void apply_generation_defaults(ember::cli::Args& args, const std::string& model_dir) {
    std::string path = model_dir + "/generation_config.json";
    if (!fs::exists(path)) return;
    std::string content = read_text_file(path);
    if (content.empty()) return;
    
    double value = 0.0;
    if (!args.temperature_set && find_json_number_value(content, "temperature", value)) {
        args.temperature = static_cast<float>(value);
    }
    if (!args.top_p_set && find_json_number_value(content, "top_p", value)) {
        args.top_p = static_cast<float>(value);
    }
    if (!args.top_k_set && find_json_number_value(content, "top_k", value)) {
        args.top_k = static_cast<int>(value);
    }
}

static std::string build_chat_prompt(const std::string& user_prompt) {
    std::string prompt;
    prompt += "<|im_start|>user\n";
    prompt += user_prompt;
    prompt += "<|im_end|>\n";
    prompt += "<|im_start|>assistant\n";
    return prompt;
}

static bool is_chat_model(const std::string& model_path) {
    std::string lower = model_path;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return (lower.find("instruct") != std::string::npos) ||
           (lower.find("chat") != std::string::npos);
}

// =============================================================================
// 主函数
// =============================================================================

int main(int argc, char** argv) {
    ember::cli::Args args;
    if (!ember::cli::parse_args(argc, argv, args)) {
        ember::cli::print_usage(argv[0]);
        return 1;
    }

    ember_banner();
    
    // 检查 GPU
    int num_gpus = ember::cuda::get_device_count();
    if (num_gpus == 0) {
        std::cerr << "Error: No CUDA devices found\n";
        return 1;
    }
    
    std::cout << "[System] Found " << num_gpus << " CUDA device(s)\n";
    for (int i = 0; i < num_gpus; ++i) {
        auto info = ember::cuda::get_gpu_info(i);
        std::cout << "  GPU " << i << ": " << info.name 
                  << " (" << (info.total_memory / (1024*1024*1024)) << " GB)\n";
    }
    std::cout << "\n";
    
    // 验证设备
    for (int dev : args.devices) {
        if (dev >= num_gpus) {
            std::cerr << "Error: Invalid device ID " << dev << "\n";
            return 1;
        }
    }
    
    // 自动检测 HuggingFace 缓存目录结构
    // 如果是 models--Org--Model 格式，自动找到 snapshots 中的最新版本
    auto resolve_hf_cache_path = [](const std::string& path) -> std::string {
        namespace fs = std::filesystem;
        
        // 检查是否已经是有效的模型目录
        if (fs::exists(path + "/config.json")) {
            return path;
        }
        
        // 检查是否是 HuggingFace 缓存目录
        fs::path snapshots_dir = fs::path(path) / "snapshots";
        if (fs::exists(snapshots_dir) && fs::is_directory(snapshots_dir)) {
            // 找到 snapshots 下的最新目录（按修改时间或字母顺序）
            std::vector<fs::path> snapshot_dirs;
            for (const auto& entry : fs::directory_iterator(snapshots_dir)) {
                if (entry.is_directory()) {
                    snapshot_dirs.push_back(entry.path());
                }
            }
            
            if (!snapshot_dirs.empty()) {
                // 按名称排序，取最后一个（通常是最新的 hash）
                std::sort(snapshot_dirs.begin(), snapshot_dirs.end());
                std::string resolved = snapshot_dirs.back().string();
                std::cout << "[Info] Resolved HuggingFace cache path:\n";
                std::cout << "       " << path << "\n";
                std::cout << "    -> " << resolved << "\n\n";
                return resolved;
            }
        }
        
        return path;
    };
    
    args.model_path = resolve_hf_cache_path(args.model_path);

    apply_generation_defaults(args, args.model_path);
    if (args.check_mode) {
        args.temperature = 0.0f;
        args.top_p = 1.0f;
        args.top_k = 1;
        if (args.dump_dir == "debug") {
            args.dump_dir = (fs::path("debug") / ("check_" + model_name_from_path(args.model_path))).string();
        }
    }
    
    // 加载模型配置
    std::string config_path = args.model_path + "/config.json";
    if (!fs::exists(config_path)) {
        std::cerr << "Error: config.json not found in " << args.model_path << "\n";
        return 1;
    }
    
    ember::ModelConfig model_config;
    try {
        model_config = ember::parse_model_config(config_path);
    } catch (const std::exception& e) {
        std::cerr << "Error parsing config: " << e.what() << "\n";
        return 1;
    }
    
    std::cout << "[Model] " << model_config.model_type << "\n";
    std::cout << "  Vocab size: " << model_config.vocab_size << "\n";
    std::cout << "  Hidden size: " << model_config.hidden_size << "\n";
    std::cout << "  Layers: " << model_config.num_layers << "\n";
    std::cout << "  Heads: " << model_config.num_heads << " (KV: " << model_config.num_kv_heads << ")\n";
    std::cout << "  Intermediate: " << model_config.intermediate_size << "\n";
    std::cout << "\n";
    
    // 创建设备映射
    ember::DeviceMap device_map = ember::DeviceMap::single_device(model_config.num_layers, args.devices[0]);
    if (args.devices.size() > 1) {
        // 多卡：收集显存信息
        std::vector<size_t> gpu_memory;
        for (int dev : args.devices) {
            auto info = ember::cuda::get_gpu_info(dev);
            gpu_memory.push_back(static_cast<size_t>(info.free_memory * args.memory_fraction));
        }
        device_map = ember::DeviceMap::auto_map(model_config, gpu_memory, args.ctx_size, 1);
    }
    
    // 创建运行时
    auto runtime = ember::RuntimeFactory::create_cuda();
    if (!runtime || !runtime->available()) {
        std::cerr << "Error: CUDA runtime not available\n";
        return 1;
    }
    
    // 显存估算
    auto mem_est = runtime->estimate_memory(model_config, args.ctx_size, 1);
    std::cout << mem_est.to_string() << "\n";
    
    // 创建运行时配置
    ember::RuntimeConfig runtime_config;
    runtime_config.max_ctx_len = args.ctx_size;
    runtime_config.temperature = args.temperature;
    runtime_config.top_p = args.top_p;
    runtime_config.top_k = args.top_k;
    runtime_config.repetition_penalty = args.repeat_penalty;
    runtime_config.presence_penalty = args.presence_penalty;
    runtime_config.frequency_penalty = args.frequency_penalty;
    runtime_config.no_repeat_ngram_size = args.no_repeat_ngram;
    runtime_config.device_ids = args.devices;
    runtime_config.memory_fraction = args.memory_fraction;
    runtime_config.kv_cache_dtype = ember::dtype_from_string(model_config.torch_dtype);
    if (runtime_config.kv_cache_dtype == ember::DType::UNKNOWN) {
        runtime_config.kv_cache_dtype = ember::DType::F16;
    }
    runtime_config.check_correctness = args.check_mode;
    runtime_config.dump_layer = args.dump_layer;
    runtime_config.dump_dir = args.dump_dir;

    // 加载模型并初始化会话/KV cache
    std::cout << "[Loading] Model from " << args.model_path << "...\n";
    auto start_load = std::chrono::high_resolution_clock::now();

    ember::RuntimeSetup runtime_setup;
    ember::Error err = ember::load_runtime(*runtime, args.model_path, model_config, device_map, runtime_setup);
    if (err) {
        std::cerr << "Error loading model: " << err.message() << "\n";
        return 1;
    }

    err = ember::init_session_and_kv(*runtime, model_config, runtime_config, runtime_setup);
    if (err) {
        std::cerr << "Error allocating KV cache: " << err.message() << "\n";
        return 1;
    }

    auto end_load = std::chrono::high_resolution_clock::now();
    auto load_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_load - start_load).count();
    std::cout << "[Loaded] in " << load_time << " ms\n\n";

    ember::Session& session = runtime_setup.session;

    ember::PhaseAwareScheduler scheduler(
        *runtime,
        ember::PhaseAwareSchedulerConfig{
            .prefill_chunk_len = args.prefill_chunk_len,
            .prefill_overlap = args.prefill_overlap,
            .decode_batch_size = 1,
        }
    );
    
    // 创建采样器
    ember::Sampler sampler(args.temperature, args.top_k, args.top_p);
    
    // 加载 tokenizer
    ember::HFTokenizer tokenizer;
    err = tokenizer.load(args.model_path);
    if (err) {
        std::cerr << "Warning: Failed to load tokenizer: " << err.message() << "\n";
        std::cerr << "Using simple tokenizer (output will show token IDs)\n";
    }
    
    // Chat 模板（适用于 Instruct 模型）
    std::string chat_template = load_chat_template(args.model_path);
    bool use_chat_template = !chat_template.empty() && is_chat_model(args.model_path);
    std::vector<int> eos_ids = load_eos_token_ids(args.model_path);
    if (eos_ids.empty()) {
        eos_ids.push_back(tokenizer.eos_token_id());
    }
    std::unordered_set<int> eos_set(eos_ids.begin(), eos_ids.end());
    
    auto format_prompt = [&](const std::string& prompt) -> std::string {
        if (!use_chat_template) return prompt;
        if (prompt.find("<|im_start|>") != std::string::npos) return prompt;
        return build_chat_prompt(prompt);
    };

    auto run_check = [&](const std::string& prompt) {
        std::string formatted_prompt = format_prompt(prompt);
        std::cout << "\n[Prompt] " << formatted_prompt << "\n\n";
        
        session.reset();
        
        std::vector<int> tokens = tokenizer.encode(formatted_prompt);
        std::cout << "[Tokens] " << tokens.size() << " input tokens\n";
        if (tokens.empty()) {
            std::cout << "[Warning] Empty prompt, using BOS token\n";
            tokens.push_back(tokenizer.bos_token_id());
        }
        
        std::cout << "[Prefill] Processing prompt...\n";
        auto start_prefill = std::chrono::high_resolution_clock::now();
        
        std::vector<float> logits;
        if (args.phase_aware) {
            err = scheduler.prefill_with_logits(tokens, session, logits);
        } else {
            err = runtime->prefill_with_logits(tokens, session, logits);
        }
        if (err) {
            std::cerr << "Prefill error: " << err.message() << "\n";
            return;
        }
        
        auto end_prefill = std::chrono::high_resolution_clock::now();
        auto prefill_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_prefill - start_prefill).count();
        std::cout << "[Prefill] Done in " << prefill_time << " ms "
                  << "(" << (tokens.size() * 1000.0 / prefill_time) << " tok/s)\n";
        
        fs::path out_dir = runtime_config.dump_dir;
        std::error_code ec;
        fs::create_directories(out_dir, ec);
        if (ec) {
            std::cerr << "[Check] Failed to create output dir: " << out_dir.string() << "\n";
            return;
        }
        
        bool ok = true;
        ok &= write_text_file(out_dir / "prompt.txt", formatted_prompt);
        ok &= write_ints_file(out_dir / "tokens.txt", tokens);
        ok &= write_f32_binary(out_dir / "logits.bin", logits);
        ok &= write_check_meta(out_dir / "meta.json", args.model_path, formatted_prompt,
                               model_config.vocab_size, model_config.hidden_size,
                               model_config.num_layers, static_cast<int>(tokens.size()));
        
        if (!ok) {
            std::cerr << "[Check] Failed to write debug outputs in " << out_dir.string() << "\n";
            return;
        }
        
        std::cout << "[Check] Saved outputs to " << out_dir.string() << "\n";
        if (args.verbose && !logits.empty()) {
            int top_id = static_cast<int>(std::distance(logits.begin(),
                        std::max_element(logits.begin(), logits.end())));
            float max_logit = logits[top_id];
            std::cout << "[Check] Top1 id=" << top_id << " logit=" << max_logit << "\n";
        }
    };
    
    // 交互模式或单次生成
    auto run_generation = [&](const std::string& prompt) {
        std::string formatted_prompt = format_prompt(prompt);
        std::cout << "\n[Prompt] " << formatted_prompt << "\n\n";
        
        // 重置会话
        session.reset();
        
        // 编码 prompt
        std::vector<int> tokens = tokenizer.encode(formatted_prompt);
        std::cout << "[Tokens] " << tokens.size() << " input tokens";
        if (tokens.size() > 0 && tokens.size() <= 10) {
            std::cout << " [";
            for (size_t i = 0; i < tokens.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << tokens[i];
            }
            std::cout << "]";
        }
        std::cout << "\n";
        
        if (tokens.empty()) {
            std::cout << "[Warning] Empty prompt, using BOS token\n";
            tokens.push_back(tokenizer.bos_token_id());
        }
        
        // Prefill 并获取第一个 token 的 logits
        std::cout << "[Prefill] Processing prompt...\n";
        auto start_prefill = std::chrono::high_resolution_clock::now();
        
        std::vector<float> logits;
        if (args.phase_aware) {
            err = scheduler.prefill_with_logits(tokens, session, logits);
        } else {
            err = runtime->prefill_with_logits(tokens, session, logits);
        }
        if (err) {
            std::cerr << "Prefill error: " << err.message() << "\n";
            return;
        }
        
        auto end_prefill = std::chrono::high_resolution_clock::now();
        auto prefill_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_prefill - start_prefill).count();
        std::cout << "[Prefill] Done in " << prefill_time << " ms "
                  << "(" << (tokens.size() * 1000.0 / prefill_time) << " tok/s)\n\n";
        
        // 生成
        std::cout << "[Generating]\n";
        std::cout << "---\n";
        
        std::vector<int> generated;
        std::vector<int> history = tokens;
        
        auto start_gen = std::chrono::high_resolution_clock::now();
        
        // 从 prefill logits 采样第一个 token
        int next_token = sampler.sample(logits, runtime_config, history);
        generated.push_back(next_token);
        history.push_back(next_token);
        
        // 调试输出
        if (args.verbose) {
            float max_logit = *std::max_element(logits.begin(), logits.end());
            std::cout << "[Debug] First token=" << next_token << " max_logit=" << max_logit << "\n";
        }
        
        // 输出第一个 token
        if (!eos_set.count(next_token)) {
            std::string token_str = tokenizer.decode({next_token});
            std::cout << token_str << std::flush;
        }
        
        // 继续生成剩余 tokens
        for (int i = 1; i < args.n_predict && session.can_continue() && !eos_set.count(next_token); ++i) {
            // Decode
            err = runtime->decode(next_token, session, logits);
            if (err) {
                std::cerr << "\nDecode error: " << err.message() << "\n";
                break;
            }
            
            // Sample
            next_token = sampler.sample(logits, runtime_config, history);
            generated.push_back(next_token);
            history.push_back(next_token);
            
            // 调试输出（前几个 token）
            if (args.verbose && generated.size() <= 5) {
                float max_logit = *std::max_element(logits.begin(), logits.end());
                std::cout << "\n[Debug] token=" << next_token << " max_logit=" << max_logit;
            }
            
            // 检查 EOS
            if (eos_set.count(next_token)) {
                break;
            }
            
            // 输出 token
            std::string token_str = tokenizer.decode({next_token});
            std::cout << token_str << std::flush;
        }
        
        auto end_gen = std::chrono::high_resolution_clock::now();
        auto gen_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_gen - start_gen).count();
        
        std::cout << "\n---\n";
        std::cout << "[Generated] " << generated.size() << " tokens in " << gen_time << " ms "
                  << "(" << (generated.size() * 1000.0 / gen_time) << " tok/s)\n";
    };
    
    if (args.check_mode) {
        std::string prompt = args.prompt.empty() ? "Hello, my name is" : args.prompt;
        run_check(prompt);
    } else if (args.interactive) {
        std::cout << "Interactive mode. Type 'quit' to exit.\n";
        while (true) {
            std::cout << "\n> ";
            std::string line;
            if (!std::getline(std::cin, line) || line == "quit") {
                break;
            }
            if (line.empty()) continue;
            run_generation(line);
        }
    } else if (!args.prompt.empty()) {
        run_generation(args.prompt);
    } else {
        // 默认测试 prompt
        run_generation("Hello, my name is");
    }
    
    // 清理
    runtime->free_kv_cache(session);
    runtime->unload();
    
    std::cout << "\n[Done]\n";
    return 0;
}

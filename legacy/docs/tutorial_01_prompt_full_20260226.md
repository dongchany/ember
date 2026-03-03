# Tutorial #1 Prompt (Full, With Code + Reports)

> 生成日期：2026-02-26
> 用途：整份复制到新 chat 作为写作输入
> 标注：大文件（如 cuda_runtime.cpp）按“该篇相关段落”摘取，避免上下文爆炸

---

## 0) 系统 Prompt（先贴这段）

```text
我正在为我的开源项目 Ember 写一个系列教程。请帮我写第 1 篇。

## 项目简介
Ember (https://github.com/dongchany/ember) 是一个从零手写的 Qwen3 CUDA 推理引擎，
纯 C++ + CUDA，不依赖 ggml/llama.cpp。支持消费级多 GPU Pipeline Parallel（如双卡 RTX 3080Ti）。
除了推理，Ember 还支持完整的 RL 训练闭环：多候选 Rollout → Verifier/Reward → LoRA 热更新 →
Cache 策略复用，实现了统一后端（推理和训练共享同份权重），相比双栈方案节省 50% 显存。

## 项目 5 层结构
Layer 1: 推理引擎（CUDA kernels, Transformer forward, Pipeline Parallel）
Layer 2: Rollout 能力（多候选、logprobs、stop sequences）
Layer 3: LoRA 热更新 + Cache 策略（UpdateLocality / Prefix / Periodic / Hybrid）
Layer 4: 验证器 + Reward（Extraction / SQL verifier，字段级打分）
Layer 5: 训练闭环（SFT → Best-of-N → DPO → GRPO 可选）+ 统一后端 vs 双栈

## 写作硬性要求
1. 目标读者：想了解 LLM 内部原理的开发者，数学基础较弱也能看懂
2. 数学四步法：直觉 → 小例子手算 → 公式 → 对应 CUDA/训练代码
3. 语言：中文为主，术语和代码注释保留英文
4. 必须引用我提供的真实源码与真实报告，不得编造实验数字
5. 每篇开头必须写：源文件路径、前置知识、下一篇链接
6. 每篇结尾自然放 GitHub 链接：https://github.com/dongchany/ember
7. 风格：友好、像学长讲解，不要居高临下
8. 不要只列 bullet；以叙述为主

## 输出质量要求（必须遵守）
- 你只能使用我提供的“完整代码片段”和“完整报告片段”作为事实来源
- 所有结论都要标注来自哪个文件
- 任何数字都要能在报告中定位到
- 如果某结论缺证据，明确写“当前资料不足”

## 数学深度加严（额外要求）
- 在不影响可读性的前提下，尽量给出更详细的数学推导
- 对每个关键公式都解释“它在数值稳定性/并行实现上的意义”
- 允许在附录给出更完整推导（正文保持循序渐进）
```

---

## 1) 写作任务

```text
请写第 1 篇：一个 Token 的一生（全局数据流概览）。

## 本篇必须引用的代码与报告都在本消息里（不要再让我补充路径）

- 强调 prefill vs decode 的差异
- 给一张“数据如何在模块间流动”的 ASCII 图
- 对关键路径给出“代码定位”小节（函数名 + 文件名）
```

---

## 2) 代码上下文（完整/相关段落）

### File: apps/ember_cli/main.cpp

````cpp
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
                             int hidden_size, int num_layers, int token_count,
                             const std::string& adapter_path, float lora_scale) {
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
    out << "  \"token_count\": " << token_count;
    if (!adapter_path.empty()) {
        out << ",\n";
        out << "  \"adapter_path\": \"" << adapter_path << "\",\n";
        out << "  \"lora_scale\": " << lora_scale << "\n";
    } else {
        out << "\n";
    }
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

    if (!args.adapter_path.empty()) {
        auto* cuda_rt = dynamic_cast<ember::cuda::CudaRuntime*>(runtime.get());
        if (cuda_rt == nullptr) {
            std::cerr << "Error: --adapter is only supported by CUDA runtime\n";
            return 1;
        }
        ember::cuda::CudaRuntime::LoraApplyStats st{};
        err = cuda_rt->apply_lora_adapter(args.adapter_path, args.lora_scale, false, &st);
        if (err) {
            std::cerr << "Error applying LoRA adapter: " << err.message() << "\n";
            return 1;
        }
        std::cout << "[LoRA] Applied adapter: " << args.adapter_path
                  << " (scale=" << args.lora_scale
                  << ", effective_scale=" << st.scale_used
                  << ", updated=" << st.updated_matrices
                  << ", skipped=" << st.skipped_matrices
                  << ", wall_ms=" << st.wall_ms << ")\n";
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
                               model_config.num_layers, static_cast<int>(tokens.size()),
                               args.adapter_path, args.lora_scale);
        
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

````

### File: core/session.h

````h
#pragma once

#include "types.h"
#include "tensor.h"
#include "config.h"
#include <vector>
#include <memory>

namespace ember {

// 单层的 KV Cache
struct LayerKVCache {
    Tensor key_cache;    // [batch, num_kv_heads, max_ctx, head_dim]
    Tensor value_cache;  // [batch, num_kv_heads, max_ctx, head_dim]
    int device_id = 0;
    
    bool allocated() const { return key_cache.data != nullptr; }
};

// KV Cache 管理器
class KVCache {
public:
    KVCache() = default;
    
    // 初始化缓存（不分配内存，只设置元数据）
    void init(int num_layers, int batch_size, int max_ctx_len, 
              int num_kv_heads, int head_dim, DType dtype) {
        num_layers_ = num_layers;
        batch_size_ = batch_size;
        max_ctx_len_ = max_ctx_len;
        num_kv_heads_ = num_kv_heads;
        head_dim_ = head_dim;
        dtype_ = dtype;
        
        layer_caches_.resize(num_layers);
        for (int i = 0; i < num_layers; ++i) {
            layer_caches_[i].key_cache.shape = {batch_size, num_kv_heads, max_ctx_len, head_dim};
            layer_caches_[i].key_cache.dtype = dtype;
            layer_caches_[i].value_cache.shape = {batch_size, num_kv_heads, max_ctx_len, head_dim};
            layer_caches_[i].value_cache.dtype = dtype;
        }
    }
    
    // 获取指定层的 KV cache
    LayerKVCache& layer(int i) { return layer_caches_[i]; }
    const LayerKVCache& layer(int i) const { return layer_caches_[i]; }
    
    // 设置层的缓存指针
    void set_layer_data(int layer_idx, void* key_data, void* value_data, int device_id) {
        layer_caches_[layer_idx].key_cache.data = key_data;
        layer_caches_[layer_idx].key_cache.device_id = device_id;
        layer_caches_[layer_idx].value_cache.data = value_data;
        layer_caches_[layer_idx].value_cache.device_id = device_id;
        layer_caches_[layer_idx].device_id = device_id;
    }
    
    // 计算单层缓存大小
    size_t layer_size_bytes() const {
        return static_cast<size_t>(batch_size_) * num_kv_heads_ * max_ctx_len_ * head_dim_ 
               * dtype_size(dtype_) * 2;  // K 和 V
    }
    
    // 计算总缓存大小
    size_t total_size_bytes() const {
        return layer_size_bytes() * num_layers_;
    }
    
    int num_layers() const { return num_layers_; }
    int max_ctx_len() const { return max_ctx_len_; }
    DType dtype() const { return dtype_; }

private:
    int num_layers_ = 0;
    int batch_size_ = 1;
    int max_ctx_len_ = 0;
    int num_kv_heads_ = 0;
    int head_dim_ = 0;
    DType dtype_ = DType::F16;
    
    std::vector<LayerKVCache> layer_caches_;
};

// 推理会话状态
class Session {
public:
    Session() = default;
    
    // 初始化会话
    void init(const ModelConfig& model_config, const RuntimeConfig& runtime_config) {
        model_config_ = model_config;
        runtime_config_ = runtime_config;
        
        kv_cache_.init(
            model_config.num_layers,
            runtime_config.batch_size,
            runtime_config.max_ctx_len,
            model_config.num_kv_heads,
            model_config.head_dim,
            runtime_config.kv_cache_dtype
        );
        
        cur_pos_by_batch_.assign(static_cast<size_t>(runtime_config.batch_size), 0);
    }
    
    // 当前位置（已处理的 token 数）
    int cur_pos() const { return cur_pos_by_batch_.empty() ? 0 : cur_pos_by_batch_[0]; }
    int cur_pos(int slot) const { return cur_pos_by_batch_.at(static_cast<size_t>(slot)); }

    // Backward-compatible: for uniform batches, set/advance all slots.
    void set_cur_pos(int pos) {
        for (int& p : cur_pos_by_batch_) p = pos;
    }
    void set_cur_pos(int slot, int pos) { cur_pos_by_batch_.at(static_cast<size_t>(slot)) = pos; }

    void advance(int n = 1) {
        for (int& p : cur_pos_by_batch_) {
            if (p >= 0) p += n;
        }
    }
    void advance_slot(int slot, int n = 1) { cur_pos_by_batch_.at(static_cast<size_t>(slot)) += n; }

    void set_inactive(int slot) { cur_pos_by_batch_.at(static_cast<size_t>(slot)) = -1; }
    bool active(int slot) const { return cur_pos(slot) >= 0; }
    
    // 剩余可用上下文
    int remaining_ctx() const { return runtime_config_.max_ctx_len - cur_pos(); }
    int remaining_ctx(int slot) const { return runtime_config_.max_ctx_len - cur_pos(slot); }
    
    // 是否还能继续生成
    bool can_continue() const { return cur_pos() < runtime_config_.max_ctx_len; }
    bool can_continue(int slot) const { return cur_pos(slot) < runtime_config_.max_ctx_len; }
    
    // 重置会话（清除 KV cache 内容，重置位置）
    void reset() {
        for (int& p : cur_pos_by_batch_) p = 0;
        // 注意：不释放内存，只重置位置
    }
    
    // 获取 KV cache
    KVCache& kv_cache() { return kv_cache_; }
    const KVCache& kv_cache() const { return kv_cache_; }
    
    // 获取配置
    const ModelConfig& model_config() const { return model_config_; }
    const RuntimeConfig& runtime_config() const { return runtime_config_; }
    
    // 生成的 token 序列
    std::vector<int>& generated_tokens() { return generated_tokens_; }
    const std::vector<int>& generated_tokens() const { return generated_tokens_; }

private:
    ModelConfig model_config_;
    RuntimeConfig runtime_config_;
    KVCache kv_cache_;
    std::vector<int> cur_pos_by_batch_;
    std::vector<int> generated_tokens_;
};

}  // namespace ember

````

### File: core/sampler.h

````h
#pragma once

#include "config.h"
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_set>
#include <unordered_map>

namespace ember {

// 采样器
class Sampler {
public:
    Sampler() : rng_(std::random_device{}()), 
                temperature_(1.0f), top_k_(0), top_p_(1.0f) {}
    
    Sampler(float temperature, int top_k, float top_p)
        : rng_(std::random_device{}()),
          temperature_(temperature), top_k_(top_k), top_p_(top_p) {}
    
    // 设置随机种子
    void set_seed(uint64_t seed) {
        rng_.seed(seed);
    }
    
    // 从 logits 采样下一个 token（使用内部参数）
    int sample(const std::vector<float>& logits) {
        std::vector<float> probs = logits;
        
        // 应用 temperature
        if (temperature_ > 0 && temperature_ != 1.0f) {
            for (float& p : probs) {
                p /= temperature_;
            }
        }
        
        // Softmax
        softmax(probs);
        
        // 应用 top-k
        if (top_k_ > 0 && top_k_ < static_cast<int>(probs.size())) {
            top_k_filter(probs, top_k_);
        }
        
        // 应用 top-p (nucleus sampling)
        if (top_p_ > 0 && top_p_ < 1.0f) {
            top_p_filter(probs, top_p_);
        }
        
        // 根据 temperature 决定采样方式
        if (temperature_ <= 0) {
            // Greedy: 返回最大概率的 token
            return static_cast<int>(std::distance(probs.begin(), 
                   std::max_element(probs.begin(), probs.end())));
        } else {
            // 随机采样
            return categorical_sample(probs);
        }
    }
    
    // 从 logits 采样下一个 token（使用 RuntimeConfig）
    int sample(const std::vector<float>& logits, const RuntimeConfig& config) {
        static const std::vector<int> empty_history;
        return sample(logits, config, empty_history);
    }

    // 从 logits 采样下一个 token（使用 RuntimeConfig + 历史 tokens）
    int sample(const std::vector<float>& logits,
               const RuntimeConfig& config,
               const std::vector<int>& history) {
        std::vector<float> probs = logits;
        
        if (!history.empty()) {
            std::unordered_map<int, int> counts;
            counts.reserve(history.size());
            for (int token : history) {
                if (token >= 0 && token < static_cast<int>(probs.size())) {
                    ++counts[token];
                }
            }

            // Repetition penalty (pre-softmax)
            if (config.repetition_penalty > 1.0f) {
                for (const auto& item : counts) {
                    float& logit = probs[item.first];
                    if (logit < 0.0f) {
                        logit *= config.repetition_penalty;
                    } else {
                        logit /= config.repetition_penalty;
                    }
                }
            }

            // Presence/frequency penalties (pre-softmax)
            if (config.presence_penalty != 0.0f || config.frequency_penalty != 0.0f) {
                for (const auto& item : counts) {
                    float& logit = probs[item.first];
                    if (config.presence_penalty != 0.0f) {
                        logit -= config.presence_penalty;
                    }
                    if (config.frequency_penalty != 0.0f) {
                        logit -= config.frequency_penalty * static_cast<float>(item.second);
                    }
                }
            }

            // No-repeat ngram: ban tokens that would repeat the last n-gram
            int ngram = config.no_repeat_ngram_size;
            if (ngram > 1 && history.size() >= static_cast<size_t>(ngram)) {
                size_t prefix_start = history.size() - static_cast<size_t>(ngram - 1);
                for (size_t i = 0; i + static_cast<size_t>(ngram) <= history.size(); ++i) {
                    bool match = true;
                    for (int j = 0; j < ngram - 1; ++j) {
                        if (history[i + static_cast<size_t>(j)] !=
                            history[prefix_start + static_cast<size_t>(j)]) {
                            match = false;
                            break;
                        }
                    }
                    if (match) {
                        int banned = history[i + static_cast<size_t>(ngram - 1)];
                        if (banned >= 0 && banned < static_cast<int>(probs.size())) {
                            probs[banned] = -1e9f;
                        }
                    }
                }
            }
        }

        // 应用 temperature
        if (config.temperature > 0 && config.temperature != 1.0f) {
            for (float& p : probs) {
                p /= config.temperature;
            }
        }
        
        // Softmax
        softmax(probs);
        
        // 应用 top-k
        if (config.top_k > 0 && config.top_k < static_cast<int>(probs.size())) {
            top_k_filter(probs, config.top_k);
        }
        
        // 应用 top-p (nucleus sampling)
        if (config.top_p > 0 && config.top_p < 1.0f) {
            top_p_filter(probs, config.top_p);
        }
        
        // 根据 temperature 决定采样方式
        if (config.temperature <= 0) {
            // Greedy: 返回最大概率的 token
            return static_cast<int>(std::distance(probs.begin(), 
                   std::max_element(probs.begin(), probs.end())));
        } else {
            // 随机采样
            return categorical_sample(probs);
        }
    }
    
    // Greedy 采样（直接返回最大 logit 的索引）
    static int argmax(const std::vector<float>& logits) {
        return static_cast<int>(std::distance(logits.begin(), 
               std::max_element(logits.begin(), logits.end())));
    }
    
private:
    std::mt19937_64 rng_;
    float temperature_;
    int top_k_;
    float top_p_;
    
    // Softmax（原地修改）
    static void softmax(std::vector<float>& x) {
        float max_val = *std::max_element(x.begin(), x.end());
        float sum = 0;
        for (float& v : x) {
            v = std::exp(v - max_val);
            sum += v;
        }
        for (float& v : x) {
            v /= sum;
        }
    }
    
    // Top-K 过滤
    static void top_k_filter(std::vector<float>& probs, int k) {
        if (k >= static_cast<int>(probs.size())) return;
        if (k <= 0) return;
        
        // 找到第 k 大的值
        std::vector<float> sorted = probs;
        std::nth_element(sorted.begin(), sorted.begin() + (k - 1), sorted.end(), std::greater<float>());
        float threshold = sorted[k - 1];
        
        // 过滤掉小于阈值的
        for (float& p : probs) {
            if (p < threshold) p = 0;
        }
        
        // 重新归一化
        float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
        if (sum > 0) {
            for (float& p : probs) p /= sum;
        }
    }
    
    // Top-P (Nucleus) 过滤
    static void top_p_filter(std::vector<float>& probs, float p) {
        // 创建索引-概率对并排序
        std::vector<std::pair<int, float>> indexed(probs.size());
        for (size_t i = 0; i < probs.size(); ++i) {
            indexed[i] = {static_cast<int>(i), probs[i]};
        }
        std::sort(indexed.begin(), indexed.end(), 
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // 累积概率，找到截断点
        float cumsum = 0;
        size_t cutoff = indexed.size();
        for (size_t i = 0; i < indexed.size(); ++i) {
            cumsum += indexed[i].second;
            if (cumsum >= p) {
                cutoff = i + 1;
                break;
            }
        }
        
        // 过滤
        std::vector<float> new_probs(probs.size(), 0);
        for (size_t i = 0; i < cutoff; ++i) {
            new_probs[indexed[i].first] = indexed[i].second;
        }
        
        // 重新归一化
        float sum = std::accumulate(new_probs.begin(), new_probs.end(), 0.0f);
        if (sum > 0) {
            for (float& p : new_probs) p /= sum;
        }
        
        probs = std::move(new_probs);
    }
    
    // 分类采样
    int categorical_sample(const std::vector<float>& probs) {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float r = dist(rng_);
        
        float cumsum = 0;
        for (size_t i = 0; i < probs.size(); ++i) {
            cumsum += probs[i];
            if (r < cumsum) {
                return static_cast<int>(i);
            }
        }
        
        // 回退到最后一个有效 token
        for (int i = static_cast<int>(probs.size()) - 1; i >= 0; --i) {
            if (probs[i] > 0) return i;
        }
        return 0;
    }
};

}  // namespace ember

````

### File: backends/cuda/cuda_runtime.cpp (prefill_with_logits)

````cpp
        err = forward_layer(layer_idx, batch_size, seq_len, start_pos, session, /*skip_input_copy=*/false);
        if (err) return err;
    }
    
    // 同步所有设备
    for (int dev = 0; dev < device_map_.num_devices; ++dev) {
        cuda_sync(dev);
    }

    EMBER_RETURN_IF_ERROR(finalize_stage_profile_(/*sync_all_devices=*/false,
                                                  /*include_final_norm=*/false,
                                                  /*include_lm_head=*/false));
    
    // 更新位置
    session.set_cur_pos(start_pos + seq_len);
    
    return Error::success();
}

// 带 logits 返回的 prefill (用于立即采样)
Error CudaRuntime::prefill_with_logits(const std::vector<int>& tokens, Session& session, std::vector<float>& logits) {
    Error err = prefill(tokens, session);
    if (err) return err;
    
    // Final norm 和 LM head 来计算最后一个位置的 logits
    int batch_size = 1;
    int seq_len = static_cast<int>(tokens.size());
    
    err = forward_final_norm(batch_size, seq_len, session);
    if (err) return err;
    
    err = forward_lm_head(batch_size, seq_len);
    if (err) return err;
    
    // 拷贝 logits
    int lm_device = device_map_.lm_head_device;
    cuda_sync(lm_device);
    
    logits.resize(config_.vocab_size);
    auto d2h_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(logits.data(), activations_[lm_device].logits, 
                          config_.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    auto d2h_end = std::chrono::high_resolution_clock::now();
    add_stage_profile_d2h_ms_(std::chrono::duration<double, std::milli>(d2h_end - d2h_start).count());
    
    return Error::success();
}

Error CudaRuntime::decode(int last_token, Session& session, std::vector<float>& logits) {
    if (!loaded_) {
        return Error(ErrorCode::MODEL_NOT_LOADED, "Model not loaded");
    }

    if (profile_layers_) {
        last_layer_profile_ms_.assign(static_cast<size_t>(config_.num_layers), 0.0f);
    }
    EMBER_RETURN_IF_ERROR(decode_single_forward_to_lm_head_(last_token, session));
    
    // 同步并拷贝 logits 回 CPU
    int lm_device = device_map_.lm_head_device;
    if (profile_stages_) {
        for (int dev = 0; dev < device_map_.num_devices; ++dev) {
            cuda_sync(dev);
        }
    } else {
    cuda_sync(lm_device);
    }
    
    logits.resize(config_.vocab_size);
    auto d2h_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(logits.data(), activations_[lm_device].logits, 
                          config_.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    auto d2h_end = std::chrono::high_resolution_clock::now();
    add_stage_profile_d2h_ms_(std::chrono::duration<double, std::milli>(d2h_end - d2h_start).count());

    EMBER_RETURN_IF_ERROR(finalize_stage_profile_(/*sync_all_devices=*/false,
                                                  /*include_final_norm=*/true,
                                                  /*include_lm_head=*/true));
    
    // 更新位置
    session.advance(1);
    
    return Error::success();
}

Error CudaRuntime::forward_layer(int layer_idx,
                                 int batch_size,
                                 int seq_len,
                                 int start_pos,
                                 Session& session,
                                 bool skip_input_copy,
                                 const int* start_pos_by_batch) {
    int device_id = device_map_.layer_to_device[layer_idx];
    auto& act = activations_[device_id];
    auto& layer = weights_.layers[layer_idx];
    auto stream = streams_[device_id];
    auto& cublas = cublas_handles_[device_id];
    
    CUDA_CHECK(cudaSetDevice(device_id));
    cublasSetStream(cublas.get(), stream);
    
    DType compute_dtype = weights_.dtype;
    cudaDataType_t cuda_dtype = to_cuda_dtype(compute_dtype);
    size_t hidden_size = config_.hidden_size;
    size_t num_heads = config_.num_heads;
    size_t num_kv_heads = config_.num_kv_heads;
    size_t head_dim = config_.head_dim;
    size_t intermediate_size = config_.intermediate_size;
    size_t elem_size = dtype_size(compute_dtype);
    
    // 如果上一层在不同设备，需要拷贝 hidden_states
    if (layer_idx > 0) {
        int prev_device = device_map_.layer_to_device[layer_idx - 1];
        if (prev_device != device_id) {
            if (!skip_input_copy) {
            size_t size = batch_size * seq_len * hidden_size * elem_size;
            auto t0 = std::chrono::high_resolution_clock::now();
            Error err = copy_bytes_peer_or_staged(act.hidden_states, device_id,
                                                  activations_[prev_device].hidden_states, prev_device,
                                                  size);
            if (err) return err;
            if (profile_stages_) {
                auto t1 = std::chrono::high_resolution_clock::now();
                last_stage_profile_ms_.p2p_ms += static_cast<float>(
                    std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            }
        }
    }

    if (profile_layers_ && static_cast<size_t>(layer_idx) < last_layer_profile_ms_.size()) {
        auto& ev = profile_events_[device_id];
        CUDA_CHECK(cudaEventRecord(ev.start, stream));
    }

    if (session.runtime_config().check_correctness) {
        int target = session.runtime_config().dump_layer;
        if (target < 0 || target == layer_idx) {
            Error err = dump_last_row(session.runtime_config().dump_dir,
                                      "layer_" + std::to_string(layer_idx) + "_layer_input",
                                      device_id, act.hidden_states,
                                      seq_len, hidden_size, compute_dtype, stream);
            if (err) return err;
        }
    }
    
    // =====================================================================
    // Input LayerNorm
    // =====================================================================
    if (profile_stages_) {
        auto& sev = layer_stage_events_[static_cast<size_t>(layer_idx)];
        CUDA_CHECK(cudaEventRecord(sev.in_norm_start, stream));
    }
    if (compute_dtype == DType::BF16) {
        kernels::rms_norm_bf16(
            static_cast<__nv_bfloat16*>(act.norm_out),
            static_cast<const __nv_bfloat16*>(act.hidden_states),
            static_cast<const __nv_bfloat16*>(layer.input_layernorm_weight),
            batch_size, seq_len, hidden_size,
            config_.rms_norm_eps,
            stream
        );
    } else {
        kernels::rms_norm_f16(
            static_cast<half*>(act.norm_out),
            static_cast<const half*>(act.hidden_states),
            static_cast<const half*>(layer.input_layernorm_weight),
            batch_size, seq_len, hidden_size,
            config_.rms_norm_eps,
            stream
        );
    }
    if (profile_stages_) {
        auto& sev = layer_stage_events_[static_cast<size_t>(layer_idx)];
        CUDA_CHECK(cudaEventRecord(sev.in_norm_end, stream));
        CUDA_CHECK(cudaEventRecord(sev.attn_start, stream));
    }
    
    // =====================================================================
    // QKV Projection: norm_out @ W_q/k/v -> q/k/v_proj_out
    // 权重布局: [out_features, in_features] (row-major, Qwen3 safetensors格式)
    // 输入: [batch*seq, hidden_size]
    // 输出: [batch*seq, out_features]
    // =====================================================================
    const float alpha_one = 1.0f;
    const float beta_zero = 0.0f;
    
    int M = batch_size * seq_len;  // 批次*序列长度
    
    // Q projection: [M, hidden] @ [hidden, num_heads*head_dim]^T = [M, num_heads*head_dim]
    CUBLAS_CHECK(cublasGemmEx(
        cublas.get(),
        CUBLAS_OP_T, CUBLAS_OP_N,  // W^T @ X
        num_heads * head_dim, M, hidden_size,
        &alpha_one,
        layer.q_proj_weight, cuda_dtype, hidden_size,  // W: [num_heads*head_dim, hidden] -> W^T
        act.norm_out, cuda_dtype, hidden_size,         // X: [M, hidden]
        &beta_zero,
        act.q_proj_out, cuda_dtype, num_heads * head_dim,  // Y: [M, num_heads*head_dim]
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
    
    // K projection
    CUBLAS_CHECK(cublasGemmEx(
        cublas.get(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_kv_heads * head_dim, M, hidden_size,
        &alpha_one,
        layer.k_proj_weight, cuda_dtype, hidden_size,
        act.norm_out, cuda_dtype, hidden_size,
        &beta_zero,
        act.k_proj_out, cuda_dtype, num_kv_heads * head_dim,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
    
    // V projection
    CUBLAS_CHECK(cublasGemmEx(
        cublas.get(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_kv_heads * head_dim, M, hidden_size,
        &alpha_one,
        layer.v_proj_weight, cuda_dtype, hidden_size,
        act.norm_out, cuda_dtype, hidden_size,
        &beta_zero,
        act.v_proj_out, cuda_dtype, num_kv_heads * head_dim,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));

    // =====================================================================
    // Q/K per-head RMSNorm (Qwen3)
    // =====================================================================
    int q_rows = batch_size * seq_len * static_cast<int>(num_heads);
    int k_rows = batch_size * seq_len * static_cast<int>(num_kv_heads);
    if (compute_dtype == DType::BF16) {
        kernels::rms_norm_bf16(
            static_cast<__nv_bfloat16*>(act.q_proj_out),
            static_cast<const __nv_bfloat16*>(act.q_proj_out),
            static_cast<const __nv_bfloat16*>(layer.q_norm_weight),
            q_rows, 1, static_cast<int>(head_dim),
            config_.rms_norm_eps,
            stream
        );
        kernels::rms_norm_bf16(
            static_cast<__nv_bfloat16*>(act.k_proj_out),
            static_cast<const __nv_bfloat16*>(act.k_proj_out),
            static_cast<const __nv_bfloat16*>(layer.k_norm_weight),
            k_rows, 1, static_cast<int>(head_dim),
            config_.rms_norm_eps,
            stream
        );
    } else {
        kernels::rms_norm_f16(
            static_cast<half*>(act.q_proj_out),
            static_cast<const half*>(act.q_proj_out),
            static_cast<const half*>(layer.q_norm_weight),
            q_rows, 1, static_cast<int>(head_dim),
            config_.rms_norm_eps,
            stream
        );
        kernels::rms_norm_f16(
            static_cast<half*>(act.k_proj_out),
            static_cast<const half*>(act.k_proj_out),
            static_cast<const half*>(layer.k_norm_weight),
            k_rows, 1, static_cast<int>(head_dim),
            config_.rms_norm_eps,
            stream
        );
    }
    
    // =====================================================================
    // Apply RoPE to Q and K
    // =====================================================================
    if (!start_pos_by_batch) {
        if (compute_dtype == DType::BF16) {
            kernels::apply_rope_bf16(
                static_cast<__nv_bfloat16*>(act.q_proj_out),
                static_cast<__nv_bfloat16*>(act.k_proj_out),
                batch_size, seq_len, num_heads, num_kv_heads, head_dim,

````

### File: backends/cuda/cuda_runtime.cpp (decode)

````cpp
    if (!guard_r.ok()) return guard_r.error();
    auto guard = std::move(guard_r.value());

    // prefill_chunked_pipeline uses session.cur_pos() (slot0). Temporarily mirror this slot's
    // position into slot0, then after success copy the updated pos back into this slot.
    if (slot != 0) {
        auto pos_guard = ember::CurPosGuard::set(session, /*slot=*/0, /*new_pos=*/start_pos);
        Error err = prefill_chunked_pipeline(tokens, session, chunk_len, overlap, out_logits);
        if (err) {
            return err;  // pos_guard restores slot0 automatically
        }
        const int new_pos0 = session.cur_pos(0);
        session.set_cur_pos(slot, new_pos0);
        return Error::success();
    }

    // slot0: no swap needed; prefill_chunked_pipeline already updates slot0.
    return prefill_chunked_pipeline(tokens, session, chunk_len, overlap, out_logits);
}

Error CudaRuntime::decode_single_forward_to_lm_head_(int last_token, Session& session) {
    const int batch_size = 1;
    const int seq_len = 1;
    const int start_pos = session.cur_pos();
    if (start_pos >= session.runtime_config().max_ctx_len) {
        return Error(ErrorCode::CONTEXT_TOO_LONG, "Context full");
    }

    const int max_ctx = session.runtime_config().max_ctx_len;
    Error err = allocate_activation_buffers(/*max_seq_len=*/1, batch_size, /*attn_q_max=*/1, /*attn_k_max=*/max_ctx);
    if (err) return err;

    int embed_device = device_map_.embedding_device;
    auto& act = activations_[embed_device];

    EMBER_RETURN_IF_ERROR(begin_stage_profile_());

    int* d_input_id = nullptr;
    CUDA_CHECK(cudaSetDevice(embed_device));
    CUDA_CHECK(cudaMalloc(&d_input_id, sizeof(int)));
    auto h2d_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(d_input_id, &last_token, sizeof(int), cudaMemcpyHostToDevice));
    auto h2d_end = std::chrono::high_resolution_clock::now();
    add_stage_profile_h2d_ms_(std::chrono::duration<double, std::milli>(h2d_end - h2d_start).count());

    if (profile_stages_) {
        CUDA_CHECK(cudaEventRecord(embedding_events_.start, streams_[embed_device]));
    }
    if (weights_.dtype == DType::BF16) {
        kernels::embedding_lookup_bf16(
            static_cast<__nv_bfloat16*>(act.hidden_states),
            static_cast<const __nv_bfloat16*>(weights_.embed_tokens),
            d_input_id,
            batch_size, seq_len, config_.hidden_size,
            streams_[embed_device]
        );
    } else {
        kernels::embedding_lookup_f16(
            static_cast<half*>(act.hidden_states),
            static_cast<const half*>(weights_.embed_tokens),
            d_input_id,
            batch_size, seq_len, config_.hidden_size,
            streams_[embed_device]
        );
    }
    if (profile_stages_) {
        CUDA_CHECK(cudaEventRecord(embedding_events_.end, streams_[embed_device]));
    }
    cudaFree(d_input_id);

    for (int layer_idx = 0; layer_idx < config_.num_layers; ++layer_idx) {
        err = forward_layer(layer_idx, batch_size, seq_len, start_pos, session, /*skip_input_copy=*/false);
        if (err) return err;
    }

    err = forward_final_norm(batch_size, seq_len, session);
    if (err) return err;
    err = forward_lm_head(batch_size, seq_len);
    if (err) return err;
    return Error::success();
}

Error CudaRuntime::decode_to_device(int last_token, Session& session) {
    if (!loaded_) {
        return Error(ErrorCode::MODEL_NOT_LOADED, "Model not loaded");
    }
    EMBER_RETURN_IF_ERROR(decode_single_forward_to_lm_head_(last_token, session));
    EMBER_RETURN_IF_ERROR(finalize_stage_profile_(/*sync_all_devices=*/true,
                                                  /*include_final_norm=*/true,
                                                  /*include_lm_head=*/true));
    session.advance(1);
    return Error::success();
}

Error CudaRuntime::decode_batch_to_device(const std::vector<int>& last_tokens, Session& session) {
    if (!loaded_) {
        return Error(ErrorCode::MODEL_NOT_LOADED, "Model not loaded");
    }
    const int batch_size = session.runtime_config().batch_size;
    const int seq_len = 1;
    if (batch_size <= 0) {
        return Error::invalid_argument("runtime_config.batch_size must be > 0");
    }
    if (static_cast<int>(last_tokens.size()) != batch_size) {
        return Error::invalid_argument("last_tokens size must equal runtime_config.batch_size");
    }
    const int max_ctx = session.runtime_config().max_ctx_len;

    std::vector<int> start_pos_by_batch(static_cast<size_t>(batch_size), -1);
    bool any_inactive = false;
    bool all_same = true;
    int first_active_pos = -1;
    for (int b = 0; b < batch_size; ++b) {
        const int sp = session.cur_pos(b);
        start_pos_by_batch[static_cast<size_t>(b)] = sp;
        if (sp < 0) {
            any_inactive = true;
            continue;
        }
        if (sp >= max_ctx) {
            return Error(ErrorCode::CONTEXT_TOO_LONG, "Context full");
        }
        if (first_active_pos < 0) first_active_pos = sp;
        else if (sp != first_active_pos) all_same = false;
    }
    if (first_active_pos < 0) {
        return Error::invalid_argument("decode_batch_to_device: no active slots");
    }
    const bool use_varpos = any_inactive || !all_same;

    Error err = allocate_activation_buffers(/*max_seq_len=*/1, batch_size, /*attn_q_max=*/1, /*attn_k_max=*/max_ctx);
    if (err) return err;

    int embed_device = device_map_.embedding_device;
    auto& act = activations_[embed_device];

    int* d_input_ids = nullptr;
    CUDA_CHECK(cudaSetDevice(embed_device));
    CUDA_CHECK(cudaMalloc(&d_input_ids, static_cast<size_t>(batch_size) * sizeof(int)));
    auto h2d_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(d_input_ids, last_tokens.data(),
                          static_cast<size_t>(batch_size) * sizeof(int),
                          cudaMemcpyHostToDevice));
    auto h2d_end = std::chrono::high_resolution_clock::now();
    add_stage_profile_h2d_ms_(std::chrono::duration<double, std::milli>(h2d_end - h2d_start).count());

    if (weights_.dtype == DType::BF16) {
        kernels::embedding_lookup_bf16(
            static_cast<__nv_bfloat16*>(act.hidden_states),
            static_cast<const __nv_bfloat16*>(weights_.embed_tokens),
            d_input_ids,
            batch_size, seq_len, config_.hidden_size,
            streams_[embed_device]
        );
    } else {
        kernels::embedding_lookup_f16(
            static_cast<half*>(act.hidden_states),
            static_cast<const half*>(weights_.embed_tokens),
            d_input_ids,
            batch_size, seq_len, config_.hidden_size,
            streams_[embed_device]
        );
    }
    cudaFree(d_input_ids);

    for (int layer_idx = 0; layer_idx < config_.num_layers; ++layer_idx) {
        err = forward_layer(layer_idx,
                            batch_size,
                            seq_len,
                            use_varpos ? 0 : first_active_pos,
                            session,
                            /*skip_input_copy=*/false,
                            use_varpos ? start_pos_by_batch.data() : nullptr);
        if (err) return err;
    }

    err = forward_final_norm(batch_size, seq_len, session);
    if (err) return err;
    err = forward_lm_head(batch_size, seq_len);
    if (err) return err;

    if (!use_varpos) {
        session.advance(1);
    } else {
        for (int b = 0; b < batch_size; ++b) {
            const int sp = start_pos_by_batch[static_cast<size_t>(b)];
            if (sp >= 0) session.set_cur_pos(b, sp + 1);
        }
    }
    return Error::success();
}

Error CudaRuntime::prefill(const std::vector<int>& tokens, Session& session) {
    if (!loaded_) {
        return Error(ErrorCode::MODEL_NOT_LOADED, "Model not loaded");
    }
    
    int batch_size = 1;
    int seq_len = static_cast<int>(tokens.size());
    
    if (seq_len > session.runtime_config().max_ctx_len) {
        return Error(ErrorCode::CONTEXT_TOO_LONG, "Input too long");
    }
    
    // 确保激活缓冲区已分配
    const int max_ctx = session.runtime_config().max_ctx_len;
    Error err = allocate_activation_buffers(seq_len, batch_size, /*attn_q_max=*/seq_len, /*attn_k_max=*/max_ctx);
    if (err) return err;

    if (profile_layers_) {
        last_layer_profile_ms_.assign(static_cast<size_t>(config_.num_layers), 0.0f);
    }
    EMBER_RETURN_IF_ERROR(begin_stage_profile_());
    
    // 拷贝 input_ids 到 GPU
    input_ids_cpu_ = tokens;
    int* d_input_ids = nullptr;
    int embed_device = device_map_.embedding_device;
    
    CUDA_CHECK(cudaSetDevice(embed_device));
    CUDA_CHECK(cudaMalloc(&d_input_ids, seq_len * sizeof(int)));
    auto h2d_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(d_input_ids, tokens.data(), seq_len * sizeof(int), cudaMemcpyHostToDevice));
    auto h2d_end = std::chrono::high_resolution_clock::now();
    add_stage_profile_h2d_ms_(std::chrono::duration<double, std::milli>(h2d_end - h2d_start).count());
    
    // Embedding lookup
    auto& act = activations_[embed_device];
    if (profile_stages_) {
        CUDA_CHECK(cudaEventRecord(embedding_events_.start, streams_[embed_device]));
    }
    if (weights_.dtype == DType::BF16) {
        kernels::embedding_lookup_bf16(
            static_cast<__nv_bfloat16*>(act.hidden_states),
            static_cast<const __nv_bfloat16*>(weights_.embed_tokens),
            d_input_ids,
            batch_size, seq_len, config_.hidden_size,
            streams_[embed_device]
        );
    } else {
        kernels::embedding_lookup_f16(
            static_cast<half*>(act.hidden_states),
            static_cast<const half*>(weights_.embed_tokens),
            d_input_ids,
            batch_size, seq_len, config_.hidden_size,
            streams_[embed_device]
        );
    }
    if (profile_stages_) {
        CUDA_CHECK(cudaEventRecord(embedding_events_.end, streams_[embed_device]));
    }
    
    cudaFree(d_input_ids);
    
    // 逐层前向
    int start_pos = session.cur_pos();
    
    for (int layer_idx = 0; layer_idx < config_.num_layers; ++layer_idx) {
        err = forward_layer(layer_idx, batch_size, seq_len, start_pos, session, /*skip_input_copy=*/false);
        if (err) return err;
    }
    
    // 同步所有设备
    for (int dev = 0; dev < device_map_.num_devices; ++dev) {
        cuda_sync(dev);
    }

    EMBER_RETURN_IF_ERROR(finalize_stage_profile_(/*sync_all_devices=*/false,
                                                  /*include_final_norm=*/false,
                                                  /*include_lm_head=*/false));
    
    // 更新位置
    session.set_cur_pos(start_pos + seq_len);
    
    return Error::success();
}

// 带 logits 返回的 prefill (用于立即采样)
Error CudaRuntime::prefill_with_logits(const std::vector<int>& tokens, Session& session, std::vector<float>& logits) {
    Error err = prefill(tokens, session);
    if (err) return err;

````

### File: backends/cuda/cuda_runtime.cpp (forward_layer (header + early body))

````cpp
    EMBER_RETURN_IF_ERROR(finalize_stage_profile_(/*sync_all_devices=*/false,
                                                  /*include_final_norm=*/true,
                                                  /*include_lm_head=*/true));
    
    // 更新位置
    session.advance(1);
    
    return Error::success();
}

Error CudaRuntime::forward_layer(int layer_idx,
                                 int batch_size,
                                 int seq_len,
                                 int start_pos,
                                 Session& session,
                                 bool skip_input_copy,
                                 const int* start_pos_by_batch) {
    int device_id = device_map_.layer_to_device[layer_idx];
    auto& act = activations_[device_id];
    auto& layer = weights_.layers[layer_idx];
    auto stream = streams_[device_id];
    auto& cublas = cublas_handles_[device_id];
    
    CUDA_CHECK(cudaSetDevice(device_id));
    cublasSetStream(cublas.get(), stream);
    
    DType compute_dtype = weights_.dtype;
    cudaDataType_t cuda_dtype = to_cuda_dtype(compute_dtype);
    size_t hidden_size = config_.hidden_size;
    size_t num_heads = config_.num_heads;
    size_t num_kv_heads = config_.num_kv_heads;
    size_t head_dim = config_.head_dim;
    size_t intermediate_size = config_.intermediate_size;
    size_t elem_size = dtype_size(compute_dtype);
    
    // 如果上一层在不同设备，需要拷贝 hidden_states
    if (layer_idx > 0) {
        int prev_device = device_map_.layer_to_device[layer_idx - 1];
        if (prev_device != device_id) {
            if (!skip_input_copy) {
            size_t size = batch_size * seq_len * hidden_size * elem_size;
            auto t0 = std::chrono::high_resolution_clock::now();
            Error err = copy_bytes_peer_or_staged(act.hidden_states, device_id,
                                                  activations_[prev_device].hidden_states, prev_device,
                                                  size);
            if (err) return err;
            if (profile_stages_) {
                auto t1 = std::chrono::high_resolution_clock::now();
                last_stage_profile_ms_.p2p_ms += static_cast<float>(
                    std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            }
        }
    }

    if (profile_layers_ && static_cast<size_t>(layer_idx) < last_layer_profile_ms_.size()) {
        auto& ev = profile_events_[device_id];
        CUDA_CHECK(cudaEventRecord(ev.start, stream));
    }

    if (session.runtime_config().check_correctness) {
        int target = session.runtime_config().dump_layer;
        if (target < 0 || target == layer_idx) {
            Error err = dump_last_row(session.runtime_config().dump_dir,
                                      "layer_" + std::to_string(layer_idx) + "_layer_input",
                                      device_id, act.hidden_states,
                                      seq_len, hidden_size, compute_dtype, stream);
            if (err) return err;
        }
    }
    
    // =====================================================================
    // Input LayerNorm
    // =====================================================================
    if (profile_stages_) {
        auto& sev = layer_stage_events_[static_cast<size_t>(layer_idx)];
        CUDA_CHECK(cudaEventRecord(sev.in_norm_start, stream));
    }
    if (compute_dtype == DType::BF16) {
        kernels::rms_norm_bf16(
            static_cast<__nv_bfloat16*>(act.norm_out),
            static_cast<const __nv_bfloat16*>(act.hidden_states),
            static_cast<const __nv_bfloat16*>(layer.input_layernorm_weight),
            batch_size, seq_len, hidden_size,
            config_.rms_norm_eps,
            stream
        );
    } else {
        kernels::rms_norm_f16(
            static_cast<half*>(act.norm_out),
            static_cast<const half*>(act.hidden_states),
            static_cast<const half*>(layer.input_layernorm_weight),
            batch_size, seq_len, hidden_size,
            config_.rms_norm_eps,
            stream
        );
    }
    if (profile_stages_) {
        auto& sev = layer_stage_events_[static_cast<size_t>(layer_idx)];
        CUDA_CHECK(cudaEventRecord(sev.in_norm_end, stream));
        CUDA_CHECK(cudaEventRecord(sev.attn_start, stream));
    }
    
    // =====================================================================
    // QKV Projection: norm_out @ W_q/k/v -> q/k/v_proj_out
    // 权重布局: [out_features, in_features] (row-major, Qwen3 safetensors格式)
    // 输入: [batch*seq, hidden_size]
    // 输出: [batch*seq, out_features]
    // =====================================================================
    const float alpha_one = 1.0f;
    const float beta_zero = 0.0f;
    
    int M = batch_size * seq_len;  // 批次*序列长度
    
    // Q projection: [M, hidden] @ [hidden, num_heads*head_dim]^T = [M, num_heads*head_dim]
    CUBLAS_CHECK(cublasGemmEx(
        cublas.get(),
        CUBLAS_OP_T, CUBLAS_OP_N,  // W^T @ X
        num_heads * head_dim, M, hidden_size,
        &alpha_one,
        layer.q_proj_weight, cuda_dtype, hidden_size,  // W: [num_heads*head_dim, hidden] -> W^T
        act.norm_out, cuda_dtype, hidden_size,         // X: [M, hidden]
        &beta_zero,
        act.q_proj_out, cuda_dtype, num_heads * head_dim,  // Y: [M, num_heads*head_dim]
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
    
    // K projection
    CUBLAS_CHECK(cublasGemmEx(
        cublas.get(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_kv_heads * head_dim, M, hidden_size,
        &alpha_one,
        layer.k_proj_weight, cuda_dtype, hidden_size,
        act.norm_out, cuda_dtype, hidden_size,
        &beta_zero,
        act.k_proj_out, cuda_dtype, num_kv_heads * head_dim,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
    
    // V projection
    CUBLAS_CHECK(cublasGemmEx(
        cublas.get(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_kv_heads * head_dim, M, hidden_size,
        &alpha_one,
        layer.v_proj_weight, cuda_dtype, hidden_size,
        act.norm_out, cuda_dtype, hidden_size,
        &beta_zero,
        act.v_proj_out, cuda_dtype, num_kv_heads * head_dim,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));

    // =====================================================================
    // Q/K per-head RMSNorm (Qwen3)
    // =====================================================================
    int q_rows = batch_size * seq_len * static_cast<int>(num_heads);
    int k_rows = batch_size * seq_len * static_cast<int>(num_kv_heads);
    if (compute_dtype == DType::BF16) {
        kernels::rms_norm_bf16(
            static_cast<__nv_bfloat16*>(act.q_proj_out),
            static_cast<const __nv_bfloat16*>(act.q_proj_out),
            static_cast<const __nv_bfloat16*>(layer.q_norm_weight),
            q_rows, 1, static_cast<int>(head_dim),
            config_.rms_norm_eps,
            stream
        );
        kernels::rms_norm_bf16(
            static_cast<__nv_bfloat16*>(act.k_proj_out),
            static_cast<const __nv_bfloat16*>(act.k_proj_out),
            static_cast<const __nv_bfloat16*>(layer.k_norm_weight),
            k_rows, 1, static_cast<int>(head_dim),
            config_.rms_norm_eps,
            stream
        );
    } else {
        kernels::rms_norm_f16(
            static_cast<half*>(act.q_proj_out),
            static_cast<const half*>(act.q_proj_out),
            static_cast<const half*>(layer.q_norm_weight),
            q_rows, 1, static_cast<int>(head_dim),
            config_.rms_norm_eps,
            stream
        );
        kernels::rms_norm_f16(
            static_cast<half*>(act.k_proj_out),
            static_cast<const half*>(act.k_proj_out),
            static_cast<const half*>(layer.k_norm_weight),
            k_rows, 1, static_cast<int>(head_dim),
            config_.rms_norm_eps,
            stream
        );
    }
    
    // =====================================================================
    // Apply RoPE to Q and K
    // =====================================================================
    if (!start_pos_by_batch) {
        if (compute_dtype == DType::BF16) {
            kernels::apply_rope_bf16(
                static_cast<__nv_bfloat16*>(act.q_proj_out),
                static_cast<__nv_bfloat16*>(act.k_proj_out),
                batch_size, seq_len, num_heads, num_kv_heads, head_dim,
                start_pos, config_.rope_theta,
                stream
            );
        } else {
            kernels::apply_rope_f16(
                static_cast<half*>(act.q_proj_out),
                static_cast<half*>(act.k_proj_out),
                batch_size, seq_len, num_heads, num_kv_heads, head_dim,
                start_pos, config_.rope_theta,
                stream
            );
        }
    }
    
    // =====================================================================
    // Update KV Cache
    // =====================================================================
    auto& kv_cache = session.kv_cache();
    auto& layer_kv = kv_cache.layer(layer_idx);
    
    const int max_seq = session.runtime_config().max_ctx_len;
    if (!start_pos_by_batch) {
        if (compute_dtype == DType::BF16) {
            kernels::update_kv_cache_bf16(
                static_cast<__nv_bfloat16*>(layer_kv.key_cache.data),

````

---

## 3) 报告上下文（完整）

### Report: reports/stage1_milestone_4b_20260225_mainline/stage1_summary.md

````md
# Stage 1.1 Milestone Summary

- Model: `/home/dong/xilinx/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554`
- Generated at: `2026-02-25T01:29:40`

| gpus | split | mode | prompt_len | decode_steps | prefill_ms | decode_per_token_ms | prefill_share_% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0+1 | 18+18 | no_overlap | 1024 | 128 | 189.054 | 17.173 | 7.92 |
| 0+1 | 18+18 | no_overlap | 1024 | 256 | 190.418 | 17.172 | 4.15 |
| 0+1 | 18+18 | no_overlap | 1024 | 64 | 187.029 | 17.171 | 14.54 |
| 0+1 | 18+18 | no_overlap | 2048 | 128 | 454.479 | 18.476 | 16.12 |
| 0+1 | 18+18 | no_overlap | 2048 | 256 | 452.585 | 18.538 | 8.71 |
| 0+1 | 18+18 | no_overlap | 2048 | 64 | 448.461 | 18.185 | 27.81 |
| 0+1 | 18+18 | no_overlap | 4096 | 128 | 1084.414 | 20.495 | 29.25 |
| 0+1 | 18+18 | no_overlap | 4096 | 256 | 1095.467 | 20.289 | 17.42 |
| 0+1 | 18+18 | no_overlap | 4096 | 64 | 1085.963 | 20.162 | 45.70 |
| 0+1 | 18+18 | no_overlap | 512 | 128 | 88.459 | 16.664 | 3.98 |
| 0+1 | 18+18 | no_overlap | 512 | 256 | 88.575 | 16.666 | 2.03 |
| 0+1 | 18+18 | no_overlap | 512 | 64 | 88.663 | 16.593 | 7.71 |
| 0+1 | 18+18 | overlap | 1024 | 128 | 190.202 | 17.192 | 7.96 |
| 0+1 | 18+18 | overlap | 1024 | 256 | 193.988 | 17.265 | 4.20 |
| 0+1 | 18+18 | overlap | 1024 | 64 | 185.929 | 17.217 | 14.44 |
| 0+1 | 18+18 | overlap | 2048 | 128 | 447.763 | 18.501 | 15.90 |
| 0+1 | 18+18 | overlap | 2048 | 256 | 450.038 | 18.578 | 8.64 |
| 0+1 | 18+18 | overlap | 2048 | 64 | 448.556 | 18.344 | 27.64 |
| 0+1 | 18+18 | overlap | 4096 | 128 | 1085.979 | 20.138 | 29.64 |
| 0+1 | 18+18 | overlap | 4096 | 256 | 1078.841 | 20.312 | 17.18 |
| 0+1 | 18+18 | overlap | 4096 | 64 | 1087.917 | 20.453 | 45.39 |
| 0+1 | 18+18 | overlap | 512 | 128 | 88.449 | 16.684 | 3.98 |
| 0+1 | 18+18 | overlap | 512 | 256 | 91.693 | 16.691 | 2.10 |
| 0+1 | 18+18 | overlap | 512 | 64 | 88.055 | 16.610 | 7.65 |

## Key Point
- Highest prefill share: `45.70%` (prompt_len=4096, decode_steps=64, gpus=0+1, mode=no_overlap).

````

### Report: reports/stage1_split_profile_4b_20260225_mainline/stage12_summary.md

````md
# Stage 1.2 Pipeline Split Profiling

- Model: `/home/dong/xilinx/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554`
- Generated at: `2026-02-25T01:32:46`

## Throughput vs Split
| split | mode | total_ms | tok/s_est | prefill_share_% |
| --- | --- | --- | --- | --- |
| 12+24 | no_overlap | 2770.230 | 46.206 | 15.73 |
| 12+24 | overlap | 2767.109 | 46.258 | 15.61 |
| 18+18 | no_overlap | 2830.028 | 45.229 | 16.08 |
| 18+18 | overlap | 2817.737 | 45.427 | 16.07 |
| 24+12 | no_overlap | 2957.505 | 43.280 | 16.86 |
| 24+12 | overlap | 2915.003 | 43.911 | 16.63 |
| 27+9 | no_overlap | 3014.120 | 42.467 | 16.69 |
| 27+9 | overlap | 2987.478 | 42.846 | 16.74 |
| 9+27 | no_overlap | 2769.227 | 46.222 | 15.83 |
| 9+27 | overlap | 2755.312 | 46.456 | 15.61 |

## Key Point
- Best rollout tok/s split: `9+27` in `overlap` mode (`46.456 tok/s`, total `2755.312 ms`).

## Bubble vs Split
| split | no_overlap_ms | overlap_ms | speedup_x | hidden_% |
| --- | --- | --- | --- | --- |
| 12+24 | 2770.230 | 2767.109 | 1.0011 | 0.11 |
| 18+18 | 2830.028 | 2817.737 | 1.0044 | 0.43 |
| 24+12 | 2957.505 | 2915.003 | 1.0146 | 1.44 |
| 27+9 | 3014.120 | 2987.478 | 1.0089 | 0.88 |
| 9+27 | 2769.227 | 2755.312 | 1.0051 | 0.50 |

````

---

## 4) 风格参考

附件上传：`tutorial_01_life_of_a_token.md`

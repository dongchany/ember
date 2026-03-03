# Tutorial #0 Prompt (Full, With Code + Reports)

> 生成日期：2026-02-26
> 用途：整份复制到新 chat 作为写作输入
> 标注：大文件（如 cuda_runtime.cpp）按“该篇相关段落”摘取，避免上下文爆炸

---

## 0) 系统 Prompt（先贴这段）

```text
我正在为我的开源项目 Ember 写一个系列教程。请帮我写第 0 篇。

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
请写第 0 篇：5 分钟跑通 Ember（编译、下载模型、生成第一个 token）。

## 本篇必须引用的代码与报告都在本消息里（不要再让我补充路径）

- 用一台 11GB 显存卡用户也能照做的口径写
- 明确区分“最小可跑通路径”和“推荐性能路径”
- 给出常见报错排查（模型路径、CUDA 架构、显存不足）
```

---

## 2) 代码上下文（完整/相关段落）

### File: README.md

````md
# Ember - Qwen3 CUDA Inference Engine

轻量级 Qwen3 CUDA 推理引擎，专为消费级多 GPU（如双卡 RTX 3080Ti）设计。

## 快速开始

1) 编译（以 RTX 3080Ti 为例，SM=86）：

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

2) 下载模型（safetensors 格式，首次可能需要先 `huggingface-cli login`）：

```bash
# 若未安装 huggingface-cli：pip install -U huggingface_hub
huggingface-cli download Qwen/Qwen3-0.6B --local-dir ./qwen3-0.6b
```

3) 运行：

```bash
./build/ember -m ./qwen3-0.6b -p "Hello, my name is"
```

## 文档

- [开发指南](docs/development.md)
- [测试与回归](docs/testing.md)
- [架构概览](docs/architecture.md)
- [采样器解析](docs/sampler_explanation.md)

## 特性

- **原生 CUDA 实现**：完全控制计算流程，不依赖 ggml/llama.cpp
- **Safetensors 直接加载**：原生解析 HuggingFace 格式，无需转换
- **Pipeline Parallel**：支持多卡层级切分，自动根据显存分配
- **FP16 计算**：权重和激活使用 FP16，cuBLAS GEMM 用于矩阵运算
- **自定义 Kernels**：RMSNorm、RoPE、Softmax、SiLU、Attention 等

## 项目结构

```
ember/
├── apps/ember_cli/         # CLI 入口（main.cpp）
├── cli/                    # CLI 参数解析
├── core/                   # 核心抽象（纯 C++，不依赖 CUDA）
├── runtime/                # 调度/设备映射等 runtime 逻辑
├── formats/                # safetensors/config 等加载
├── backends/cuda/          # CUDA runtime + kernels
├── examples/               # 最小可运行示例
├── benchmarks/             # 性能/互联带宽基准
├── tests/                  # 单测与 smoke tests
├── scripts/                # CI/对齐检查脚本
└── docs/                   # 设计/测试等文档
```

## 构建

### 依赖

- CMake 3.18+
- CUDA Toolkit 11.0+（推荐 12.x）
- C++17 编译器

### 编译

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release  # RTX 3080Ti
cmake --build build --parallel
```

CUDA 架构参数：
- `86` - RTX 3080/3090
- `89` - RTX 4090
- `80` - A100

## 使用

### 基本用法

```bash
# 单卡推理
./build/ember -m /path/to/qwen3-4b -p "Hello, my name is"

# 双卡推理
./build/ember -m /path/to/qwen3-14b --devices 0,1 -p "Explain quantum computing"

# 交互模式
./build/ember -m /path/to/qwen3-4b -i
```

### 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-m, --model` | 模型目录（包含 safetensors 和 config.json） | 必填 |
| `-p, --prompt` | 输入提示 | "Hello, my name is" |
| `--devices` | GPU 设备列表 | 0 |
| `-c, --ctx-size` | 上下文长度 | 2048 |
| `-n, --n-predict` | 生成 token 数 | 128 |
| `--temp` | 温度 | 0.7 |
| `--top-p` | Top-P | 0.9 |
| `--top-k` | Top-K | 40 |
| `-i, --interactive` | 交互模式 | false |
| `-v, --verbose` | 详细输出 | false |

## 支持的模型

- Qwen3-0.6B
- Qwen3-1.7B
- Qwen3-4B
- Qwen3-8B
- Qwen3-14B（需双卡）

模型需从 HuggingFace 下载 safetensors 格式：

```bash
# 使用 huggingface-cli
# 若未安装 huggingface-cli：pip install -U huggingface_hub
huggingface-cli download Qwen/Qwen3-4B --local-dir ./qwen3-4b
```

## 架构设计

### 核心抽象

1. **Tensor**: 轻量视图（shape + dtype + data 指针），不管理生命周期
2. **Session**: 推理会话，管理 KV Cache 和生成状态
3. **IRuntime**: 后端接口，支持 CUDA（未来可扩展 CPU）
4. **DeviceMap**: 层级设备映射，支持 Pipeline Parallel

### 计算流程

```
Input IDs
    ↓
Embedding Lookup (GPU 0)
    ↓
┌─────────────────────────────┐
│ Layer 0-N (可跨多卡)         │
│   ├── Input LayerNorm       │
│   ├── QKV Projection        │
│   ├── RoPE                  │
│   ├── KV Cache Update       │
│   ├── Attention (Q@K^T→V)   │
│   ├── O Projection          │
│   ├── Residual Add          │
│   ├── Post-Attn LayerNorm   │
│   ├── MLP (SwiGLU)          │
│   └── Residual Add          │
└─────────────────────────────┘
    ↓
Final LayerNorm
    ↓
LM Head → Logits
    ↓
Sampling → Next Token
```

### 多卡策略

采用 Pipeline Parallel，按层切分：

```
GPU 0: Embedding + Layers 0-13
GPU 1: Layers 14-27 + LM Head
```

层间通过 `cudaMemcpyPeer` 传输 hidden_states。

## 性能

（实际性能取决于硬件和模型大小）

| 模型 | 显存占用 | 预期速度 |
|------|----------|----------|
| Qwen3-4B (FP16) | ~8 GB | ~40 tok/s |
| Qwen3-8B (FP16) | ~16 GB | ~25 tok/s |
| Qwen3-14B (FP16) | ~28 GB | ~15 tok/s (双卡) |

## 路线图

- [x] M0: 单卡推理基础框架
- [ ] M1: 双卡 Pipeline Parallel
- [ ] M2: 量化支持 (INT8/INT4)
- [ ] M3: FlashAttention 优化

## 引用

如果你在论文/报告中使用 Ember，可引用 Zenodo 归档：

- DOI: https://doi.org/10.5281/zenodo.18477269

## 许可证

Apache-2.0 

## 致谢

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - 架构参考
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - 模型格式
- [Qwen](https://github.com/QwenLM/Qwen) - 模型权重

````

### File: docs/development.md

````md
# Development Guide

This guide helps contributors build, run, and modify Ember quickly.

## Build

Requirements:
- CMake 3.18+
- CUDA Toolkit 11.0+ (12.x recommended)
- C++17 compiler

Build:
```
mkdir -p build
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --parallel
```

CI-style build (isolated dir):
```
scripts/ci/build.sh
```

## Run

```
./build/ember -m /path/to/qwen3-0.6b -p "Hello, my name is"
```

## Code layout

- `apps/ember_cli/main.cpp`: CLI entry point and sampling loop.
- `core/`: configs, session, sampler, tokenizer.
- `runtime/`: runtime interface, device map.
- `formats/`: safetensors loader.
- `backends/cuda/`: CUDA runtime and kernels.

## How to add features

1. Add CLI flags in `apps/ember_cli/main.cpp`.
2. Add runtime config fields in `core/config.h`.
3. Implement kernel changes in `backends/cuda/kernels/`.
4. Wire into `backends/cuda/cuda_runtime.cpp`.
5. Update docs and tests.

## Local debug workflow

- Use `--check` to dump logits/hidden states:
```
./build/ember -m /path/to/model --check --dump-layer 2 -p "Hello"
```
- Compare with HuggingFace:
```
python3 scripts/compare_logits.py --model /path/to/model --debug-dir debug/check_...
python3 scripts/compare_hidden.py --model /path/to/model --debug-dir debug/check_... --layer 2
```

## First tasks for new contributors

- Change a CLI default (e.g., sampling) and update README.
- Add a small check to `compare_logits.py`.
- Improve a kernel comment or error message.

````

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

### File: scripts/ci/build.sh

````sh
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

BUILD_DIR="${BUILD_DIR:-build}"
CUDA_ARCH="${CUDA_ARCH:-86}"
CMAKE_FORCE_CONFIGURE="${CMAKE_FORCE_CONFIGURE:-0}"
CMAKE_CACHE="${BUILD_DIR}/CMakeCache.txt"

if [[ -t 1 ]]; then
  COLOR_YELLOW=$'\033[1;33m'
  COLOR_GREEN=$'\033[0;32m'
  COLOR_RESET=$'\033[0m'
else
  COLOR_YELLOW=""
  COLOR_GREEN=""
  COLOR_RESET=""
fi

log_notice() {
  printf '%b\n' "${COLOR_YELLOW}==> $*${COLOR_RESET}"
}

log_ok() {
  printf '%b\n' "${COLOR_GREEN}==> $*${COLOR_RESET}"
}

needs_configure=0
if [[ "${CMAKE_FORCE_CONFIGURE}" == "1" ]]; then
  needs_configure=1
elif [[ ! -f "${CMAKE_CACHE}" ]]; then
  needs_configure=1
else
  if find . -path "./${BUILD_DIR}" -prune -o \( -name 'CMakeLists.txt' -o -name '*.cmake' \) -newer "${CMAKE_CACHE}" -print -quit | grep -q .; then
    needs_configure=1
  fi
fi

if [[ "${needs_configure}" == "1" ]]; then
  log_ok "Running CMake configure..."
  cmake -S . -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"
else
  log_notice "CMake configure skipped (cache is up-to-date)."
fi
cmake --build "${BUILD_DIR}" --parallel

````

---

## 3) 报告上下文（完整）

### Report: reports/stage1_milestone_4b_20260225_mainline/stage1_mainline_ready.md

````md
# Stage 1.1 Mainline Readiness

- usable summary rows: 24
- failed combos: 0
- oom combos: 0

## Missing Combos
- none

## Recommendation
- Use current 8B results for Stage 1.1 mainline plots/tables.
- Full grid coverage achieved for current Stage 1.1 matrix.

````

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

---

## 4) 风格参考

附件上传：`tutorial_01_life_of_a_token.md`

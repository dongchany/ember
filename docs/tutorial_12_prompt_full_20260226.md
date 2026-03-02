# Tutorial #12 Prompt (Full, With Code + Reports)

> 生成日期：2026-02-26
> 用途：整份复制到新 chat 作为写作输入
> 标注：大文件（如 cuda_runtime.cpp）按“该篇相关段落”摘取，避免上下文爆炸

---

## 0) 系统 Prompt（先贴这段）

```text
我正在为我的开源项目 Ember 写一个系列教程。请帮我写第 12 篇。

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
请写第 12 篇：LoRA 注入与热更新 — W <- W + scale*(B@A)。

## 本篇必须引用的代码与报告都在本消息里（不要再让我补充路径）

- 数学从低秩分解直觉出发，不要默认读者线代很强
- 讲清 merge/rollback/replace 三种路径的差异
```

---

## 2) 代码上下文（完整/相关段落）

### File: docs/lora_adapter_quickstart.md

````md
# LoRA Adapter Quickstart

目标：快速得到一个**真实 PEFT LoRA adapter**，用于 Ember 的热更新 benchmark。

## 1) 训练最小 LoRA adapter

### 依赖安装（仅首次）

```bash
pip install torch transformers peft accelerate sentencepiece
```

### 方式 A：直接用内置合成数据（最快）

```bash
scripts/train/run_min_lora.sh \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --output-dir reports/adapters/qwen3_4b_min \
  --max-steps 50 \
  --max-length 256 \
  --per-device-batch-size 1 \
  --grad-acc-steps 8
```

### 方式 B：使用你自己的 `jsonl` 数据

`jsonl` 每行至少包含一组 prompt/response（字段名支持多种别名）：

- prompt: `prompt` / `instruction` / `input` / `question`
- response: `response` / `output` / `answer`

```bash
scripts/train/run_min_lora.sh \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --dataset-jsonl /abs/path/train.jsonl \
  --output-dir reports/adapters/qwen3_4b_real \
  --max-steps 100 \
  --max-length 512
```

训练完成后，`--output-dir` 下应有：

- `adapter_config.json`
- `adapter_model.safetensors`

## 2) 跑 Ember LoRA 热更新 benchmark

```bash
python3 scripts/report/run_stage1_lora_hot_update.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter reports/adapters/qwen3_4b_min \
  --gpus 0,1 \
  --split 9,27 \
  --iters 3 \
  --warmup 1 \
  --out-dir reports/stage31_lora_hot_update_qwen3_4b_real
```

## 3) 常见注意事项

- Qwen3-4B 训练仍有显存压力，建议先用较小 `max-length`（256）和较少 `max-steps`。
- 如果你的数据质量一般，先追求“能产出 adapter 并打通流程”，再逐步提升训练设置。

````

### File: benchmarks/lora_hot_update_benchmark.cpp

````cpp
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
    bool replace_existing = false;
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
                << "  --replace-existing  unmerge previous adapter before applying new one\n"
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
        } else if (arg == "--replace-existing") {
            replace_existing = true;
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
        err = cuda_rt->apply_lora_adapter(adapter_dir, scale, replace_existing, &st);
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
        << (replace_existing ? 1 : 0) << ","
        << effective_scale << ","
        << iters << ","
        << warmup << ","
        << updated_avg << ","
        << skipped_avg << ","
        << ext_ms_avg << ","
        << inner_ms_avg;

    const std::string header =
        "mode,model_dir,adapter_dir,gpus,split,scale,replace_existing,effective_scale,iters,warmup,"
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

````

### File: scripts/report/run_stage1_lora_hot_update.py

````py
#!/usr/bin/env python3
import argparse
import datetime as dt
import os
from pathlib import Path
from typing import Dict, List, Optional

from common_report import die, read_csv, run_cmd, safe_float, split_ints


def hf_hub_root() -> Path:
    hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE", "").strip()
    if hub_cache:
        return Path(hub_cache).expanduser().resolve()
    hf_home = os.environ.get("HF_HOME", "").strip()
    if hf_home:
        return (Path(hf_home).expanduser().resolve() / "hub")
    return (Path.home() / ".cache" / "huggingface" / "hub").resolve()


def resolve_snapshot_dir(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    if (path / "config.json").exists() and list(path.glob("*.safetensors")):
        return path
    snap_root = path / "snapshots"
    if not snap_root.exists():
        return None
    candidates = [
        p
        for p in snap_root.iterdir()
        if p.is_dir() and (p / "config.json").exists() and list(p.glob("*.safetensors"))
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def resolve_model_dir(model_arg: str) -> Path:
    raw = model_arg.strip()
    if not raw:
        die("--model is empty")
    p = Path(raw).expanduser().resolve()
    resolved = resolve_snapshot_dir(p)
    if resolved is not None:
        return resolved
    hub_root = hf_hub_root()
    model_cache_dir = hub_root / ("models--" + raw.replace("/", "--"))
    resolved = resolve_snapshot_dir(model_cache_dir)
    if resolved is not None:
        return resolved
    die(
        "failed to resolve model from local cache: "
        f"{raw}. Checked path='{p}' and HF cache='{model_cache_dir}'."
    )
    raise AssertionError("unreachable")


def resolve_adapter_dir(adapter_arg: str) -> Path:
    p = Path(adapter_arg).expanduser().resolve()
    if p.is_file():
        p = p.parent
    if not p.exists():
        die(f"adapter path not found: {p}")
    if not list(p.glob("*.safetensors")):
        die(f"no .safetensors found in adapter dir: {p}")
    return p


def write_summary_md(path: Path, model_dir: Path, adapter_dir: Path, row: Dict[str, str]) -> None:
    lines: List[str] = []
    lines.append("# Stage 3.1 LoRA Hot Update")
    lines.append("")
    lines.append(f"- Model: `{model_dir}`")
    lines.append(f"- Adapter: `{adapter_dir}`")
    lines.append(f"- Generated at: `{dt.datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    lines.append(f"| gpus | `{row.get('gpus','')}` |")
    lines.append(f"| split | `{row.get('split','')}` |")
    lines.append(f"| scale | `{row.get('scale','')}` |")
    lines.append(f"| replace_existing | `{row.get('replace_existing','')}` |")
    lines.append(f"| effective_scale | `{row.get('effective_scale','')}` |")
    lines.append(f"| updated_matrices | `{row.get('updated_matrices','')}` |")
    lines.append(f"| skipped_matrices | `{row.get('skipped_matrices','')}` |")
    lines.append(f"| apply_ms_ext | `{row.get('apply_ms_ext','')}` |")
    lines.append(f"| apply_ms_inner | `{row.get('apply_ms_inner','')}` |")
    lines.append("")
    lines.append("## Key Point")
    lines.append(
        "- Attention q/k/v/o matrices can be merged in-place from PEFT LoRA adapter "
        "without reloading base model weights."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_p1_input(path: Path, row: Dict[str, str]) -> None:
    lines: List[str] = []
    lines.append("# P1 Input — LoRA Hot Update")
    lines.append("")
    lines.append(
        f"- LoRA hot update latency (external wall): `{row.get('apply_ms_ext','')}` ms; "
        f"runtime internal: `{row.get('apply_ms_inner','')}` ms."
    )
    lines.append(
        f"- Updated matrices: `{row.get('updated_matrices','')}`, "
        f"skipped: `{row.get('skipped_matrices','')}`."
    )
    lines.append(
        f"- Effective scale used in merge: `{row.get('effective_scale','')}` "
        f"(user scale `{row.get('scale','')}`)."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Stage 3.1 LoRA hot update benchmark.")
    ap.add_argument("--model", type=str, required=True, help="model path or HF model id")
    ap.add_argument("--adapter", type=str, required=True, help="LoRA adapter dir")
    ap.add_argument("--gpus", type=str, default="0,1")
    ap.add_argument("--split", type=str, default="9,27")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--replace-existing", action="store_true", default=False)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--build-dir", type=str, default="build")
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    if args.iters <= 0:
        die("--iters must be > 0")
    if args.warmup < 0:
        die("--warmup must be >= 0")
    _ = split_ints(args.gpus)
    if args.split.strip():
        s = split_ints(args.split)
        if len(s) != 2:
            die("--split expects A,B")

    model_dir = resolve_model_dir(args.model)
    adapter_dir = resolve_adapter_dir(args.adapter)

    repo = Path.cwd()
    bench_bin = (repo / args.build_dir / "ember_lora_hot_update_benchmark").resolve()
    if not bench_bin.exists():
        die(f"missing binary: {bench_bin}")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (repo / "reports" / f"stage31_lora_hot_update_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "stage31_lora_hot_update.csv"
    cmd = [
        str(bench_bin),
        "--model",
        str(model_dir),
        "--adapter",
        str(adapter_dir),
        "--gpus",
        args.gpus,
        "--split",
        args.split,
        "--scale",
        str(args.scale),
        "--iters",
        str(args.iters),
        "--warmup",
        str(args.warmup),
        "--csv",
        str(csv_path),
    ]
    if args.replace_existing:
        cmd.append("--replace-existing")

    p = run_cmd(cmd, cwd=repo, log_path=logs_dir / "run.log", check=False)
    if p.returncode != 0:
        die(f"benchmark failed rc={p.returncode}; see {logs_dir / 'run.log'}")

    rows = read_csv(csv_path)
    if not rows:
        die(f"empty csv: {csv_path}")
    row = rows[0]

    write_summary_md(out_dir / "stage31_summary.md", model_dir, adapter_dir, row)
    write_p1_input(out_dir / "stage31_p1_input.md", row)

    print(f"[done] out_dir={out_dir}")
    print(
        f"[result] updated={row.get('updated_matrices','')} "
        f"apply_ms_ext={safe_float(row.get('apply_ms_ext','0')):.3f} "
        f"apply_ms_inner={safe_float(row.get('apply_ms_inner','0')):.3f}"
    )


if __name__ == "__main__":
    main()

````

### File: backends/cuda/cuda_runtime.cpp (LoRA: name parsing)

````cpp
    cudaFreeHost(host);
    return Error::success();
}

struct LoraTargetKey {
    int layer_idx = -1;
    std::string proj;
    bool is_a = false;
};

static bool parse_lora_target_key(const std::string& tensor_name, LoraTargetKey& out) {
    // Supports:
    //   ...layers.<i>.self_attn.<proj>.lora_A.weight
    //   ...layers.<i>.self_attn.<proj>.lora_B.weight
    //   ...layers.<i>.self_attn.<proj>.lora_A.default.weight
    //   ...layers.<i>.self_attn.<proj>.lora_B.default.weight
    static const std::regex kPat(
        R"(layers\.([0-9]+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.lora_([AB])(?:\.default)?\.weight$)"
    );
    std::smatch m;
    if (!std::regex_search(tensor_name, m, kPat)) {
        return false;
    }
    out.layer_idx = std::stoi(m[1].str());
    out.proj = m[2].str();
    out.is_a = (m[3].str() == "A");
    return true;
}

static float read_lora_alpha_over_r(const std::string& adapter_dir) {
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

std::string MemoryEstimate::to_string() const {
    std::string s;
    s += "Memory Estimate:\n";
    s += "  Weights:     " + format_bytes(weights_bytes) + "\n";
    s += "  KV Cache:    " + format_bytes(kv_cache_bytes) + "\n";
    s += "  Activations: " + format_bytes(activation_bytes) + "\n";
    s += "  Workspace:   " + format_bytes(workspace_bytes) + "\n";
    s += "  Total:       " + format_bytes(total_bytes) + "\n";
    return s;
}

void DeviceMap::print() const {
    std::cout << "Device Map:\n";
    std::cout << "  Embedding: GPU " << embedding_device << "\n";
    std::cout << "  LM Head:   GPU " << lm_head_device << "\n";
    std::cout << "  Layers:\n";
    
    int prev_device = -1;
    int start_layer = 0;
    for (size_t i = 0; i <= layer_to_device.size(); ++i) {
        int device = (i < layer_to_device.size()) ? layer_to_device[i] : -1;
        if (device != prev_device) {
            if (prev_device >= 0) {
                std::cout << "    Layers " << start_layer << "-" << (i-1) 
                          << " -> GPU " << prev_device << "\n";
            }
            start_layer = i;
            prev_device = device;
        }
    }
}

DeviceMap DeviceMap::auto_map(const ModelConfig& config,
                              const std::vector<size_t>& gpu_free_memory,
                              int ctx_len, int batch_size) {
    DeviceMap dm;
    int num_gpus = static_cast<int>(gpu_free_memory.size());
    int num_layers = config.num_layers;
    
    if (num_gpus == 0) {
        // 没有 GPU，返回空映射
        return dm;
    }
    
    if (num_gpus == 1) {
        // 单卡，所有层都在 GPU 0
        return single_device(num_layers, 0);
    }

    // ---------------------------------------------------------------------
    // 多卡：以“每层显存”作为单位，做连续分段切分（更适合 pipeline）。
    // 注意：这里是估算（权重 + KV），用于初步切分避免 OOM，精确值以实际分配为准。
    // ---------------------------------------------------------------------
    DType dtype = dtype_from_string(config.torch_dtype);
    if (dtype == DType::UNKNOWN) dtype = DType::F16;
    const size_t elem = dtype_size(dtype);

    auto estimate_layer_weight_bytes = [&]() -> size_t {
        const size_t H = static_cast<size_t>(config.hidden_size);
        const size_t I = static_cast<size_t>(config.intermediate_size);
        const size_t Nh = static_cast<size_t>(config.num_heads);
        const size_t Nk = static_cast<size_t>(config.num_kv_heads);
        const size_t Hd = static_cast<size_t>(config.head_dim);

        size_t total = 0;
        total += H * (Nh * Hd) * elem;   // q_proj
        total += H * (Nk * Hd) * elem;   // k_proj
        total += H * (Nk * Hd) * elem;   // v_proj
        total += (Nh * Hd) * H * elem;   // o_proj
        total += (Hd * 2) * elem;        // q_norm/k_norm
        total += (H * I) * elem;         // gate_proj
        total += (H * I) * elem;         // up_proj
        total += (I * H) * elem;         // down_proj
        total += (H * 2) * elem;         // layernorm weights (approx)
        return total;
    };

    const size_t per_layer_weights = estimate_layer_weight_bytes();
    const size_t per_layer_kv = config.kv_cache_size_per_layer(ctx_len, batch_size, dtype);
    const size_t per_layer_total = per_layer_weights + per_layer_kv;

    // 额外权重：embedding / lm_head / final_norm
    const size_t embed_bytes = static_cast<size_t>(config.vocab_size) * static_cast<size_t>(config.hidden_size) * elem;
    const size_t final_norm_bytes = static_cast<size_t>(config.hidden_size) * elem;
    const size_t lm_head_bytes = config.tie_word_embeddings ? embed_bytes
                                                            : (static_cast<size_t>(config.vocab_size) * static_cast<size_t>(config.hidden_size) * elem);

    std::vector<size_t> avail = gpu_free_memory;
    if (!avail.empty()) {
        avail[0] = (avail[0] > embed_bytes) ? (avail[0] - embed_bytes) : 0;
    }
    if (avail.size() >= 2) {
        size_t& last = avail[avail.size() - 1];
        const size_t extra = final_norm_bytes + lm_head_bytes;
        last = (last > extra) ? (last - extra) : 0;
    }

    std::vector<size_t> capacity(num_gpus, 0);
    size_t cap_sum = 0;
    for (int i = 0; i < num_gpus; ++i) {
        capacity[i] = (per_layer_total == 0) ? 0 : (avail[i] / per_layer_total);
        cap_sum += capacity[i];
    }

    dm.layer_to_device.resize(num_layers);
    dm.num_devices = num_gpus;
    dm.embedding_device = 0;
    dm.lm_head_device = num_gpus - 1;

    // 容量估算不足：退化为均分（由后续实际分配决定是否 OOM）。
    if (cap_sum < static_cast<size_t>(num_layers) || per_layer_total == 0) {
        int current = 0;
        const int layers_per_device = (num_layers + num_gpus - 1) / num_gpus;
        for (int l = 0; l < num_layers; ++l) {
            dm.layer_to_device[l] = current;
            if ((l + 1) % layers_per_device == 0 && current < num_gpus - 1) current++;
        }
        return dm;
    }


````

### File: backends/cuda/cuda_runtime.cpp (LoRA: apply adapter (core))

````cpp
    cuda_free(up_tmp);
    if (err) {
        cuda_free(layer.gate_up_proj_weight);
        layer.gate_up_proj_weight = nullptr;
        return err;
    }

    layer.gate_proj_weight = layer.gate_up_proj_weight;
    layer.up_proj_weight = up_dst;
    layer.gate_up_proj_packed = true;
    EMBER_RETURN_IF_ERROR(load_weight("mlp.down_proj.weight", &layer.down_proj_weight));
    
    // LayerNorms
    EMBER_RETURN_IF_ERROR(load_weight("input_layernorm.weight", &layer.input_layernorm_weight));
    EMBER_RETURN_IF_ERROR(load_weight("post_attention_layernorm.weight", &layer.post_attention_layernorm_weight));
    
    layer.allocated = true;
    return Error::success();
}

Error CudaRuntime::apply_lora_adapter(const std::string& adapter_dir,
                                      float scale,
                                      bool replace_existing,
                                      LoraApplyStats* stats) {
    if (!loaded_) {
        return Error(ErrorCode::MODEL_NOT_LOADED, "Model not loaded");
    }
    if (adapter_dir.empty()) {
        return Error::invalid_argument("adapter_dir is empty");
    }
    if (weights_.dtype != DType::F16 && weights_.dtype != DType::BF16) {
        return Error(ErrorCode::INVALID_FORMAT, "LoRA apply supports F16/BF16 weights only");
    }

    auto apply_one = [&](const std::string& one_adapter_dir,
                         float one_scale,
                         bool print_log,
                         LoraApplyStats* out_stats) -> Error {
        LoraApplyStats local_stats{};
        auto t0 = std::chrono::high_resolution_clock::now();

        ModelWeightLoader loader;
        Error err = loader.open(one_adapter_dir);
        if (err) return err;

        struct ABPair {
            std::string a_name;
            std::string b_name;
        };
        std::unordered_map<std::string, ABPair> pairs;
        const std::vector<std::string> names = loader.tensor_names();
        for (const std::string& name : names) {
            LoraTargetKey key;
            if (!parse_lora_target_key(name, key)) continue;
            const std::string pair_key = std::to_string(key.layer_idx) + ":" + key.proj;
            auto& p = pairs[pair_key];
            if (key.is_a) {
                p.a_name = name;
            } else {
                p.b_name = name;
            }
        }
        if (pairs.empty()) {
            return Error(ErrorCode::WEIGHT_NOT_FOUND,
                         "No supported LoRA tensors found under " + one_adapter_dir);
        }

        const float alpha_over_r = read_lora_alpha_over_r(one_adapter_dir);
        const float effective_scale = one_scale * alpha_over_r;
        local_stats.scale_used = effective_scale;

        const cudaDataType_t cuda_dtype = to_cuda_dtype(weights_.dtype);

        auto pick_target = [&](int layer_idx, const std::string& proj, void** weight_ptr,
                               int* out_dim, int* in_dim, int* device_id) -> Error {
            if (layer_idx < 0 || layer_idx >= config_.num_layers) {
                return Error(ErrorCode::INVALID_ARGUMENT,
                             "LoRA layer index out of range: " + std::to_string(layer_idx));
            }
            auto& layer = weights_.layers[static_cast<size_t>(layer_idx)];
            *device_id = layer.device_id;
            if (proj == "q_proj") {
                *weight_ptr = layer.q_proj_weight;
                *out_dim = config_.num_heads * config_.head_dim;
                *in_dim = config_.hidden_size;
            } else if (proj == "k_proj") {
                *weight_ptr = layer.k_proj_weight;
                *out_dim = config_.num_kv_heads * config_.head_dim;
                *in_dim = config_.hidden_size;
            } else if (proj == "v_proj") {
                *weight_ptr = layer.v_proj_weight;
                *out_dim = config_.num_kv_heads * config_.head_dim;
                *in_dim = config_.hidden_size;
            } else if (proj == "o_proj") {
                *weight_ptr = layer.o_proj_weight;
                *out_dim = config_.hidden_size;
                *in_dim = config_.num_heads * config_.head_dim;
            } else {
                return Error(ErrorCode::INVALID_ARGUMENT, "Unsupported LoRA target: " + proj);
            }
            if (*weight_ptr == nullptr) {
                return Error(ErrorCode::WEIGHT_NOT_FOUND,
                             "Target weight not allocated for layer " + std::to_string(layer_idx) +
                             " proj " + proj);
            }
            return Error::success();
        };

        for (const auto& kv : pairs) {
            const ABPair& p = kv.second;
            if (p.a_name.empty() || p.b_name.empty()) {
                local_stats.skipped_matrices++;
                continue;
            }

            LoraTargetKey key{};
            if (!parse_lora_target_key(p.a_name, key)) {
                local_stats.skipped_matrices++;
                continue;
            }

            const SafetensorsMeta* a_meta = loader.get_meta(p.a_name);
            const SafetensorsMeta* b_meta = loader.get_meta(p.b_name);
            if (!a_meta || !b_meta || a_meta->shape.size() != 2 || b_meta->shape.size() != 2) {
                return Error(ErrorCode::SHAPE_MISMATCH,
                             "Invalid LoRA tensor shape for pair: " + p.a_name + " / " + p.b_name);
            }

            void* weight_ptr = nullptr;
            int out_dim = 0;
            int in_dim = 0;
            int device_id = 0;
            EMBER_RETURN_IF_ERROR(
                pick_target(key.layer_idx, key.proj, &weight_ptr, &out_dim, &in_dim, &device_id));

            const int r = static_cast<int>(a_meta->shape[0]);
            const int a_in = static_cast<int>(a_meta->shape[1]);
            const int b_out = static_cast<int>(b_meta->shape[0]);
            const int b_r = static_cast<int>(b_meta->shape[1]);
            if (r <= 0 || a_in <= 0 || b_out <= 0 || b_r <= 0) {
                return Error(ErrorCode::SHAPE_MISMATCH,
                             "Non-positive LoRA dimensions for pair: " + p.a_name + " / " + p.b_name);
            }
            if (a_in != in_dim || b_out != out_dim || b_r != r) {
                return Error(ErrorCode::SHAPE_MISMATCH,
                             "LoRA shape mismatch at layer " + std::to_string(key.layer_idx) +
                             " proj " + key.proj +
                             " (A=[" + std::to_string(r) + "," + std::to_string(a_in) + "]"
                             ", B=[" + std::to_string(b_out) + "," + std::to_string(b_r) + "]"
                             ", expected out=" + std::to_string(out_dim) +
                             ", in=" + std::to_string(in_dim) + ")");
            }

            void* d_a = nullptr;
            void* d_b = nullptr;
            auto cleanup = [&]() {
                cuda_free(d_a);
                cuda_free(d_b);
                d_a = nullptr;
                d_b = nullptr;
            };

            err = load_tensor_to_device(loader, p.a_name, device_id, weights_.dtype, &d_a);
            if (err) {
                cleanup();
                return err;
            }
            err = load_tensor_to_device(loader, p.b_name, device_id, weights_.dtype, &d_b);
            if (err) {
                cleanup();
                return err;
            }

            cudaError_t cu_err = cudaSetDevice(device_id);
            if (cu_err != cudaSuccess) {
                cleanup();
                return Error::cuda_error(std::string("cudaSetDevice failed: ") + cudaGetErrorString(cu_err));
            }

            cublasHandle_t handle = cublas_handles_[static_cast<size_t>(device_id)].get();
            cudaStream_t stream = streams_[static_cast<size_t>(device_id)];

            cublasStatus_t cb = cublasSetStream(handle, stream);
            if (cb != CUBLAS_STATUS_SUCCESS) {
                cleanup();
                return Error::cuda_error("cublasSetStream failed: " + std::to_string(static_cast<int>(cb)));
            }

            // Row-major update:
            //   W_row[out, in] += scale * (B_row[out, r] @ A_row[r, in])
            // Compute through column-major view:
            //   W_col[in, out] += scale * (A_col[in, r] @ B_col[r, out])
            const float alpha = effective_scale;
            const float beta = 1.0f;
            cb = cublasGemmEx(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                in_dim, out_dim, r,
                &alpha,
                d_a, cuda_dtype, in_dim,
                d_b, cuda_dtype, r,
                &beta,
                weight_ptr, cuda_dtype, in_dim,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP
            );
            if (cb != CUBLAS_STATUS_SUCCESS) {
                cleanup();
                return Error::cuda_error(
                    "cublasGemmEx (LoRA merge) failed: " + std::to_string(static_cast<int>(cb)));
            }

            cu_err = cudaStreamSynchronize(stream);
            if (cu_err != cudaSuccess) {
                cleanup();
                return Error::cuda_error(
                    std::string("cudaStreamSynchronize failed: ") + cudaGetErrorString(cu_err));
            }

            cleanup();
            local_stats.updated_matrices++;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        local_stats.wall_ms = static_cast<float>(
            std::chrono::duration<double, std::milli>(t1 - t0).count());

        if (local_stats.updated_matrices <= 0) {
            return Error(ErrorCode::WEIGHT_NOT_FOUND,
                         "No complete LoRA A/B pairs were applied under " + one_adapter_dir);
        }
        if (out_stats) {
            *out_stats = local_stats;
        }
        if (print_log) {
            std::cout << "[CudaRuntime] LoRA applied: updated=" << local_stats.updated_matrices
                      << ", skipped=" << local_stats.skipped_matrices
                      << ", scale=" << local_stats.scale_used
                      << ", wall_ms=" << local_stats.wall_ms << std::endl;
        }
        return Error::success();
    };

    LoraApplyStats aggregate_stats{};
    if (replace_existing && has_active_lora_adapter_) {
        LoraApplyStats rollback_stats{};
        Error rollback_err = apply_one(active_lora_adapter_dir_, -active_lora_scale_, false, &rollback_stats);
        if (rollback_err) return rollback_err;
        aggregate_stats.updated_matrices += rollback_stats.updated_matrices;
        aggregate_stats.skipped_matrices += rollback_stats.skipped_matrices;
        aggregate_stats.wall_ms += rollback_stats.wall_ms;
        has_active_lora_adapter_ = false;
        active_lora_adapter_dir_.clear();
        active_lora_scale_ = 0.0f;
    }

    LoraApplyStats local_stats{};
    Error err = apply_one(adapter_dir, scale, true, &local_stats);
    if (err) return err;
    aggregate_stats.updated_matrices += local_stats.updated_matrices;
    aggregate_stats.skipped_matrices += local_stats.skipped_matrices;
    aggregate_stats.scale_used = local_stats.scale_used;
    aggregate_stats.wall_ms += local_stats.wall_ms;

    has_active_lora_adapter_ = true;
    active_lora_adapter_dir_ = adapter_dir;
    active_lora_scale_ = scale;

    if (stats) {
        *stats = aggregate_stats;
    }
    return Error::success();
}

Error CudaRuntime::debug_copy_attention_weight(int layer_idx,
                                               const std::string& proj,
                                               std::vector<float>& out,
                                               int* out_dim,
                                               int* in_dim) const {
    if (!loaded_) {
        return Error(ErrorCode::MODEL_NOT_LOADED, "Model not loaded");
    }
    if (layer_idx < 0 || layer_idx >= config_.num_layers) {
        return Error::invalid_argument("layer_idx out of range");
    }

    const auto& layer = weights_.layers[static_cast<size_t>(layer_idx)];
    const void* w_ptr = nullptr;
    int o_dim = 0;
    int i_dim = 0;
    if (proj == "q_proj") {
        w_ptr = layer.q_proj_weight;
        o_dim = config_.num_heads * config_.head_dim;
        i_dim = config_.hidden_size;
    } else if (proj == "k_proj") {
        w_ptr = layer.k_proj_weight;
        o_dim = config_.num_kv_heads * config_.head_dim;
        i_dim = config_.hidden_size;
    } else if (proj == "v_proj") {
        w_ptr = layer.v_proj_weight;
        o_dim = config_.num_kv_heads * config_.head_dim;

````

---

## 3) 报告上下文（完整）

### Report: reports/stage31_lora_hot_update_4b_20260225_mainline/stage31_summary.md

````md
# Stage 3.1 LoRA Hot Update

- Model: `/home/dong/xilinx/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554`
- Adapter: `/home/dong/workspace/ember/reports/synthetic_lora_qwen3_4b_r8`
- Generated at: `2026-02-25T13:19:30`

| metric | value |
| --- | --- |
| gpus | `0+1` |
| split | `9+27` |
| scale | `1.000` |
| replace_existing | `0` |
| effective_scale | `2.000` |
| updated_matrices | `144` |
| skipped_matrices | `0` |
| apply_ms_ext | `353.980` |
| apply_ms_inner | `353.876` |

## Key Point
- Attention q/k/v/o matrices can be merged in-place from PEFT LoRA adapter without reloading base model weights.

````

### Report: reports/stage31_lora_hot_update_4b_20260225_mainline/stage31_lora_hot_update.csv

````csv
mode,model_dir,adapter_dir,gpus,split,scale,replace_existing,effective_scale,iters,warmup,updated_matrices,skipped_matrices,apply_ms_ext,apply_ms_inner
lora_hot_update,/home/dong/xilinx/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554,/home/dong/workspace/ember/reports/synthetic_lora_qwen3_4b_r8,0+1,9+27,1.000,0,2.000,1,0,144,0,353.980,353.876

````

### Report: reports/stage31_lora_weight_merge_check_4b_20260225_peft_perturb_layer0_mainline/stage31_lora_weight_merge_check.csv

````csv
mode,model_dir,adapter_dir,gpus,split,layer_idx,proj,scale,effective_scale,out_dim,in_dim,rank,delta_max_abs_diff,delta_mean_abs_diff,rollback_max_abs_diff,rollback_mean_abs_diff
lora_weight_merge_check,/home/dong/xilinx/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554,/home/dong/workspace/ember/reports/adapters/qwen3_4b_peft_perturb_r8_20260225,0,,0,k_proj,1.00000000,2.00000000,1024,2560,8,0.00024298,0.00002309,0.00024414,0.00000010
lora_weight_merge_check,/home/dong/xilinx/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554,/home/dong/workspace/ember/reports/adapters/qwen3_4b_peft_perturb_r8_20260225,0,,0,o_proj,1.00000000,2.00000000,2560,4096,8,0.00024298,0.00001944,0.00024414,0.00000008
lora_weight_merge_check,/home/dong/xilinx/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554,/home/dong/workspace/ember/reports/adapters/qwen3_4b_peft_perturb_r8_20260225,0,,0,q_proj,1.00000000,2.00000000,4096,2560,8,0.00024426,0.00002228,0.00024414,0.00000010
lora_weight_merge_check,/home/dong/xilinx/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554,/home/dong/workspace/ember/reports/adapters/qwen3_4b_peft_perturb_r8_20260225,0,,0,v_proj,1.00000000,2.00000000,1024,2560,8,0.00024368,0.00002251,0.00024414,0.00000010

````

### Report: reports/stage31_lora_delta_profile_4b_20260225_peft_perturb_mainline_v2/stage31_lora_delta_freeze_summary.csv

````csv
freeze_layers,frozen_prefix_max_delta,frozen_prefix_max_base
8,0.14062500,0.80468750
12,0.14062500,1.28320312
18,0.23437500,2.78125000
24,0.28125000,2.78125000
30,0.56250000,4.02709961

````

---

## 4) 风格参考

附件上传：`tutorial_01_life_of_a_token.md`

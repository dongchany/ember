# Tutorial #18 Prompt (Full, With Code + Reports)

> 生成日期：2026-02-26
> 用途：整份复制到新 chat 作为写作输入
> 标注：大文件（如 cuda_runtime.cpp）按“该篇相关段落”摘取，避免上下文爆炸

---

## 0) 系统 Prompt（先贴这段）

```text
我正在为我的开源项目 Ember 写一个系列教程。请帮我写第 18 篇。

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
请写第 18 篇：统一后端 vs 双栈 — 为什么 Ember 把推理和训练放在一起。

## 本篇必须引用的代码与报告都在本消息里（不要再让我补充路径）

- 必须明确写出关键数字：14.985 GiB vs 7.492 GiB；28 ms vs 312 ms
- 区分“显存优势”“更新延迟优势”“端到端吞吐优势”三条证据链
```

---

## 2) 代码上下文（完整/相关段落）

### File: benchmarks/rollout_update_loop_benchmark.cpp

````cpp
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

````

### File: scripts/report/run_stage53_unified_backend_advantage.py

````py
#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional

from common_report import die, read_csv, safe_float


def size_of_safetensors_bytes(dir_path: Path) -> int:
    total = 0
    for p in dir_path.glob("*.safetensors"):
        if p.is_file():
            total += p.stat().st_size
    return total


def to_gib(x: int) -> float:
    return float(x) / (1024.0 ** 3)


def pick_rollout_tok_s(
    framework_csv: Path,
    engine: str,
    scenario: str,
) -> Dict[str, str]:
    rows = read_csv(framework_csv)
    if not rows:
        die(f"empty framework compare csv: {framework_csv}")
    engine_l = engine.strip().lower()
    scenario_l = scenario.strip().lower()
    cands: List[Dict[str, str]] = []
    for r in rows:
        if str(r.get("status", "")).strip().lower() != "ok":
            continue
        if str(r.get("engine", "")).strip().lower() != engine_l:
            continue
        if scenario_l and str(r.get("scenario", "")).strip().lower() != scenario_l:
            continue
        cands.append(r)
    if not cands:
        die(
            f"no matched row in framework csv for engine='{engine}' scenario='{scenario}'. "
            f"file={framework_csv}"
        )
    best = max(cands, key=lambda r: safe_float(r.get("rollout_tok_s", "0")))
    tok_s = safe_float(best.get("rollout_tok_s", "0"))
    if tok_s <= 0:
        die(f"invalid rollout_tok_s in matched row: {best}")
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 5.3 unified-backend advantage summary.")
    ap.add_argument("--model-dir", type=str, required=True)
    ap.add_argument("--hot-update-csv", type=str, required=True, help="stage31_lora_hot_update.csv")
    ap.add_argument("--adapter-dir", type=str, default="")
    ap.add_argument("--num-rounds", type=int, default=30)
    ap.add_argument("--pcie-bandwidth-gbps", type=float, default=24.0, help="effective host<->device GB/s")
    ap.add_argument("--framework-compare-csv", type=str, default="", help="optional framework_compare.csv for rollout tok/s")
    ap.add_argument("--rollout-engine", type=str, default="ember")
    ap.add_argument("--rollout-scenario", type=str, default="dual(0,1)")
    ap.add_argument("--num-prompts", type=int, default=100)
    ap.add_argument("--num-candidates", type=int, default=8)
    ap.add_argument("--decode-steps", type=int, default=128)
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    if args.num_rounds <= 0:
        die("--num-rounds must be > 0")
    if args.pcie_bandwidth_gbps <= 0:
        die("--pcie-bandwidth-gbps must be > 0")
    if args.num_prompts <= 0 or args.num_candidates <= 0 or args.decode_steps <= 0:
        die("--num-prompts/--num-candidates/--decode-steps must be > 0")

    model_dir = Path(args.model_dir).expanduser().resolve()
    hot_csv = Path(args.hot_update_csv).expanduser().resolve()
    if not model_dir.exists():
        die(f"model-dir not found: {model_dir}")
    if not hot_csv.exists():
        die(f"hot-update-csv not found: {hot_csv}")

    rows = read_csv(hot_csv)
    if not rows:
        die(f"empty hot-update csv: {hot_csv}")
    r0 = rows[0]
    measured_hot_update_ms = float(r0.get("apply_ms_ext", "0") or "0")

    adapter_dir = Path(args.adapter_dir).expanduser().resolve() if args.adapter_dir.strip() else None
    if adapter_dir is None:
        raw = r0.get("adapter_dir", "")
        if raw:
            adapter_dir = Path(raw).expanduser().resolve()
    adapter_bytes = 0
    if adapter_dir is not None and adapter_dir.exists():
        adapter_bytes = size_of_safetensors_bytes(adapter_dir)

    model_bytes = size_of_safetensors_bytes(model_dir)
    if model_bytes <= 0:
        die(f"no *.safetensors under {model_dir}")

    dual_model_gib = to_gib(model_bytes * 2)
    unified_model_gib = to_gib(model_bytes)
    model_mem_saved_gib = dual_model_gib - unified_model_gib
    model_mem_saved_pct = (model_mem_saved_gib / dual_model_gib * 100.0) if dual_model_gib > 0 else 0.0

    bw_bytes_per_s = args.pcie_bandwidth_gbps * (1024.0 ** 3)
    full_sync_ms = float(model_bytes) / bw_bytes_per_s * 1000.0
    lora_sync_ms = float(adapter_bytes) / bw_bytes_per_s * 1000.0 if adapter_bytes > 0 else 0.0

    total_full_sync_ms = full_sync_ms * args.num_rounds
    total_lora_sync_ms = lora_sync_ms * args.num_rounds
    total_hot_update_ms = measured_hot_update_ms * args.num_rounds

    framework_csv: Optional[Path] = None
    rollout_row: Optional[Dict[str, str]] = None
    if args.framework_compare_csv.strip():
        framework_csv = Path(args.framework_compare_csv).expanduser().resolve()
        if not framework_csv.exists():
            die(f"framework-compare-csv not found: {framework_csv}")
        rollout_row = pick_rollout_tok_s(
            framework_csv=framework_csv,
            engine=args.rollout_engine,
            scenario=args.rollout_scenario,
        )

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (Path.cwd() / "reports" / f"stage53_unified_backend_advantage_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_md = out_dir / "stage53_summary.md"
    out_csv = out_dir / "stage53_summary.csv"

    summary_rows = [
        {
            "metric": "model_weights_gib",
            "value": f"{to_gib(model_bytes):.6f}",
            "notes": "sum of model *.safetensors",
        },
        {
            "metric": "adapter_weights_gib",
            "value": f"{to_gib(adapter_bytes):.6f}",
            "notes": "sum of adapter *.safetensors",
        },
        {
            "metric": "dual_stack_model_only_gib",
            "value": f"{dual_model_gib:.6f}",
            "notes": "2x model copy (train + infer process)",
        },
        {
            "metric": "unified_model_only_gib",
            "value": f"{unified_model_gib:.6f}",
            "notes": "single model copy",
        },
        {
            "metric": "model_memory_saved_gib",
            "value": f"{model_mem_saved_gib:.6f}",
            "notes": f"{model_mem_saved_pct:.3f}% vs dual-stack model-only footprint",
        },
        {
            "metric": "sync_full_model_ms_per_round_est",
            "value": f"{full_sync_ms:.6f}",
            "notes": f"assume {args.pcie_bandwidth_gbps:.2f} GiB/s",
        },
        {
            "metric": "sync_lora_ms_per_round_est",
            "value": f"{lora_sync_ms:.6f}",
            "notes": f"assume {args.pcie_bandwidth_gbps:.2f} GiB/s",
        },
        {
            "metric": "hot_update_ms_per_round_measured",
            "value": f"{measured_hot_update_ms:.6f}",
            "notes": f"from {hot_csv.name}",
        },
        {
            "metric": f"sync_full_model_ms_{args.num_rounds}round_est",
            "value": f"{total_full_sync_ms:.6f}",
            "notes": "transfer-only estimate",
        },
        {
            "metric": f"sync_lora_ms_{args.num_rounds}round_est",
            "value": f"{total_lora_sync_ms:.6f}",
            "notes": "transfer-only estimate",
        },
        {
            "metric": f"hot_update_ms_{args.num_rounds}round_measured",
            "value": f"{total_hot_update_ms:.6f}",
            "notes": "in-process adapter merge",
        },
    ]

    if rollout_row is not None:
        rollout_tok_s = safe_float(rollout_row.get("rollout_tok_s", "0"))
        rollout_tokens_per_round = float(args.num_prompts * args.num_candidates * args.decode_steps)
        rollout_ms_per_round = (rollout_tokens_per_round / rollout_tok_s * 1000.0) if rollout_tok_s > 0 else 0.0
        dual_full_round_ms = rollout_ms_per_round + full_sync_ms
        dual_lora_round_ms = rollout_ms_per_round + lora_sync_ms
        unified_round_ms = rollout_ms_per_round + measured_hot_update_ms
        dual_full_tok_s_e2e = (rollout_tokens_per_round / dual_full_round_ms * 1000.0) if dual_full_round_ms > 0 else 0.0
        dual_lora_tok_s_e2e = (rollout_tokens_per_round / dual_lora_round_ms * 1000.0) if dual_lora_round_ms > 0 else 0.0
        unified_tok_s_e2e = (rollout_tokens_per_round / unified_round_ms * 1000.0) if unified_round_ms > 0 else 0.0
        speedup_vs_dual_full = (dual_full_round_ms / unified_round_ms) if unified_round_ms > 0 else 0.0
        speedup_vs_dual_lora = (dual_lora_round_ms / unified_round_ms) if unified_round_ms > 0 else 0.0

        summary_rows.extend(
            [
                {
                    "metric": "e2e_rollout_tok_s_source",
                    "value": f"{rollout_tok_s:.6f}",
                    "notes": f"{rollout_row.get('engine','')}/{rollout_row.get('scenario','')} from framework csv",
                },
                {
                    "metric": "e2e_rollout_tokens_per_round",
                    "value": f"{rollout_tokens_per_round:.0f}",
                    "notes": f"num_prompts={args.num_prompts}, num_candidates={args.num_candidates}, decode_steps={args.decode_steps}",
                },
                {
                    "metric": "e2e_dual_fullsync_round_ms_est",
                    "value": f"{dual_full_round_ms:.6f}",
                    "notes": "rollout + full model sync",
                },
                {
                    "metric": "e2e_unified_round_ms_est",
                    "value": f"{unified_round_ms:.6f}",
                    "notes": "rollout + measured in-process hot update",
                },
                {
                    "metric": "e2e_dual_fullsync_tok_s_est",
                    "value": f"{dual_full_tok_s_e2e:.6f}",
                    "notes": "end-to-end throughput",
                },
                {
                    "metric": "e2e_unified_tok_s_est",
                    "value": f"{unified_tok_s_e2e:.6f}",
                    "notes": "end-to-end throughput",
                },
                {
                    "metric": "e2e_unified_speedup_vs_dual_fullsync_x",
                    "value": f"{speedup_vs_dual_full:.6f}",
                    "notes": "round-time ratio",
                },
                {
                    "metric": "e2e_dual_lora_sync_round_ms_est",
                    "value": f"{dual_lora_round_ms:.6f}",
                    "notes": "rollout + LoRA-size sync estimate",
                },
                {
                    "metric": "e2e_dual_lora_sync_tok_s_est",
                    "value": f"{dual_lora_tok_s_e2e:.6f}",
                    "notes": "end-to-end throughput",
                },
                {
                    "metric": "e2e_unified_speedup_vs_dual_lora_sync_x",
                    "value": f"{speedup_vs_dual_lora:.6f}",
                    "notes": "round-time ratio",
                },
            ]
        )

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "value", "notes"])
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    lines = [
        "# Stage 5.3 Unified Backend Advantage (Memory + Sync + E2E)",
        "",
        f"- Generated at: `{dt.datetime.now().isoformat(timespec='seconds')}`",
        f"- Model dir: `{model_dir}`",
        f"- Adapter dir: `{adapter_dir if adapter_dir else ''}`",
        f"- Hot-update source: `{hot_csv}`",
        "",
        "| metric | value | notes |",
        "| --- | --- | --- |",
    ]
    for r in summary_rows:
        lines.append(f"| {r['metric']} | {r['value']} | {r['notes']} |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("[done] stage5.3 unified backend advantage")
    print(f"- summary md: {out_md}")
    print(f"- summary csv: {out_csv}")


if __name__ == "__main__":
    main()

````

### File: scripts/report/run_stage53_e2e_loop_compare.py

````py
#!/usr/bin/env python3
import argparse
import datetime as dt
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from common_report import die, read_csv, run_cmd, safe_float, write_csv


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
    cands = [p for p in snap_root.iterdir() if p.is_dir() and (p / "config.json").exists() and list(p.glob("*.safetensors"))]
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


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
        die(f"no .safetensors in adapter dir: {p}")
    return p


def size_of_safetensors_bytes(dir_path: Path) -> int:
    total = 0
    for p in dir_path.glob("*.safetensors"):
        if p.is_file():
            total += p.stat().st_size
    return total


def sync_ms_from_bytes(num_bytes: int, bandwidth_gib_per_s: float) -> float:
    return float(num_bytes) / (bandwidth_gib_per_s * (1024.0 ** 3)) * 1000.0


def one_run(
    *,
    bench_bin: Path,
    repo: Path,
    logs_dir: Path,
    mode_name: str,
    model_dir: Path,
    adapter_dir: Path,
    gpus: str,
    split: str,
    prompt_len: int,
    gen_len: int,
    num_candidates: int,
    rounds: int,
    warmup: int,
    chunk_len: int,
    overlap: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    scale: float,
    replace_existing: bool,
    update_mode: str,
    simulate_sync_ms: float,
    out_dir: Path,
) -> Tuple[Path, Path]:
    summary_csv = out_dir / f"stage53_{mode_name}.csv"
    per_round_csv = out_dir / f"stage53_{mode_name}_per_round.csv"
    cmd = [
        str(bench_bin),
        "--model",
        str(model_dir),
        "--adapter",
        str(adapter_dir),
        "--update-mode",
        update_mode,
        "--simulate-sync-ms",
        f"{simulate_sync_ms:.8f}",
        "--gpus",
        gpus,
        "--split",
        split,
        "--prompt-len",
        str(prompt_len),
        "--gen-len",
        str(gen_len),
        "--num-candidates",
        str(num_candidates),
        "--rounds",
        str(rounds),
        "--warmup",
        str(warmup),
        "--chunk-len",
        str(chunk_len),
        "--temperature",
        str(temperature),
        "--top-p",
        str(top_p),
        "--top-k",
        str(top_k),
        "--seed",
        str(seed),
        "--scale",
        str(scale),
        "--csv",
        str(summary_csv),
        "--per-round-csv",
        str(per_round_csv),
    ]
    if overlap:
        cmd.append("--overlap")
    else:
        cmd.append("--no-overlap")
    if not replace_existing:
        cmd.append("--no-replace-existing")
    p = run_cmd(cmd, cwd=repo, log_path=logs_dir / f"{mode_name}.log", check=False)
    if p.returncode != 0:
        die(f"benchmark failed for mode={mode_name}; see {logs_dir / (mode_name + '.log')}")
    if not summary_csv.exists():
        die(f"missing output csv for mode={mode_name}: {summary_csv}")
    return summary_csv, per_round_csv


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 5.3 measured e2e rollout+update loop comparison.")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--adapter", type=str, required=True)
    ap.add_argument("--gpus", type=str, default="0,1")
    ap.add_argument("--split", type=str, default="18,18")
    ap.add_argument("--prompt-len", type=int, default=1024)
    ap.add_argument("--gen-len", type=int, default=128)
    ap.add_argument("--num-candidates", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--chunk-len", type=int, default=512)
    ap.add_argument("--overlap", action="store_true", default=True)
    ap.add_argument("--no-overlap", dest="overlap", action="store_false")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--replace-existing", action="store_true", default=True)
    ap.add_argument("--no-replace-existing", dest="replace_existing", action="store_false")
    ap.add_argument("--sync-bandwidth-gib-per-s", type=float, default=24.0)
    ap.add_argument("--build-dir", type=str, default="build")
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    if args.prompt_len <= 0 or args.gen_len <= 0 or args.num_candidates <= 0:
        die("prompt/gen/candidates must be > 0")
    if args.rounds <= 0 or args.warmup < 0:
        die("--rounds must be > 0 and --warmup >= 0")
    if args.sync_bandwidth_gib_per_s <= 0:
        die("--sync-bandwidth-gib-per-s must be > 0")

    repo = Path.cwd()
    model_dir = resolve_model_dir(args.model)
    adapter_dir = resolve_adapter_dir(args.adapter)
    bench_bin = (repo / args.build_dir / "ember_rollout_update_loop_benchmark").resolve()
    if not bench_bin.exists():
        die(f"missing benchmark binary: {bench_bin}")

    model_bytes = size_of_safetensors_bytes(model_dir)
    adapter_bytes = size_of_safetensors_bytes(adapter_dir)
    if model_bytes <= 0:
        die(f"no model safetensors under {model_dir}")
    if adapter_bytes <= 0:
        die(f"no adapter safetensors under {adapter_dir}")
    full_sync_ms = sync_ms_from_bytes(model_bytes, args.sync_bandwidth_gib_per_s)
    lora_sync_ms = sync_ms_from_bytes(adapter_bytes, args.sync_bandwidth_gib_per_s)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (
        repo / "reports" / f"stage53_e2e_loop_compare_{ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        ("unified_apply", "apply", 0.0),
        ("dual_fullsync_sim", "skip", full_sync_ms),
        ("dual_lora_sync_sim", "skip", lora_sync_ms),
    ]

    rows: List[Dict[str, str]] = []
    for idx, (name, update_mode, sync_ms) in enumerate(runs):
        summary_csv, _ = one_run(
            bench_bin=bench_bin,
            repo=repo,
            logs_dir=logs_dir,
            mode_name=name,
            model_dir=model_dir,
            adapter_dir=adapter_dir,
            gpus=args.gpus,
            split=args.split,
            prompt_len=args.prompt_len,
            gen_len=args.gen_len,
            num_candidates=args.num_candidates,
            rounds=args.rounds,
            warmup=args.warmup,
            chunk_len=args.chunk_len,
            overlap=args.overlap,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed + idx,
            scale=args.scale,
            replace_existing=args.replace_existing,
            update_mode=update_mode,
            simulate_sync_ms=sync_ms,
            out_dir=out_dir,
        )
        r = read_csv(summary_csv)
        if not r:
            die(f"empty summary csv: {summary_csv}")
        row = dict(r[0])
        row["scenario"] = name
        row["sync_ms_assumed"] = f"{sync_ms:.6f}"
        rows.append(row)

    out_csv = out_dir / "stage53_e2e_compare.csv"
    out_md = out_dir / "stage53_e2e_compare.md"

    # normalized summary table
    summary_rows: List[Dict[str, str]] = []
    for r in rows:
        summary_rows.append(
            {
                "scenario": r.get("scenario", ""),
                "update_mode": r.get("update_mode", ""),
                "simulate_sync_ms": r.get("simulate_sync_ms", ""),
                "update_ms_ext_avg": r.get("update_ms_ext_avg", ""),
                "rollout_ms_avg": r.get("rollout_ms_avg", ""),
                "round_ms_avg": r.get("round_ms_avg", ""),
                "rollout_tok_s": r.get("rollout_tok_s", ""),
                "e2e_tok_s": r.get("e2e_tok_s", ""),
                "tokens_per_round": r.get("tokens_per_round", ""),
                "total_tokens_measured": r.get("total_tokens_measured", ""),
            }
        )
    write_csv(out_csv, summary_rows)

    by_name = {r["scenario"]: r for r in summary_rows}
    u = by_name.get("unified_apply")
    f = by_name.get("dual_fullsync_sim")
    l = by_name.get("dual_lora_sync_sim")
    speedup_vs_full = 0.0
    speedup_vs_lora = 0.0
    if u and f:
        f_ms = safe_float(f.get("round_ms_avg", "0"))
        u_ms = safe_float(u.get("round_ms_avg", "0"))
        speedup_vs_full = (f_ms / u_ms) if u_ms > 0 else 0.0
    if u and l:
        l_ms = safe_float(l.get("round_ms_avg", "0"))
        u_ms = safe_float(u.get("round_ms_avg", "0"))
        speedup_vs_lora = (l_ms / u_ms) if u_ms > 0 else 0.0

    lines: List[str] = []
    lines.append("# Stage 5.3 Measured E2E Loop Compare")
    lines.append("")
    lines.append(f"- Generated at: `{dt.datetime.now().isoformat(timespec='seconds')}`")
    lines.append(f"- Model: `{model_dir}`")
    lines.append(f"- Adapter: `{adapter_dir}`")
    lines.append(f"- rounds={args.rounds}, warmup={args.warmup}, prompt_len={args.prompt_len}, gen_len={args.gen_len}, num_candidates={args.num_candidates}")
    lines.append(f"- sync bandwidth assumption: `{args.sync_bandwidth_gib_per_s:.3f} GiB/s`")
    lines.append(f"- full_sync_ms_est=`{full_sync_ms:.6f}`, lora_sync_ms_est=`{lora_sync_ms:.6f}`")
    lines.append("")
    lines.append("| scenario | update_mode | simulate_sync_ms | update_ms_ext_avg | rollout_ms_avg | round_ms_avg | rollout_tok_s | e2e_tok_s |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in summary_rows:
        lines.append(
            f"| {r['scenario']} | {r['update_mode']} | {r['simulate_sync_ms']} | "
            f"{r['update_ms_ext_avg']} | {r['rollout_ms_avg']} | {r['round_ms_avg']} | "
            f"{r['rollout_tok_s']} | {r['e2e_tok_s']} |"
        )
    lines.append("")
    lines.append("## Key Point")
    lines.append(f"- Unified vs dual_fullsync(sim): speedup `{speedup_vs_full:.6f}x` (round_ms ratio).")
    lines.append(f"- Unified vs dual_lora_sync(sim): speedup `{speedup_vs_lora:.6f}x` (round_ms ratio).")
    lines.append("")
    lines.append("## Notes")
    lines.append("- `unified_apply` is measured in-process `apply_lora_adapter + rollout`.")
    lines.append("- dual-stack rows are simulated by adding per-round sync sleep with measured rollout path.")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[done] out_dir={out_dir}")
    print(f"- compare_csv: {out_csv}")
    print(f"- compare_md: {out_md}")


if __name__ == "__main__":
    main()

````

### File: docs/ember_task_checklist_v3.md (Section 5.3 only)

````md
### 5.3 统一后端优势证明

- [x] 双栈 vs 统一后端显存对比（model-only 视角）
- [x] 权重同步开销 vs 原地热更新延迟（transfer estimate + measured hot-update）
- [~] 端到端 rollout+update 吞吐对比（unified 闭环实测已完成；dual-stack 侧仍为 sync-sim）

**新增产出（2026-02-25）：**
- `benchmarks/rollout_update_loop_benchmark.cpp`
- `scripts/report/run_stage53_e2e_loop_compare.py`
- `scripts/report/run_stage53_unified_backend_advantage.py`
- `reports/stage53_unified_backend_advantage_4b_20260225_mainline_v1/stage53_summary.md`
- `reports/stage53_unified_backend_advantage_4b_20260225_mainline_v2/stage53_summary.md`
- `reports/stage53_e2e_loop_compare_4b_20260225_mainline_v1/stage53_e2e_compare.md`

**当前可引用数字（Qwen3-4B, 30 轮）：**
- 模型权重 footprint：dual-stack `14.985 GiB` vs unified `7.492 GiB`（节省 `50%`）
- 全量权重同步估算：`312.186 ms/round`（30 轮 `9365.592 ms`）
- 实测原地热更新：`28.206 ms/round`（30 轮 `846.180 ms`）
- E2E 吞吐估算（rollout-heavy: 100×8×128, rollout tok/s=47.586）：unified vs dual-fullsync `1.000132x`（几乎无差异，说明该配置下同步并非吞吐主瓶颈）
- E2E 闭环实测（512/64, candidates=4, rounds=6,warmup=2）：
  - `unified_apply`: `update_ms_ext_avg=42.666`, `e2e_tok_s=84.720`
  - `dual_fullsync_sim`: `round_ms_avg=3301.325`, `e2e_tok_s=77.545`
  - `dual_lora_sync_sim`: `round_ms_avg=2982.843`, `e2e_tok_s=85.824`
  - 结论：unified vs dual_fullsync(sim) `1.0925x`；vs dual_lora_sync(sim) `0.9871x`

---

## 6. 延后项（P1 数据锁定后再做）

````

---

## 3) 报告上下文（完整）

### Report: reports/stage53_unified_backend_advantage_4b_20260225_mainline_v2/stage53_summary.md

````md
# Stage 5.3 Unified Backend Advantage (Memory + Sync + E2E)

- Generated at: `2026-02-25T22:20:15`
- Model dir: `/home/dong/xilinx/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554`
- Adapter dir: `/home/dong/workspace/ember/reports/synthetic_lora_qwen3_4b_r8`
- Hot-update source: `/home/dong/workspace/ember/reports/stage31_lora_hot_update_4b_20260225_mainline_avg/stage31_lora_hot_update.csv`

| metric | value | notes |
| --- | --- | --- |
| model_weights_gib | 7.492473 | sum of model *.safetensors |
| adapter_weights_gib | 0.011022 | sum of adapter *.safetensors |
| dual_stack_model_only_gib | 14.984947 | 2x model copy (train + infer process) |
| unified_model_only_gib | 7.492473 | single model copy |
| model_memory_saved_gib | 7.492473 | 50.000% vs dual-stack model-only footprint |
| sync_full_model_ms_per_round_est | 312.186390 | assume 24.00 GiB/s |
| sync_lora_ms_per_round_est | 0.459237 | assume 24.00 GiB/s |
| hot_update_ms_per_round_measured | 28.206000 | from stage31_lora_hot_update.csv |
| sync_full_model_ms_30round_est | 9365.591686 | transfer-only estimate |
| sync_lora_ms_30round_est | 13.777120 | transfer-only estimate |
| hot_update_ms_30round_measured | 846.180000 | in-process adapter merge |
| e2e_rollout_tok_s_source | 47.586000 | ember/dual(0,1) from framework csv |
| e2e_rollout_tokens_per_round | 102400 | num_prompts=100, num_candidates=8, decode_steps=128 |
| e2e_dual_fullsync_round_ms_est | 2152205.600419 | rollout + full model sync |
| e2e_unified_round_ms_est | 2151921.620029 | rollout + measured in-process hot update |
| e2e_dual_fullsync_tok_s_est | 47.579097 | end-to-end throughput |
| e2e_unified_tok_s_est | 47.585376 | end-to-end throughput |
| e2e_unified_speedup_vs_dual_fullsync_x | 1.000132 | round-time ratio |
| e2e_dual_lora_sync_round_ms_est | 2151893.873267 | rollout + LoRA-size sync estimate |
| e2e_dual_lora_sync_tok_s_est | 47.585990 | end-to-end throughput |
| e2e_unified_speedup_vs_dual_lora_sync_x | 0.999987 | round-time ratio |

````

### Report: reports/stage53_e2e_loop_compare_4b_20260225_mainline_v1/stage53_e2e_compare.md

````md
# Stage 5.3 Measured E2E Loop Compare

- Generated at: `2026-02-25T22:32:49`
- Model: `/home/dong/xilinx/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554`
- Adapter: `/home/dong/workspace/ember/reports/synthetic_lora_qwen3_4b_r8`
- rounds=6, warmup=2, prompt_len=512, gen_len=64, num_candidates=4
- sync bandwidth assumption: `24.000 GiB/s`
- full_sync_ms_est=`312.186390`, lora_sync_ms_est=`0.459237`

| scenario | update_mode | simulate_sync_ms | update_ms_ext_avg | rollout_ms_avg | round_ms_avg | rollout_tok_s | e2e_tok_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| unified_apply | apply | 0.000000 | 42.666143 | 2979.054327 | 3021.720470 | 85.933310 | 84.719948 |
| dual_fullsync_sim | skip | 312.186390 | 0.000000 | 2989.138573 | 3301.324962 | 85.643403 | 77.544623 |
| dual_lora_sync_sim | skip | 0.459237 | 0.000000 | 2982.383327 | 2982.842564 | 85.837390 | 85.824174 |

## Key Point
- Unified vs dual_fullsync(sim): speedup `1.092532x` (round_ms ratio).
- Unified vs dual_lora_sync(sim): speedup `0.987134x` (round_ms ratio).

## Notes
- `unified_apply` is measured in-process `apply_lora_adapter + rollout`.
- dual-stack rows are simulated by adding per-round sync sleep with measured rollout path.

````

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

---

## 4) 风格参考

附件上传：`tutorial_01_life_of_a_token.md`

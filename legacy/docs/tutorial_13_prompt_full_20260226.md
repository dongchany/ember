# Tutorial #13 Prompt (Full, With Code + Reports)

> 生成日期：2026-02-26
> 用途：整份复制到新 chat 作为写作输入
> 标注：大文件（如 cuda_runtime.cpp）按“该篇相关段落”摘取，避免上下文爆炸

---

## 0) 系统 Prompt（先贴这段）

```text
我正在为我的开源项目 Ember 写一个系列教程。请帮我写第 13 篇。

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
请写第 13 篇：Cache 策略 — UpdateLocality / Prefix / Periodic / Hybrid。

## 本篇必须引用的代码与报告都在本消息里（不要再让我补充路径）

- 明确解释“为什么多轮 RL 放大 prefill 成本”
- 给出策略选择建议（何时选 UpdateLocality / Prefix / Hybrid）
```

---

## 2) 代码上下文（完整/相关段落）

### File: runtime/cache_policy.h

````h
#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace ember {

enum class CachePolicyType {
    NAIVE = 0,          // Recompute all layers every update.
    UPDATE_LOCALITY,    // Reuse prefix layers; recompute suffix layers.
    PERIODIC_REFRESH,   // UpdateLocality + periodic full refresh.
};

struct CachePolicyConfig {
    CachePolicyType type = CachePolicyType::NAIVE;
    int total_layers = 0;
    // Number of prefix layers to reuse across updates.
    // recompute starts from this layer (inclusive).
    int freeze_prefix_layers = 0;
    // Every k updates do full refresh. 0 disables periodic full refresh.
    int periodic_refresh_k = 0;
};

struct CachePolicyDecision {
    int update_step = 0;           // 1-based update index
    bool full_refresh = false;     // true => recompute all layers
    int recompute_layers = 0;
    int reused_layers = 0;
    float recompute_ratio = 1.0f;  // recompute_layers / total_layers
    // 1 => recompute this layer, 0 => reuse this layer
    std::vector<uint8_t> recompute_mask;
};

struct CachePolicyStats {
    int updates = 0;
    int full_refreshes = 0;
    int total_recompute_layers = 0;
    int total_reused_layers = 0;
    float avg_recompute_ratio = 1.0f;
};

class CachePolicyEngine {
public:
    CachePolicyEngine() = default;
    explicit CachePolicyEngine(const CachePolicyConfig& cfg) { configure(cfg); }

    void configure(const CachePolicyConfig& cfg) {
        config_ = cfg;
        if (config_.total_layers < 0) config_.total_layers = 0;
        if (config_.freeze_prefix_layers < 0) config_.freeze_prefix_layers = 0;
        if (config_.freeze_prefix_layers > config_.total_layers) {
            config_.freeze_prefix_layers = config_.total_layers;
        }
        if (config_.periodic_refresh_k < 0) config_.periodic_refresh_k = 0;
        reset();
    }

    void reset() {
        step_ = 0;
        stats_ = CachePolicyStats{};
        stats_.avg_recompute_ratio = (config_.total_layers > 0) ? 1.0f : 0.0f;
    }

    CachePolicyDecision on_policy_update() {
        CachePolicyDecision d;
        d.update_step = ++step_;
        d.recompute_mask.assign(static_cast<size_t>(config_.total_layers), static_cast<uint8_t>(1));

        const int total = config_.total_layers;
        const int freeze_prefix = std::clamp(config_.freeze_prefix_layers, 0, total);

        bool full_refresh = true;
        switch (config_.type) {
            case CachePolicyType::NAIVE:
                full_refresh = true;
                break;
            case CachePolicyType::UPDATE_LOCALITY:
                // First step full recompute; afterwards reuse prefix.
                full_refresh = (step_ <= 1);
                break;
            case CachePolicyType::PERIODIC_REFRESH:
                if (step_ <= 1) {
                    full_refresh = true;
                } else if (config_.periodic_refresh_k > 0 && (step_ % config_.periodic_refresh_k == 0)) {
                    full_refresh = true;
                } else {
                    full_refresh = false;
                }
                break;
            default:
                full_refresh = true;
                break;
        }

        if (!full_refresh) {
            for (int l = 0; l < freeze_prefix; ++l) {
                d.recompute_mask[static_cast<size_t>(l)] = static_cast<uint8_t>(0);
            }
        }

        d.full_refresh = full_refresh;
        int recompute = 0;
        for (uint8_t x : d.recompute_mask) recompute += (x ? 1 : 0);
        d.recompute_layers = recompute;
        d.reused_layers = total - recompute;
        d.recompute_ratio = (total > 0) ? static_cast<float>(recompute) / static_cast<float>(total) : 0.0f;

        stats_.updates += 1;
        stats_.full_refreshes += (full_refresh ? 1 : 0);
        stats_.total_recompute_layers += d.recompute_layers;
        stats_.total_reused_layers += d.reused_layers;
        if (stats_.updates > 0 && total > 0) {
            stats_.avg_recompute_ratio =
                static_cast<float>(stats_.total_recompute_layers) /
                static_cast<float>(stats_.updates * total);
        } else {
            stats_.avg_recompute_ratio = 0.0f;
        }
        return d;
    }

    const CachePolicyConfig& config() const { return config_; }
    const CachePolicyStats& stats() const { return stats_; }

    static std::string policy_name(CachePolicyType t) {
        switch (t) {
            case CachePolicyType::NAIVE: return "naive";
            case CachePolicyType::UPDATE_LOCALITY: return "update_locality";
            case CachePolicyType::PERIODIC_REFRESH: return "periodic_refresh";
            default: return "unknown";
        }
    }

private:
    CachePolicyConfig config_;
    CachePolicyStats stats_;
    int step_ = 0;
};

}  // namespace ember


````

### File: benchmarks/cache_policy_sim.cpp

````cpp
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "runtime/cache_policy.h"

namespace {

[[noreturn]] void die(const std::string& msg) {
    std::cerr << "error: " << msg << "\n";
    std::exit(1);
}

ember::CachePolicyType parse_policy(const std::string& s) {
    if (s == "naive") return ember::CachePolicyType::NAIVE;
    if (s == "update_locality") return ember::CachePolicyType::UPDATE_LOCALITY;
    if (s == "periodic_refresh") return ember::CachePolicyType::PERIODIC_REFRESH;
    die("unknown --policy: " + s);
}

}  // namespace

int main(int argc, char** argv) {
    std::string policy = "naive";
    int num_layers = 36;
    int rounds = 30;
    int freeze_layers = 18;
    int periodic_refresh_k = 10;
    std::string csv_path;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) die(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (arg == "-h" || arg == "--help") {
            std::cout
                << "Ember Cache Policy Simulator\n\n"
                << "Usage:\n"
                << "  " << argv[0] << " [options]\n\n"
                << "Options:\n"
                << "  --policy STR          naive|update_locality|periodic_refresh (default: naive)\n"
                << "  --num-layers N        model layers (default: 36)\n"
                << "  --rounds N            update rounds (default: 30)\n"
                << "  --freeze-layers N     reused prefix layers for locality (default: 18)\n"
                << "  --periodic-refresh-k N (default: 10)\n"
                << "  --csv PATH            output csv path\n";
            return 0;
        } else if (arg == "--policy") {
            policy = need("--policy");
        } else if (arg == "--num-layers") {
            num_layers = std::stoi(need("--num-layers"));
        } else if (arg == "--rounds") {
            rounds = std::stoi(need("--rounds"));
        } else if (arg == "--freeze-layers") {
            freeze_layers = std::stoi(need("--freeze-layers"));
        } else if (arg == "--periodic-refresh-k") {
            periodic_refresh_k = std::stoi(need("--periodic-refresh-k"));
        } else if (arg == "--csv") {
            csv_path = need("--csv");
        } else {
            die("unknown argument: " + arg);
        }
    }

    if (num_layers <= 0) die("--num-layers must be > 0");
    if (rounds <= 0) die("--rounds must be > 0");
    if (freeze_layers < 0 || freeze_layers > num_layers) die("--freeze-layers out of range");
    if (periodic_refresh_k < 0) die("--periodic-refresh-k must be >= 0");

    ember::CachePolicyConfig cfg;
    cfg.type = parse_policy(policy);
    cfg.total_layers = num_layers;
    cfg.freeze_prefix_layers = freeze_layers;
    cfg.periodic_refresh_k = periodic_refresh_k;

    ember::CachePolicyEngine engine(cfg);

    std::vector<std::string> lines;
    lines.push_back("round,policy,num_layers,freeze_layers,periodic_refresh_k,full_refresh,recompute_layers,reused_layers,recompute_ratio");

    for (int r = 1; r <= rounds; ++r) {
        ember::CachePolicyDecision d = engine.on_policy_update();
        std::ostringstream row;
        row << d.update_step << ","
            << policy << ","
            << num_layers << ","
            << freeze_layers << ","
            << periodic_refresh_k << ","
            << (d.full_refresh ? 1 : 0) << ","
            << d.recompute_layers << ","
            << d.reused_layers << ","
            << std::fixed << std::setprecision(6) << d.recompute_ratio;
        lines.push_back(row.str());
    }

    std::ostringstream summary;
    const auto& st = engine.stats();
    summary << "# summary"
            << "\npolicy=" << policy
            << "\nupdates=" << st.updates
            << "\nfull_refreshes=" << st.full_refreshes
            << "\navg_recompute_ratio=" << std::fixed << std::setprecision(6) << st.avg_recompute_ratio
            << "\ntotal_recompute_layers=" << st.total_recompute_layers
            << "\ntotal_reused_layers=" << st.total_reused_layers
            << "\n";

    if (!csv_path.empty()) {
        std::ofstream out(csv_path);
        if (!out.is_open()) die("failed to open csv path: " + csv_path);
        for (const auto& ln : lines) out << ln << "\n";
        out << summary.str();
    } else {
        for (const auto& ln : lines) std::cout << ln << "\n";
        std::cout << summary.str();
    }

    return 0;
}


````

### File: scripts/report/run_stage33_cache_policy.py

````py
#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
from pathlib import Path
from typing import Dict, List

from common_report import die, run_cmd, safe_float, write_csv


def read_policy_csv(path: Path) -> List[Dict[str, str]]:
    lines: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            if "=" in ln and "," not in ln:
                continue
            lines.append(ln)
    if not lines:
        return []
    reader = csv.DictReader(lines)
    return list(reader)


def safe_int(v: str, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return default


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 3.3 cache policy simulation report.")
    ap.add_argument("--num-layers", type=int, default=36)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--freeze-layers", type=int, default=18)
    ap.add_argument("--periodic-refresh-k", type=int, default=10)
    ap.add_argument("--policies", type=str, default="naive,update_locality,periodic_refresh")
    ap.add_argument("--build-dir", type=str, default="build")
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    if args.num_layers <= 0:
        die("--num-layers must be > 0")
    if args.rounds <= 0:
        die("--rounds must be > 0")
    if args.freeze_layers < 0 or args.freeze_layers > args.num_layers:
        die("--freeze-layers out of range")
    if args.periodic_refresh_k < 0:
        die("--periodic-refresh-k must be >= 0")

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    if not policies:
        die("--policies empty")

    repo = Path.cwd()
    sim_bin = (repo / args.build_dir / "ember_cache_policy_sim").resolve()
    if not sim_bin.exists():
        die(f"missing binary: {sim_bin}")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (repo / "reports" / f"stage33_cache_policy_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    merged_rows: List[Dict[str, str]] = []
    summary_rows: List[Dict[str, str]] = []

    for policy in policies:
        run_csv = out_dir / f"stage33_{policy}.csv"
        cmd = [
            str(sim_bin),
            "--policy",
            policy,
            "--num-layers",
            str(args.num_layers),
            "--rounds",
            str(args.rounds),
            "--freeze-layers",
            str(args.freeze_layers),
            "--periodic-refresh-k",
            str(args.periodic_refresh_k),
            "--csv",
            str(run_csv),
        ]
        p = run_cmd(cmd, cwd=repo, log_path=logs_dir / f"{policy}.log", check=False)
        if p.returncode != 0:
            die(f"policy sim failed for {policy}; see {logs_dir / (policy + '.log')}")

        rows = read_policy_csv(run_csv)
        if not rows:
            die(f"empty csv for policy={policy}: {run_csv}")
        merged_rows.extend(rows)

        avg_ratio = sum(safe_float(r.get("recompute_ratio", "0")) for r in rows) / float(len(rows))
        refreshes = sum(safe_int(r.get("full_refresh", "0")) for r in rows)
        first = rows[0]
        summary_rows.append(
            {
                "policy": policy,
                "rounds": str(len(rows)),
                "num_layers": first.get("num_layers", str(args.num_layers)),
                "freeze_layers": first.get("freeze_layers", str(args.freeze_layers)),
                "periodic_refresh_k": first.get("periodic_refresh_k", str(args.periodic_refresh_k)),
                "full_refreshes": str(refreshes),
                "avg_recompute_ratio": f"{avg_ratio:.6f}",
                "avg_reused_ratio": f"{(1.0 - avg_ratio):.6f}",
                "cache_hit_rate_proxy": f"{(1.0 - avg_ratio):.6f}",
                "cache_miss_rate_proxy": f"{avg_ratio:.6f}",
                "recompute_layers_total": str(sum(safe_int(r.get("recompute_layers", "0")) for r in rows)),
                "reused_layers_total": str(sum(safe_int(r.get("reused_layers", "0")) for r in rows)),
            }
        )

    merged_csv = out_dir / "stage33_policy_per_round.csv"
    summary_csv = out_dir / "stage33_policy_summary.csv"
    summary_md = out_dir / "stage33_summary.md"
    p1_input_md = out_dir / "stage33_p1_input.md"

    write_csv(merged_csv, merged_rows)
    write_csv(summary_csv, summary_rows)

    lines = [
        "# Stage 3.3 Cache Policy Simulation",
        "",
        f"- Generated at: `{dt.datetime.now().isoformat(timespec='seconds')}`",
        f"- num_layers={args.num_layers}, rounds={args.rounds}, freeze_layers={args.freeze_layers}, periodic_refresh_k={args.periodic_refresh_k}",
        "",
        "| policy | full_refreshes | avg_recompute_ratio | cache_hit_rate_proxy | cache_miss_rate_proxy | recompute_layers_total |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['policy']} | {r['full_refreshes']} | {r['avg_recompute_ratio']} | "
            f"{r['cache_hit_rate_proxy']} | {r['cache_miss_rate_proxy']} | {r['recompute_layers_total']} |"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    best = min(summary_rows, key=lambda r: safe_float(r.get("avg_recompute_ratio", "1")))
    p1_lines = [
        f"# Stage3.3 P1 Input ({dt.date.today().isoformat()})",
        "",
        f"Setting: num_layers={args.num_layers}, rounds={args.rounds}, freeze_layers={args.freeze_layers}, periodic_refresh_k={args.periodic_refresh_k}",
        "",
        "## Policy summary",
    ]
    for r in summary_rows:
        p1_lines.append(
            f"- {r['policy']}: avg_recompute_ratio={r['avg_recompute_ratio']}, "
            f"cache_hit_rate_proxy={r['cache_hit_rate_proxy']}, cache_miss_rate_proxy={r['cache_miss_rate_proxy']}, "
            f"recompute_layers_total={r['recompute_layers_total']}, full_refreshes={r['full_refreshes']}"
        )
    p1_lines += [
        "",
        f"Best compute-saving policy in this sweep: {best['policy']} (avg_recompute_ratio={best['avg_recompute_ratio']}).",
    ]
    p1_input_md.write_text("\n".join(p1_lines) + "\n", encoding="utf-8")

    print(f"[done] out_dir={out_dir}")
    print(f"- per-round: {merged_csv}")
    print(f"- summary: {summary_csv}")
    print(f"- md: {summary_md}")
    print(f"- p1 input: {p1_input_md}")


if __name__ == "__main__":
    main()

````

### File: scripts/report/run_stage1_prefix_cache.py

````py
#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from common_report import die, read_csv, run_cmd, safe_float, split_ints, write_csv


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


def write_md(path: Path, model_dir: Path, rows: List[Dict[str, str]], prompt_len: int, num_docs: int) -> None:
    lines: List[str] = []
    lines.append("# Stage 1.3 Prefix Cache Sweep")
    lines.append("")
    lines.append(f"- Model: `{model_dir}`")
    lines.append(f"- Prompt length: `{prompt_len}`")
    lines.append(f"- Docs per run: `{num_docs}`")
    lines.append(f"- Generated at: `{dt.datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")
    lines.append("| prefix_len | suffix_len | no_cache_total_ms | with_cache_total_ms | speedup_x | savings_% | theoretical_savings_% |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    r.get("prefix_len", ""),
                    r.get("suffix_len", ""),
                    r.get("no_cache_total_ms", ""),
                    r.get("with_cache_total_ms", ""),
                    r.get("speedup_x", ""),
                    r.get("savings_pct", ""),
                    r.get("theoretical_savings_pct", ""),
                ]
            )
            + " |"
        )

    if rows:
        best = max(rows, key=lambda x: safe_float(x.get("savings_pct", "0")))
        mid = None
        for r in rows:
            if int(r.get("prefix_len", "0")) == 1000:
                mid = r
                break
        if mid is None:
            for r in rows:
                if int(r.get("prefix_len", "0")) == 1024:
                    mid = r
                    break
        lines.append("")
        lines.append("## Key Point")
        lines.append(
            f"- Best measured savings: prefix_len={best.get('prefix_len','')} "
            f"-> `{best.get('savings_pct','')}%` (`{best.get('speedup_x','')}x`)."
        )
        if mid is not None:
            lines.append(
                f"- Shared-prefix ~1k tokens result: savings `{mid.get('savings_pct','')}%`, "
                f"speedup `{mid.get('speedup_x','')}x`."
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Stage 1.3 prefix cache sweep.")
    ap.add_argument("--model", type=str, required=True, help="model path or HF model id")
    ap.add_argument("--gpus", type=str, default="0,1", help="GPU ids, e.g. 0,1")
    ap.add_argument("--split", type=str, default="9,27", help="2-GPU split A,B")
    ap.add_argument("--prompt-len", type=int, default=2048)
    ap.add_argument("--prefix-lens", type=str, default="0,256,512,768,1024,1280,1536")
    ap.add_argument("--num-docs", type=int, default=100)
    ap.add_argument("--chunk-len", type=int, default=512)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--overlap", action="store_true", default=True)
    ap.add_argument("--no-overlap", dest="overlap", action="store_false")
    ap.add_argument("--pipeline", action="store_true", default=False)
    ap.add_argument("--build-dir", type=str, default="build")
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    if args.prompt_len <= 0:
        die("--prompt-len must be > 0")
    if args.num_docs <= 0:
        die("--num-docs must be > 0")
    if args.iters <= 0:
        die("--iters must be > 0")
    if args.warmup < 0:
        die("--warmup must be >= 0")

    prefix_lens = split_ints(args.prefix_lens)
    if not prefix_lens:
        die("--prefix-lens is empty")
    for p in prefix_lens:
        if p < 0 or p > args.prompt_len:
            die(f"prefix_len out of range: {p}")

    model_dir = resolve_model_dir(args.model)
    repo = Path.cwd()
    bench_bin = (repo / args.build_dir / "ember_prefix_cache_benchmark").resolve()
    if not bench_bin.exists():
        die(f"missing binary: {bench_bin}")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (repo / "reports" / f"stage1_prefix_cache_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    failed: List[Dict[str, str]] = []

    for idx, prefix_len in enumerate(prefix_lens, start=1):
        run_csv = out_dir / f"stage13_raw_{idx:03d}_prefix_{prefix_len}.csv"
        cmd = [
            str(bench_bin),
            "--model",
            str(model_dir),
            "--gpus",
            args.gpus,
            "--split",
            args.split,
            "--prompt-len",
            str(args.prompt_len),
            "--prefix-len",
            str(prefix_len),
            "--num-docs",
            str(args.num_docs),
            "--chunk-len",
            str(args.chunk_len),
            "--iters",
            str(args.iters),
            "--warmup",
            str(args.warmup),
            "--seed",
            str(args.seed),
            "--csv",
            str(run_csv),
        ]
        if args.overlap:
            cmd.append("--overlap")
        else:
            cmd.append("--no-overlap")
        if args.pipeline:
            cmd.append("--pipeline")
        else:
            cmd.append("--no-pipeline")

        print(f"[run {idx}] prefix_len={prefix_len}")
        p = run_cmd(cmd, cwd=repo, log_path=logs_dir / f"run_{idx:03d}.log", check=False)
        if p.returncode != 0:
            failed.append(
                {
                    "prefix_len": str(prefix_len),
                    "return_code": str(p.returncode),
                    "log_path": str(logs_dir / f"run_{idx:03d}.log"),
                }
            )
            print(f"[fail {idx}] rc={p.returncode}")
            continue

        run_rows = read_csv(run_csv)
        if not run_rows:
            failed.append(
                {
                    "prefix_len": str(prefix_len),
                    "return_code": "empty_csv",
                    "log_path": str(logs_dir / f"run_{idx:03d}.log"),
                }
            )
            print(f"[fail {idx}] empty csv")
            continue
        rows.append(run_rows[0])

    rows.sort(key=lambda r: int(r.get("prefix_len", "0")))
    summary_csv = out_dir / "stage13_prefix_cache_sweep.csv"
    summary_md = out_dir / "stage13_prefix_cache_summary.md"
    failures_csv = out_dir / "stage13_failures.csv"
    p1_input_md = out_dir / "stage13_p1_input.md"

    if rows:
        write_csv(summary_csv, rows)
        write_md(summary_md, model_dir=model_dir, rows=rows, prompt_len=args.prompt_len, num_docs=args.num_docs)
        key_1k = None
        for r in rows:
            if int(r.get("prefix_len", "0")) in (1000, 1024):
                key_1k = r
                break
        best = max(rows, key=lambda x: safe_float(x.get("savings_pct", "0")))
        lines = [
            f"# Stage1.3 P1 Input ({dt.date.today().isoformat()})",
            "",
            f"Model: {model_dir}",
            f"Setting: prompt_len={args.prompt_len}, num_docs={args.num_docs}, mode={'overlap' if args.overlap else 'no_overlap'}",
            "",
            "## Best point",
            f"- prefix_len={best.get('prefix_len','')}, savings={best.get('savings_pct','')}%, speedup={best.get('speedup_x','')}x",
        ]
        if key_1k is not None:
            lines += [
                "",
                "## Shared prefix ~1k tokens",
                f"- prefix_len={key_1k.get('prefix_len','')}, savings={key_1k.get('savings_pct','')}%, speedup={key_1k.get('speedup_x','')}x",
                f"- no_cache_total_ms={key_1k.get('no_cache_total_ms','')}, with_cache_total_ms={key_1k.get('with_cache_total_ms','')}",
            ]
        p1_input_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with failures_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["prefix_len", "return_code", "log_path"])
        w.writeheader()
        for r in failed:
            w.writerow(r)

    print("[done] stage1.3 prefix cache sweep")
    if rows:
        print(f"- sweep csv: {summary_csv}")
        print(f"- summary md: {summary_md}")
        print(f"- p1 input md: {p1_input_md}")
    else:
        print("- no successful runs")
    print(f"- failures: {failures_csv} ({len(failed)} failed runs)")
    print(f"- logs: {logs_dir}")


if __name__ == "__main__":
    main()

````

---

## 3) 报告上下文（完整）

### Report: reports/stage33_cache_policy_20260225_mainline/stage33_summary.md

````md
# Stage 3.3 Cache Policy Simulation

- Generated at: `2026-02-25T13:08:51`
- num_layers=36, rounds=30, freeze_layers=18, periodic_refresh_k=10

| policy | full_refreshes | avg_recompute_ratio | cache_hit_rate_proxy | cache_miss_rate_proxy | recompute_layers_total |
| --- | --- | --- | --- | --- | --- |
| naive | 30 | 1.000000 | 0.000000 | 1.000000 | 1080 |
| update_locality | 1 | 0.516667 | 0.483333 | 0.516667 | 558 |
| periodic_refresh | 4 | 0.566667 | 0.433333 | 0.566667 | 612 |

````

### Report: reports/stage42_locality_sweep_4b_20260225_mainline/stage42_locality_sweep.md

````md
# Stage 4.2 Locality Sweep (Simulation)

- Generated at: `2026-02-25T10:32:41`
- Base row: split=9+27, mode=overlap, prompt_len=2048, decode_steps=128
- Requests/round: 800, rounds=30, periodic_refresh_k=10

| recompute_ratio | freeze_pct_proxy | cumulative_gpu_hours | speedup_vs_naive_x | reduction_vs_naive_% |
| --- | --- | --- | --- | --- |
| 1.000 | 0.0 | 36.737493 | 1.0000 | -0.000 |
| 0.750 | 25.0 | 35.495086 | 1.0350 | 3.382 |
| 0.500 | 50.0 | 34.252679 | 1.0725 | 6.764 |
| 0.250 | 75.0 | 33.010272 | 1.1129 | 10.146 |
| 0.000 | 100.0 | 31.767865 | 1.1564 | 13.527 |

````

### Report: reports/stage43_strategy_table_4b_20260225_mainline_v2_refactor/stage43_strategy_table.md

````md
# Stage 4.3 Strategy Table (Simulation Aggregate)

- Generated at: `2026-02-25T21:33:27`
- Quality threshold (proxy): `0.300000`

| scenario | strategy | cumulative_gpu_hours | reduction_vs_naive_% | avg_recompute_ratio | cache_hit_rate_proxy | quality_proxy_delta_max | quality_ok |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2048/128 | naive | 36.737493 | 0.000 | 1.000000 | 0.000000 |  |  |
| 2048/128 | prefix_only | 34.003876 | 7.441 |  |  |  |  |
| 2048/128 | update_locality | 33.965970 | 7.544 | 0.516667 | 0.483333 | 0.228125 | 1 |
| 2048/128 | periodic_refresh | 34.252679 | 6.764 | 0.566667 | 0.433333 | 0.209375 | 1 |
| 2048/128 | hybrid | 32.553601 | 11.389 | 0.516667 | 0.483333 | 0.228125 | 1 |
| 4096/64 | naive | 31.958787 | 0.000 | 1.000000 | 0.000000 |  |  |
| 4096/64 | prefix_only | 26.992878 | 15.538 |  |  |  |  |
| 4096/64 | update_locality | 24.947766 | 21.938 | 0.516667 | 0.483333 | 0.228125 | 1 |
| 4096/64 | periodic_refresh | 25.673044 | 19.668 | 0.566667 | 0.433333 | 0.209375 | 1 |
| 4096/64 | hybrid | 22.382047 | 29.966 | 0.516667 | 0.483333 | 0.228125 | 1 |

````

### Report: reports/stage1_prefix_cache_4b_20260225_mainline/stage13_prefix_cache_summary.md

````md
# Stage 1.3 Prefix Cache Sweep

- Model: `/home/dong/xilinx/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554`
- Prompt length: `2048`
- Docs per run: `100`
- Generated at: `2026-02-25T09:34:47`

| prefix_len | suffix_len | no_cache_total_ms | with_cache_total_ms | speedup_x | savings_% | theoretical_savings_% |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 2048 | 35957.866 | 35928.545 | 1.001 | 0.082 | 0.000 |
| 256 | 1792 | 36479.788 | 32984.744 | 1.106 | 9.581 | 12.375 |
| 512 | 1536 | 36839.712 | 29553.679 | 1.247 | 19.778 | 24.750 |
| 768 | 1280 | 37148.629 | 26100.931 | 1.423 | 29.739 | 37.125 |
| 1024 | 1024 | 37385.412 | 22677.374 | 1.649 | 39.342 | 49.500 |
| 1280 | 768 | 37400.463 | 18012.815 | 2.076 | 51.838 | 61.875 |
| 1536 | 512 | 37583.473 | 13082.734 | 2.873 | 65.190 | 74.250 |

## Key Point
- Best measured savings: prefix_len=1536 -> `65.190%` (`2.873x`).
- Shared-prefix ~1k tokens result: savings `39.342%`, speedup `1.649x`.

````

---

## 4) 风格参考

附件上传：`tutorial_01_life_of_a_token.md`

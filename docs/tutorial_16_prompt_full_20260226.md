# Tutorial #16 Prompt (Full, With Code + Reports)

> 生成日期：2026-02-26
> 用途：整份复制到新 chat 作为写作输入
> 标注：大文件（如 cuda_runtime.cpp）按“该篇相关段落”摘取，避免上下文爆炸

---

## 0) 系统 Prompt（先贴这段）

```text
我正在为我的开源项目 Ember 写一个系列教程。请帮我写第 16 篇。

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
请写第 16 篇：Best-of-N — 不训练也能提升，pass@k 的数学。

## 本篇必须引用的代码与报告都在本消息里（不要再让我补充路径）

- 从组合概率角度推导 pass@k 与 N 的关系（尽量详细）
- 解释 first vs best 两套指标为何都要看
```

---

## 2) 代码上下文（完整/相关段落）

### File: scripts/train/run_stage52_best_of_n_extraction.py

````py
#!/usr/bin/env python3
import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common_train import die, dtype_from_name, load_json, load_jsonl, render_chat, write_csv


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    start = -1
    depth = 0
    for i, ch in enumerate(text):
        if ch == "{":
            if start < 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    frag = text[start : i + 1]
                    try:
                        obj = json.loads(frag)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        return None
    return None


def canonical_str(v: Any) -> str:
    if isinstance(v, str):
        return v.strip().lower()
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    return str(v).strip().lower()


def type_ok(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    return True


def evaluate_candidate(
    *,
    pred_text: str,
    gold_obj: Dict[str, Any],
    schema_fields: Dict[str, str],
    field_weights: Dict[str, float],
    required_fields: List[str],
    reward_mode: str,
) -> Dict[str, Any]:
    pred_json = extract_first_json_object(pred_text)
    parse_ok = pred_json is not None
    if pred_json is None:
        pred_json = {}

    missing_required = sum(1 for f in required_fields if f not in pred_json)
    # Score only fields that are present in gold (supports partial/optional target rows).
    eval_fields = [f for f in gold_obj.keys() if f in schema_fields]
    type_mismatch = 0
    exact_match = 0
    field_total = 0
    weighted_num = 0.0
    weighted_den = 0.0
    for f in eval_fields:
        expected_t = schema_fields[f]
        wt = float(field_weights.get(f, 1.0))
        field_total += 1
        weighted_den += wt
        matched = 0.0
        if f in pred_json:
            ok_t = type_ok(pred_json[f], expected_t)
            if not ok_t:
                type_mismatch += 1
            if ok_t and canonical_str(pred_json[f]) == canonical_str(gold_obj.get(f, None)):
                exact_match += 1
                matched = 1.0
        weighted_num += wt * matched

    schema_ok = (missing_required == 0 and type_mismatch == 0)
    field_acc = (exact_match / field_total) if field_total > 0 else 0.0
    weighted_acc = (weighted_num / weighted_den) if weighted_den > 0 else 0.0
    exact_all = (field_total > 0 and exact_match == field_total and parse_ok and schema_ok)

    if reward_mode == "binary":
        reward = 1.0 if exact_all else 0.0
    elif reward_mode == "weighted":
        reward = weighted_acc
    elif reward_mode == "decomposed":
        reward = (0.2 if parse_ok else 0.0) + (0.2 if schema_ok else 0.0) + (0.6 * weighted_acc)
    else:
        die(f"unknown reward mode: {reward_mode}")
        raise AssertionError("unreachable")

    return {
        "parse_ok": 1 if parse_ok else 0,
        "schema_ok": 1 if schema_ok else 0,
        "missing_required": missing_required,
        "type_mismatch": type_mismatch,
        "field_exact_match": exact_match,
        "field_total": field_total,
        "field_acc": field_acc,
        "weighted_acc": weighted_acc,
        "exact_all": 1 if exact_all else 0,
        "reward": reward,
    }


def build_prompt(
    tokenizer,
    prompt: str,
    force_json_output: bool,
    schema_fields: Dict[str, str],
    schema_key_hint: bool,
) -> str:
    if force_json_output:
        hint = ""
        if schema_key_hint and schema_fields:
            pairs = ", ".join(f"{k}({schema_fields[k]})" for k in sorted(schema_fields.keys()))
            hint = f"字段名必须严格从以下列表中选择（不要使用同义词字段名）: {pairs}\n"
        prompt = (
            "请只输出一个合法 JSON 对象，不要输出解释、推理过程或代码块标记。\n"
            f"{hint}"
            "如果某字段无法确定，请在 JSON 中省略该字段。\n\n"
            f"{prompt}"
        )
    return render_chat(tokenizer, prompt=prompt, response=None, add_generation_prompt=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 5.2 Best-of-N extraction baseline.")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--adapter", type=str, default="", help="optional PEFT LoRA adapter dir for inference")
    ap.add_argument("--dataset-jsonl", type=str, required=True, help='rows: {"id","prompt","gold"}')
    ap.add_argument("--schema-json", type=str, required=True, help='{"required":[...],"fields":{"k":"string"}}')
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--num-candidates", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--decode-mode", type=str, default="sample", choices=["sample", "greedy"])
    ap.add_argument("--force-json-output", action="store_true", default=False)
    ap.add_argument(
        "--schema-key-hint",
        action="store_true",
        default=False,
        help="inject explicit schema key contract into prompt when --force-json-output is enabled",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reward-mode", type=str, default="weighted", choices=["binary", "weighted", "decomposed"])
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    if args.num_candidates <= 0:
        die("--num-candidates must be > 0")
    if args.max_new_tokens <= 0:
        die("--max-new-tokens must be > 0")

    model_dir = Path(args.model).expanduser().resolve()
    adapter_dir = Path(args.adapter).expanduser().resolve() if args.adapter.strip() else None
    ds_path = Path(args.dataset_jsonl).expanduser().resolve()
    schema_path = Path(args.schema_json).expanduser().resolve()
    for p in [model_dir, ds_path, schema_path]:
        if not p.exists():
            die(f"missing path: {p}")
    if adapter_dir is not None and not adapter_dir.exists():
        die(f"adapter path not found: {adapter_dir}")

    data = load_jsonl(ds_path)
    if not data:
        die("empty dataset")
    if args.max_samples > 0:
        data = data[: args.max_samples]
    for i, r in enumerate(data):
        if "id" not in r or "prompt" not in r or "gold" not in r:
            die(f"dataset row {i} missing required keys id/prompt/gold")
        if not isinstance(r["gold"], dict):
            die(f"dataset row {i} gold must be object")

    schema = load_json(schema_path)
    fields = schema.get("fields", {})
    required = schema.get("required", [])
    weights = schema.get("field_weights", {})
    if not isinstance(fields, dict) or not isinstance(required, list):
        die("schema invalid; need fields object and required list")
    if not isinstance(weights, dict):
        die("schema invalid; field_weights must be an object if provided")
    schema_fields: Dict[str, str] = {str(k): str(v) for k, v in fields.items()}
    required_fields = [str(x) for x in required]
    field_weights: Dict[str, float] = {k: 1.0 for k in schema_fields.keys()}
    for k, v in weights.items():
        ks = str(k)
        if ks not in schema_fields:
            die(f"schema invalid; field_weights has unknown field: {ks}")
        try:
            wv = float(v)
        except Exception:
            die(f"schema invalid; field_weights[{ks}] must be number")
        if wv < 0:
            die(f"schema invalid; field_weights[{ks}] must be >= 0")
        field_weights[ks] = wv

    dtype = dtype_from_name(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=dtype,
    )
    if adapter_dir is not None:
        try:
            from peft import PeftModel
        except Exception as ex:
            die(f"failed to import peft for --adapter: {ex}")
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    model.eval().to(args.device)

    per_candidate: List[Dict[str, str]] = []
    best_rows: List[Dict[str, str]] = []

    pass1 = 0
    passn = 0
    best_pass = 0
    mean_reward_first = 0.0
    mean_reward_best = 0.0
    mean_weighted_first = 0.0
    mean_weighted_best = 0.0

    for i, sample in enumerate(data):
        sid = str(sample["id"])
        prompt = str(sample["prompt"])
        gold = sample["gold"]
        full_prompt = build_prompt(
            tokenizer,
            prompt,
            force_json_output=args.force_json_output,
            schema_fields=schema_fields,
            schema_key_hint=args.schema_key_hint,
        )
        enc = tokenizer(full_prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(args.device)
        attn = enc["attention_mask"].to(args.device)

        cand_scores: List[Tuple[int, Dict[str, Any], str]] = []
        for c in range(args.num_candidates):
            seed = args.seed + i * 1000 + c
            torch.manual_seed(seed)
            if args.decode_mode == "greedy":
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attn,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            else:
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attn,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    pad_token_id=tokenizer.eos_token_id,
                )
            gen = out[0, input_ids.shape[1] :]
            text = tokenizer.decode(gen, skip_special_tokens=True)
            ev = evaluate_candidate(
                pred_text=text,
                gold_obj=gold,
                schema_fields=schema_fields,
                field_weights=field_weights,
                required_fields=required_fields,
                reward_mode=args.reward_mode,
            )
            cand_scores.append((c, ev, text))
            per_candidate.append(
                {
                    "id": sid,
                    "candidate_id": str(c),
                    "reward": f"{float(ev['reward']):.6f}",
                    "field_acc": f"{float(ev['field_acc']):.6f}",
                    "weighted_acc": f"{float(ev['weighted_acc']):.6f}",
                    "exact_all": str(ev["exact_all"]),
                    "parse_ok": str(ev["parse_ok"]),
                    "schema_ok": str(ev["schema_ok"]),
                    "missing_required": str(ev["missing_required"]),
                    "type_mismatch": str(ev["type_mismatch"]),
                    "output": text.replace("\n", "\\n"),
                }
            )

        first_ev = cand_scores[0][1]
        best = max(
            cand_scores,
            key=lambda x: (
                float(x[1]["reward"]),
                float(x[1]["weighted_acc"]),
                float(x[1]["field_acc"]),
                int(x[1]["exact_all"]),
            ),
        )
        any_exact = any(int(x[1]["exact_all"]) == 1 for x in cand_scores)

        pass1 += int(first_ev["exact_all"])
        passn += 1 if any_exact else 0
        best_pass += int(best[1]["exact_all"])
        mean_reward_first += float(first_ev["reward"])
        mean_reward_best += float(best[1]["reward"])
        mean_weighted_first += float(first_ev["weighted_acc"])
        mean_weighted_best += float(best[1]["weighted_acc"])

        best_rows.append(
            {
                "id": sid,
                "best_candidate_id": str(best[0]),
                "best_reward": f"{float(best[1]['reward']):.6f}",
                "best_weighted_acc": f"{float(best[1]['weighted_acc']):.6f}",
                "best_field_acc": f"{float(best[1]['field_acc']):.6f}",
                "best_exact_all": str(best[1]["exact_all"]),
                "pass_at_1": str(first_ev["exact_all"]),
                "pass_at_n": "1" if any_exact else "0",
            }
        )

    n = len(data)
    summary = {
        "num_samples": n,
        "num_candidates": args.num_candidates,
        "adapter": str(adapter_dir) if adapter_dir is not None else "",
        "pass_at_1": pass1 / n,
        "pass_at_n": passn / n,
        "best_of_n_exact_rate": best_pass / n,
        "mean_reward_first": mean_reward_first / n,
        "mean_reward_best": mean_reward_best / n,
        "mean_weighted_acc_first": mean_weighted_first / n,
        "mean_weighted_acc_best": mean_weighted_best / n,
    }

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (Path.cwd() / "reports" / f"stage52_best_of_n_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cand_csv = out_dir / "stage52_candidates.csv"
    best_csv = out_dir / "stage52_best_choice.csv"
    summary_json = out_dir / "stage52_summary.json"
    summary_md = out_dir / "stage52_summary.md"

    write_csv(cand_csv, per_candidate)
    write_csv(best_csv, best_rows)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Stage 5.2 Best-of-N Extraction Baseline",
        "",
        f"- Generated at: `{dt.datetime.now().isoformat(timespec='seconds')}`",
        f"- Reward mode: `{args.reward_mode}`",
        f"- Samples: `{n}`, N=`{args.num_candidates}`",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| pass@1 | {summary['pass_at_1']:.6f} |",
        f"| pass@N | {summary['pass_at_n']:.6f} |",
        f"| best_of_n_exact_rate | {summary['best_of_n_exact_rate']:.6f} |",
        f"| mean_reward_first | {summary['mean_reward_first']:.6f} |",
        f"| mean_reward_best | {summary['mean_reward_best']:.6f} |",
        f"| mean_weighted_acc_first | {summary['mean_weighted_acc_first']:.6f} |",
        f"| mean_weighted_acc_best | {summary['mean_weighted_acc_best']:.6f} |",
    ]
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("[done] stage5.2 best-of-n baseline")
    print(f"- candidates csv: {cand_csv}")
    print(f"- best csv: {best_csv}")
    print(f"- summary md: {summary_md}")
    print(f"- summary json: {summary_json}")


if __name__ == "__main__":
    main()

````

### File: scripts/report/run_stage52_baseline_compare.py

````py
#!/usr/bin/env python3
import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Dict, List, Tuple

from common_report import die, write_csv


def parse_spec(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        die(f"invalid --input spec (need label=path): {spec}")
    label, path = spec.split("=", 1)
    label = label.strip()
    p = Path(path.strip()).expanduser().resolve()
    if not label:
        die(f"empty label in --input spec: {spec}")
    if not p.exists():
        die(f"summary json not found: {p}")
    return label, p


def load_summary(path: Path) -> Dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as ex:
        die(f"failed to parse json {path}: {ex}")
    if not isinstance(data, dict):
        die(f"invalid summary json (not object): {path}")
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate stage5.2 baseline summaries into one compare table.")
    ap.add_argument(
        "--input",
        action="append",
        default=[],
        help="label=/abs/or/rel/path/to/stage52_summary.json; repeat for multiple rows",
    )
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    if len(args.input) < 2:
        die("need at least 2 --input entries")

    specs = [parse_spec(x) for x in args.input]
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (
        Path.cwd() / "reports" / f"stage52_baseline_compare_{ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    for label, path in specs:
        s = load_summary(path)
        rows.append(
            {
                "label": label,
                "summary_json": str(path),
                "num_samples": str(int(s.get("num_samples", 0))),
                "num_candidates": str(int(s.get("num_candidates", 0))),
                "pass_at_1": f"{float(s.get('pass_at_1', 0.0)):.6f}",
                "pass_at_n": f"{float(s.get('pass_at_n', 0.0)):.6f}",
                "best_of_n_exact_rate": f"{float(s.get('best_of_n_exact_rate', 0.0)):.6f}",
                "mean_reward_first": f"{float(s.get('mean_reward_first', 0.0)):.6f}",
                "mean_reward_best": f"{float(s.get('mean_reward_best', 0.0)):.6f}",
                "mean_weighted_acc_first": f"{float(s.get('mean_weighted_acc_first', s.get('mean_reward_first', 0.0))):.6f}",
                "mean_weighted_acc_best": f"{float(s.get('mean_weighted_acc_best', s.get('mean_reward_best', 0.0))):.6f}",
                "adapter": str(s.get("adapter", "")),
            }
        )

    csv_path = out_dir / "stage52_baseline_compare.csv"
    write_csv(csv_path, rows)

    md_path = out_dir / "stage52_baseline_compare.md"
    lines = [
        "# Stage 5.2 Baseline Compare",
        "",
        f"- Generated at: `{dt.datetime.now().isoformat(timespec='seconds')}`",
        "- Primary metric: `mean_reward` (weighted/partial-credit). Secondary: `pass@1`.",
        "",
        "| label | samples | N | reward_first | reward_best | weighted_first | weighted_best | pass@1 | pass@N | best_exact |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['num_samples']} | {r['num_candidates']} | {r['mean_reward_first']} | {r['mean_reward_best']} | "
            f"{r['mean_weighted_acc_first']} | {r['mean_weighted_acc_best']} | {r['pass_at_1']} | "
            f"{r['pass_at_n']} | {r['best_of_n_exact_rate']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("[done] stage5.2 baseline compare")
    print(f"- out_dir: {out_dir}")
    print(f"- csv: {csv_path}")
    print(f"- md: {md_path}")


if __name__ == "__main__":
    main()

````

### File: scripts/verifier/extraction_verifier.py

````py
#!/usr/bin/env python3
import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common_verifier import die, load_json, load_jsonl, write_csv


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    # Best-effort: locate first balanced {...} block.
    start = -1
    depth = 0
    for i, ch in enumerate(text):
        if ch == "{":
            if start < 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    frag = text[start : i + 1]
                    try:
                        obj = json.loads(frag)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        return None
    return None


def canonical_str(v: Any) -> str:
    if isinstance(v, str):
        return v.strip().lower()
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    return str(v).strip().lower()


def type_ok(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    return True


def evaluate_one(
    *,
    pred_text: str,
    gold_obj: Dict[str, Any],
    schema_fields: Dict[str, str],
    required_fields: List[str],
    reward_mode: str,
    field_weights: Dict[str, float],
) -> Dict[str, Any]:
    pred_json = extract_first_json_object(pred_text)
    parse_ok = pred_json is not None
    if pred_json is None:
        pred_json = {}

    missing_required = 0
    type_mismatch = 0
    exact_match = 0
    field_total = 0

    for f in required_fields:
        if f not in pred_json:
            missing_required += 1

    weighted_num = 0.0
    weighted_den = 0.0
    per_field: Dict[str, float] = {}

    # Score only fields present in gold to support partial target rows.
    eval_fields = [f for f in gold_obj.keys() if f in schema_fields]
    for f in eval_fields:
        expected_t = schema_fields[f]
        field_total += 1
        wt = field_weights.get(f, 1.0)
        weighted_den += wt
        pv = pred_json.get(f, None)
        gv = gold_obj.get(f, None)
        ok_t = type_ok(pv, expected_t) if f in pred_json else False
        if not ok_t and f in pred_json:
            type_mismatch += 1
        m = 1.0 if (f in pred_json and ok_t and canonical_str(pv) == canonical_str(gv)) else 0.0
        if m > 0.5:
            exact_match += 1
        weighted_num += wt * m
        per_field[f] = m

    schema_ok = (missing_required == 0 and type_mismatch == 0)
    field_acc = (exact_match / field_total) if field_total > 0 else 0.0
    weighted_acc = (weighted_num / weighted_den) if weighted_den > 0 else 0.0

    if reward_mode == "binary":
        reward = 1.0 if (parse_ok and schema_ok and exact_match == field_total) else 0.0
    elif reward_mode == "weighted":
        reward = weighted_acc
    elif reward_mode == "decomposed":
        # Penalize parse/schema failure, then blend exact-match score.
        reward = (0.2 if parse_ok else 0.0) + (0.2 if schema_ok else 0.0) + (0.6 * field_acc)
    else:
        die(f"unknown reward mode: {reward_mode}")
        raise AssertionError("unreachable")

    return {
        "parse_ok": 1 if parse_ok else 0,
        "schema_ok": 1 if schema_ok else 0,
        "missing_required": missing_required,
        "type_mismatch": type_mismatch,
        "field_total": field_total,
        "field_exact_match": exact_match,
        "field_acc": field_acc,
        "weighted_acc": weighted_acc,
        "reward": reward,
        "per_field": per_field,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Extraction verifier (JSON/schema/field-match reward).")
    ap.add_argument("--pred-jsonl", type=str, required=True, help="jsonl rows: {id, output}")
    ap.add_argument("--gold-jsonl", type=str, required=True, help="jsonl rows: {id, gold}")
    ap.add_argument("--schema-json", type=str, required=True, help='{"required":[...],"fields":{"name":"string",...}}')
    ap.add_argument("--reward-mode", type=str, default="weighted", choices=["binary", "weighted", "decomposed"])
    ap.add_argument("--weights-json", type=str, default="", help='optional {"field":weight,...}')
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    pred_path = Path(args.pred_jsonl).expanduser().resolve()
    gold_path = Path(args.gold_jsonl).expanduser().resolve()
    schema_path = Path(args.schema_json).expanduser().resolve()
    for p in [pred_path, gold_path, schema_path]:
        if not p.exists():
            die(f"missing file: {p}")

    preds = load_jsonl(pred_path)
    golds = load_jsonl(gold_path)
    schema = load_json(schema_path)
    if not isinstance(schema, dict):
        die("--schema-json must be a dict")
    fields = schema.get("fields", {})
    required = schema.get("required", [])
    if not isinstance(fields, dict):
        die("schema.fields must be an object")
    if not isinstance(required, list):
        die("schema.required must be a list")
    schema_fields: Dict[str, str] = {str(k): str(v) for k, v in fields.items()}
    required_fields = [str(x) for x in required]

    weights: Dict[str, float] = {k: 1.0 for k in schema_fields.keys()}
    schema_weights = schema.get("field_weights", {})
    if schema_weights is not None:
        if not isinstance(schema_weights, dict):
            die("schema.field_weights must be an object")
        for k, v in schema_weights.items():
            ks = str(k)
            if ks not in schema_fields:
                die(f"schema.field_weights has unknown field: {ks}")
            try:
                wv = float(v)
            except Exception:
                die(f"schema.field_weights[{ks}] must be number")
            if wv < 0:
                die(f"schema.field_weights[{ks}] must be >= 0")
            weights[ks] = wv
    if args.weights_json.strip():
        w = load_json(Path(args.weights_json).expanduser().resolve())
        if not isinstance(w, dict):
            die("--weights-json must be an object")
        for k, v in w.items():
            ks = str(k)
            if ks not in schema_fields:
                die(f"--weights-json has unknown field: {ks}")
            weights[ks] = float(v)

    pred_by_id: Dict[str, Dict[str, Any]] = {}
    for r in preds:
        rid = str(r.get("id", ""))
        if not rid:
            die("pred row missing id")
        pred_by_id[rid] = r
    gold_by_id: Dict[str, Dict[str, Any]] = {}
    for r in golds:
        rid = str(r.get("id", ""))
        if not rid:
            die("gold row missing id")
        if "gold" not in r or not isinstance(r["gold"], dict):
            die(f"gold row id={rid} missing dict field 'gold'")
        gold_by_id[rid] = r

    ids = sorted(set(pred_by_id.keys()) & set(gold_by_id.keys()))
    if not ids:
        die("no overlapping ids between pred and gold")

    per_sample_rows: List[Dict[str, str]] = []
    rewards: List[float] = []
    parse_ok_cnt = 0
    schema_ok_cnt = 0
    field_acc_sum = 0.0
    weighted_acc_sum = 0.0

    for rid in ids:
        output = str(pred_by_id[rid].get("output", ""))
        gold_obj = gold_by_id[rid]["gold"]
        ev = evaluate_one(
            pred_text=output,
            gold_obj=gold_obj,
            schema_fields=schema_fields,
            required_fields=required_fields,
            reward_mode=args.reward_mode,
            field_weights=weights,
        )
        rewards.append(float(ev["reward"]))
        parse_ok_cnt += int(ev["parse_ok"])
        schema_ok_cnt += int(ev["schema_ok"])
        field_acc_sum += float(ev["field_acc"])
        weighted_acc_sum += float(ev["weighted_acc"])
        per_sample_rows.append(
            {
                "id": rid,
                "parse_ok": str(ev["parse_ok"]),
                "schema_ok": str(ev["schema_ok"]),
                "missing_required": str(ev["missing_required"]),
                "type_mismatch": str(ev["type_mismatch"]),
                "field_exact_match": str(ev["field_exact_match"]),
                "field_total": str(ev["field_total"]),
                "field_acc": f"{float(ev['field_acc']):.6f}",
                "weighted_acc": f"{float(ev['weighted_acc']):.6f}",
                "reward": f"{float(ev['reward']):.6f}",
            }
        )

    n = len(ids)
    summary = {
        "num_samples": n,
        "parse_ok_rate": parse_ok_cnt / n,
        "schema_ok_rate": schema_ok_cnt / n,
        "mean_field_acc": field_acc_sum / n,
        "mean_weighted_acc": weighted_acc_sum / n,
        "mean_reward": sum(rewards) / n,
    }

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (Path.cwd() / "reports" / f"stage51_extraction_verifier_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sample_csv = out_dir / "stage51_per_sample.csv"
    summary_md = out_dir / "stage51_summary.md"
    summary_json = out_dir / "stage51_summary.json"
    write_csv(per_sample_csv, per_sample_rows)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Stage 5.1 Extraction Verifier",
        "",
        f"- Generated at: `{dt.datetime.now().isoformat(timespec='seconds')}`",
        f"- Reward mode: `{args.reward_mode}`",
        f"- Samples: `{n}`",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| parse_ok_rate | {summary['parse_ok_rate']:.6f} |",
        f"| schema_ok_rate | {summary['schema_ok_rate']:.6f} |",
        f"| mean_field_acc | {summary['mean_field_acc']:.6f} |",
        f"| mean_weighted_acc | {summary['mean_weighted_acc']:.6f} |",
        f"| mean_reward | {summary['mean_reward']:.6f} |",
    ]
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("[done] stage5.1 extraction verifier")
    print(f"- per-sample csv: {per_sample_csv}")
    print(f"- summary md: {summary_md}")
    print(f"- summary json: {summary_json}")


if __name__ == "__main__":
    main()

````

---

## 3) 报告上下文（完整）

### Report: reports/stage52_best_of_n_4b_base_20260226_external_zip22_n4_100_forcejson/stage52_summary.md

````md
# Stage 5.2 Best-of-N Extraction Baseline

- Generated at: `2026-02-26T02:24:23`
- Reward mode: `weighted`
- Samples: `100`, N=`4`

| metric | value |
| --- | --- |
| pass@1 | 0.020000 |
| pass@N | 0.020000 |
| best_of_n_exact_rate | 0.020000 |
| mean_reward_first | 0.181452 |
| mean_reward_best | 0.199667 |
| mean_weighted_acc_first | 0.181452 |
| mean_weighted_acc_best | 0.199667 |

````

### Report: reports/stage52_best_of_n_4b_sftqlora_20260226_external_zip22_v1_n4_100_forcejson/stage52_summary.md

````md
# Stage 5.2 Best-of-N Extraction Baseline

- Generated at: `2026-02-26T09:23:15`
- Reward mode: `weighted`
- Samples: `100`, N=`4`

| metric | value |
| --- | --- |
| pass@1 | 0.010000 |
| pass@N | 0.030000 |
| best_of_n_exact_rate | 0.030000 |
| mean_reward_first | 0.326786 |
| mean_reward_best | 0.468024 |
| mean_weighted_acc_first | 0.326786 |
| mean_weighted_acc_best | 0.468024 |

````

### Report: reports/stage52_best_of_n_4b_dpo_20260226_external_zip22_v4_hardpair_n4_100_forcejson_sample_gpu1/stage52_summary.md

````md
# Stage 5.2 Best-of-N Extraction Baseline

- Generated at: `2026-02-26T13:29:55`
- Reward mode: `weighted`
- Samples: `100`, N=`4`

| metric | value |
| --- | --- |
| pass@1 | 0.020000 |
| pass@N | 0.020000 |
| best_of_n_exact_rate | 0.020000 |
| mean_reward_first | 0.261024 |
| mean_reward_best | 0.280667 |
| mean_weighted_acc_first | 0.261024 |
| mean_weighted_acc_best | 0.280667 |

````

### Report: reports/stage52_baseline_compare_4b_20260226_external_zip22_n4_all_dpo_variants_v2/stage52_baseline_compare.md

````md
# Stage 5.2 Baseline Compare

- Generated at: `2026-02-26T13:30:14`
- Primary metric: `mean_reward` (weighted/partial-credit). Secondary: `pass@1`.

| label | samples | N | reward_first | reward_best | weighted_first | weighted_best | pass@1 | pass@N | best_exact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_n4 | 100 | 4 | 0.181452 | 0.199667 | 0.181452 | 0.199667 | 0.020000 | 0.020000 | 0.020000 |
| sft_n4 | 100 | 4 | 0.326786 | 0.468024 | 0.326786 | 0.468024 | 0.010000 | 0.030000 | 0.030000 |
| dpo_v1_n4 | 100 | 4 | 0.181452 | 0.199667 | 0.181452 | 0.199667 | 0.020000 | 0.020000 | 0.020000 |
| dpo_v2_refcpu_n4 | 100 | 4 | 0.181452 | 0.199667 | 0.181452 | 0.199667 | 0.020000 | 0.020000 | 0.020000 |
| dpo_v3_tuned_n4 | 100 | 4 | 0.204381 | 0.221524 | 0.204381 | 0.221524 | 0.010000 | 0.020000 | 0.020000 |
| dpo_v4_hardpair_n4_gpu1 | 100 | 4 | 0.261024 | 0.280667 | 0.261024 | 0.280667 | 0.020000 | 0.020000 | 0.020000 |

````

---

## 4) 风格参考

附件上传：`tutorial_01_life_of_a_token.md`

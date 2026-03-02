# Tutorial #14 Prompt (Full, With Code + Reports)

> 生成日期：2026-02-26
> 用途：整份复制到新 chat 作为写作输入
> 标注：大文件（如 cuda_runtime.cpp）按“该篇相关段落”摘取，避免上下文爆炸

---

## 0) 系统 Prompt（先贴这段）

```text
我正在为我的开源项目 Ember 写一个系列教程。请帮我写第 14 篇。

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
请写第 14 篇：Verifier 与 Reward 设计 — 怎么给模型打分。

## 本篇必须引用的代码与报告都在本消息里（不要再让我补充路径）

- 数学只要求 precision/recall/F1 级别（但推导尽量完整）
- 讲清 binary / weighted / field-level reward 的差异与场景
```

---

## 2) 代码上下文（完整/相关段落）

### File: scripts/verifier/README.md

````md
# Verifier Scripts

Reward/evaluation helpers used by stage 5.x training loops.

## Scripts

- `extraction_verifier.py`: JSON extraction verifier
  - parse validity
  - schema checks
  - field-level matching
  - reward modes: `binary`, `weighted`, `decomposed`
- `sql_verifier.py`: SQL verifier over SQLite
  - execute predicted SQL safely (SELECT/WITH only)
  - compare result sets against gold query outputs

## Intended Use

- Run standalone for reward diagnostics.
- Integrate into Best-of-N / DPO / GRPO training loops as reward backends.


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

### File: scripts/verifier/sql_verifier.py

````py
#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

from common_verifier import die, load_jsonl, write_csv


def normalize_rows(rows: List[Tuple[Any, ...]]) -> List[Tuple[str, ...]]:
    norm: List[Tuple[str, ...]] = []
    for row in rows:
        norm.append(tuple("null" if v is None else str(v) for v in row))
    norm.sort()
    return norm


def is_select_like(sql: str) -> bool:
    s = sql.strip().lower()
    return s.startswith("select") or s.startswith("with")


def run_query(db_path: Path, sql: str) -> Tuple[bool, List[Tuple[Any, ...]], str]:
    if not is_select_like(sql):
        return False, [], "non_select_query_blocked"
    try:
        con = sqlite3.connect(str(db_path))
        cur = con.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        con.close()
        return True, rows, ""
    except Exception as e:
        return False, [], str(e)


def main() -> None:
    ap = argparse.ArgumentParser(description="SQL verifier (SQLite result-set comparison).")
    ap.add_argument("--pred-jsonl", type=str, required=True, help='jsonl rows: {"id","sql"}')
    ap.add_argument("--gold-jsonl", type=str, required=True, help='jsonl rows: {"id","db_path","gold_sql"}')
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    pred_path = Path(args.pred_jsonl).expanduser().resolve()
    gold_path = Path(args.gold_jsonl).expanduser().resolve()
    for p in [pred_path, gold_path]:
        if not p.exists():
            die(f"missing file: {p}")

    preds = load_jsonl(pred_path)
    golds = load_jsonl(gold_path)
    pred_by_id = {str(r.get("id", "")): r for r in preds}
    gold_by_id = {str(r.get("id", "")): r for r in golds}
    ids = sorted(set(pred_by_id.keys()) & set(gold_by_id.keys()))
    if not ids:
        die("no overlapping ids")

    per_rows: List[Dict[str, str]] = []
    exec_ok_cnt = 0
    match_cnt = 0
    rewards: List[float] = []

    for rid in ids:
        pr = pred_by_id[rid]
        gr = gold_by_id[rid]
        if "sql" not in pr:
            die(f"pred id={rid} missing sql")
        if "db_path" not in gr or "gold_sql" not in gr:
            die(f"gold id={rid} missing db_path/gold_sql")

        db_path = Path(str(gr["db_path"])).expanduser().resolve()
        if not db_path.exists():
            die(f"gold id={rid} db not found: {db_path}")

        pred_sql = str(pr["sql"])
        gold_sql = str(gr["gold_sql"])
        ok_p, rows_p, err_p = run_query(db_path, pred_sql)
        ok_g, rows_g, err_g = run_query(db_path, gold_sql)
        if not ok_g:
            die(f"gold sql failed id={rid}: {err_g}")

        exec_ok = 1 if ok_p else 0
        match = 0
        if ok_p:
            match = 1 if normalize_rows(rows_p) == normalize_rows(rows_g) else 0

        reward = float(match)
        exec_ok_cnt += exec_ok
        match_cnt += match
        rewards.append(reward)

        per_rows.append(
            {
                "id": rid,
                "exec_ok": str(exec_ok),
                "result_match": str(match),
                "pred_row_count": str(len(rows_p) if ok_p else 0),
                "gold_row_count": str(len(rows_g)),
                "reward": f"{reward:.6f}",
                "error": err_p,
            }
        )

    n = len(ids)
    summary = {
        "num_samples": n,
        "exec_ok_rate": exec_ok_cnt / n,
        "result_match_rate": match_cnt / n,
        "mean_reward": sum(rewards) / n,
    }

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (Path.cwd() / "reports" / f"stage51_sql_verifier_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    per_csv = out_dir / "stage51_sql_per_sample.csv"
    summary_md = out_dir / "stage51_sql_summary.md"
    summary_json = out_dir / "stage51_sql_summary.json"
    write_csv(per_csv, per_rows)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Stage 5.1 SQL Verifier",
        "",
        f"- Generated at: `{dt.datetime.now().isoformat(timespec='seconds')}`",
        f"- Samples: `{n}`",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| exec_ok_rate | {summary['exec_ok_rate']:.6f} |",
        f"| result_match_rate | {summary['result_match_rate']:.6f} |",
        f"| mean_reward | {summary['mean_reward']:.6f} |",
    ]
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("[done] stage5.1 sql verifier")
    print(f"- per-sample csv: {per_csv}")
    print(f"- summary md: {summary_md}")
    print(f"- summary json: {summary_json}")


if __name__ == "__main__":
    main()

````

---

## 3) 报告上下文（完整）

### Report: reports/stage51_extraction_verifier_smoke_20260225/out/stage51_summary.md

````md
# Stage 5.1 Extraction Verifier

- Generated at: `2026-02-25T21:10:23`
- Reward mode: `weighted`
- Samples: `3`

| metric | value |
| --- | --- |
| parse_ok_rate | 0.666667 |
| schema_ok_rate | 0.333333 |
| mean_field_acc | 0.555556 |
| mean_weighted_acc | 0.555556 |
| mean_reward | 0.555556 |

````

### Report: reports/stage51_sql_verifier_smoke_20260225/out/stage51_sql_summary.md

````md
# Stage 5.1 SQL Verifier

- Generated at: `2026-02-25T21:11:17`
- Samples: `3`

| metric | value |
| --- | --- |
| exec_ok_rate | 0.666667 |
| result_match_rate | 0.666667 |
| mean_reward | 0.666667 |

````

### Report: reports/stage51_extraction_verifier_smoke_20260225/out/stage51_per_sample.csv

````csv
id,parse_ok,schema_ok,missing_required,type_mismatch,field_exact_match,field_total,field_acc,weighted_acc,reward
a1,1,1,0,0,3,3,1.000000,1.000000,1.000000
a2,1,0,0,1,2,3,0.666667,0.666667,0.666667
a3,0,0,3,0,0,3,0.000000,0.000000,0.000000

````

---

## 4) 风格参考

附件上传：`tutorial_01_life_of_a_token.md`

#!/usr/bin/env python3
import argparse
import datetime as dt
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common_train import die, load_json, load_jsonl, write_csv


VALID_TYPES = {"string", "number", "integer", "boolean", "object", "array"}

RULE_KEYWORDS = [
    "规则",
    "rule",
    "仅在",
    "only",
    "choose",
    "选择",
    "highest",
    "lowest",
    "max",
    "min",
    "filter",
    "排序",
    "排名",
]

TIEBREAK_KEYWORDS = [
    "并列",
    "tie-break",
    "tie break",
    "若并列",
    "earliest",
    "latest",
    "最早",
    "最晚",
]

ANSWER_LEAK_KEYWORDS = [
    "gold",
    "ground truth",
    "reference answer",
    "expected output",
    "label:",
    "答案：",
    "标准答案",
    "参考答案",
]


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
    return False


def pct(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    w = pos - lo
    return float(xs[lo] * (1.0 - w) + xs[hi] * w)


def norm_text(s: str) -> str:
    return " ".join(s.strip().lower().split())


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def compact_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def count_dups(items: List[str]) -> int:
    seen = set()
    dups = 0
    for x in items:
        if x in seen:
            dups += 1
        else:
            seen.add(x)
    return dups


def contains_any(text_lower: str, words: List[str]) -> bool:
    for w in words:
        if w in text_lower:
            return True
    return False


def parse_dataset_specs(args: argparse.Namespace) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if args.dataset_jsonl:
        out["dataset"] = Path(args.dataset_jsonl).expanduser().resolve()
    for label, raw in [
        ("train", args.train_jsonl),
        ("val", args.val_jsonl),
        ("test", args.test_jsonl),
    ]:
        if raw:
            out[label] = Path(raw).expanduser().resolve()
    if not out:
        die("provide at least one of --dataset-jsonl / --train-jsonl / --val-jsonl / --test-jsonl")
    for k, p in out.items():
        if not p.exists():
            die(f"{k} jsonl not found: {p}")
    return out


def validate_schema(schema: Dict[str, Any]) -> Tuple[List[str], Dict[str, str]]:
    required = schema.get("required", [])
    fields = schema.get("fields", {})
    weights = schema.get("field_weights", {})
    if not isinstance(required, list) or not isinstance(fields, dict):
        die("schema invalid: need {required: [...], fields: {...}}")
    if not isinstance(weights, dict):
        die("schema invalid: field_weights must be object if provided")
    req = [str(x) for x in required]
    fmap: Dict[str, str] = {}
    for k, v in fields.items():
        ks = str(k)
        vs = str(v)
        if vs not in VALID_TYPES:
            die(f"schema invalid type: fields[{ks}]={vs}")
        fmap[ks] = vs
    for r in req:
        if r not in fmap:
            die(f"schema invalid: required field not in fields: {r}")
    for k, v in weights.items():
        ks = str(k)
        if ks not in fmap:
            die(f"schema invalid: field_weights has unknown field: {ks}")
        try:
            wv = float(v)
        except Exception:
            die(f"schema invalid: field_weights[{ks}] must be number")
        if wv < 0:
            die(f"schema invalid: field_weights[{ks}] must be >= 0")
    return req, fmap


def analyze_split(
    *,
    label: str,
    rows: List[Dict[str, Any]],
    required_fields: List[str],
    schema_fields: Dict[str, str],
    max_suspicious_rows: int,
) -> Tuple[Dict[str, str], List[Dict[str, Any]], Dict[str, set]]:
    n = len(rows)
    ids: List[str] = []
    prompt_hashes: List[str] = []
    row_hashes: List[str] = []
    gold_hashes: List[str] = []

    prompt_chars: List[float] = []
    prompt_tokens: List[float] = []

    missing_key_rows = 0
    non_obj_gold_rows = 0
    missing_required_rows = 0
    type_mismatch_rows = 0
    extra_field_rows = 0
    rule_keyword_rows = 0
    tie_break_rows = 0
    multi_record_rows = 0
    answer_leak_keyword_rows = 0
    exact_gold_json_in_prompt_rows = 0
    null_target_rows = 0

    suspicious_rows: List[Dict[str, Any]] = []

    for i, r in enumerate(rows):
        rid = str(r.get("id", ""))
        prompt = str(r.get("prompt", ""))
        gold = r.get("gold", None)

        if not rid or not prompt or "gold" not in r:
            missing_key_rows += 1
            if len(suspicious_rows) < max_suspicious_rows:
                suspicious_rows.append(
                    {
                        "split": label,
                        "row_index": i,
                        "id": rid,
                        "reason": "missing id/prompt/gold",
                    }
                )
            continue
        if not isinstance(gold, dict):
            non_obj_gold_rows += 1
            if len(suspicious_rows) < max_suspicious_rows:
                suspicious_rows.append(
                    {
                        "split": label,
                        "row_index": i,
                        "id": rid,
                        "reason": "gold is not object",
                    }
                )
            continue

        ids.append(rid)
        pnorm = norm_text(prompt)
        prompt_hashes.append(sha1_text(pnorm))
        gold_compact = compact_json(gold)
        row_hashes.append(sha1_text(pnorm + "\n" + gold_compact))
        gold_hashes.append(sha1_text(gold_compact))
        prompt_chars.append(float(len(prompt)))
        prompt_tokens.append(float(len(prompt.split())))

        plower = prompt.lower()
        reasons: List[str] = []

        missing_required = [f for f in required_fields if f not in gold]
        if missing_required:
            missing_required_rows += 1
            reasons.append(f"missing_required={','.join(missing_required)}")

        type_bad: List[str] = []
        for f, expected in schema_fields.items():
            if f in gold and (not type_ok(gold[f], expected)):
                type_bad.append(f)
        if type_bad:
            type_mismatch_rows += 1
            reasons.append(f"type_mismatch={','.join(type_bad)}")

        extra_fields = [k for k in gold.keys() if str(k) not in schema_fields]
        if extra_fields:
            extra_field_rows += 1
            reasons.append(f"extra_fields={','.join(extra_fields)}")

        if any(gold.get(f, None) is None for f in required_fields):
            null_target_rows += 1

        if contains_any(plower, RULE_KEYWORDS):
            rule_keyword_rows += 1
        if contains_any(plower, TIEBREAK_KEYWORDS):
            tie_break_rows += 1

        bullet_lines = [ln for ln in prompt.splitlines() if ln.strip().startswith("- ")]
        if len(bullet_lines) >= 2 or prompt.count("employee_id=") >= 2:
            multi_record_rows += 1

        if contains_any(plower, ANSWER_LEAK_KEYWORDS):
            answer_leak_keyword_rows += 1
            reasons.append("answer_leak_keyword")
        if gold_compact in prompt:
            exact_gold_json_in_prompt_rows += 1
            reasons.append("exact_gold_json_in_prompt")

        if reasons and len(suspicious_rows) < max_suspicious_rows:
            suspicious_rows.append(
                {
                    "split": label,
                    "row_index": i,
                    "id": rid,
                    "reason": ";".join(reasons),
                }
            )

    if n == 0:
        n = 1
    row = {
        "split": label,
        "num_rows": str(len(rows)),
        "missing_key_rows": str(missing_key_rows),
        "non_obj_gold_rows": str(non_obj_gold_rows),
        "missing_required_rows": str(missing_required_rows),
        "type_mismatch_rows": str(type_mismatch_rows),
        "extra_field_rows": str(extra_field_rows),
        "duplicate_id_rows": str(count_dups(ids)),
        "duplicate_prompt_rows": str(count_dups(prompt_hashes)),
        "duplicate_row_rows": str(count_dups(row_hashes)),
        "prompt_chars_p50": f"{pct(prompt_chars, 0.5):.1f}",
        "prompt_chars_p95": f"{pct(prompt_chars, 0.95):.1f}",
        "prompt_tokens_est_p50": f"{pct(prompt_tokens, 0.5):.1f}",
        "prompt_tokens_est_p95": f"{pct(prompt_tokens, 0.95):.1f}",
        "rule_keyword_ratio": f"{rule_keyword_rows / max(1, len(rows)):.6f}",
        "tie_break_ratio": f"{tie_break_rows / max(1, len(rows)):.6f}",
        "multi_record_ratio": f"{multi_record_rows / max(1, len(rows)):.6f}",
        "answer_leak_keyword_ratio": f"{answer_leak_keyword_rows / max(1, len(rows)):.6f}",
        "exact_gold_json_in_prompt_ratio": f"{exact_gold_json_in_prompt_rows / max(1, len(rows)):.6f}",
        "null_target_ratio": f"{null_target_rows / max(1, len(rows)):.6f}",
    }

    index_sets = {
        "id_set": set(ids),
        "prompt_set": set(prompt_hashes),
        "row_set": set(row_hashes),
        "gold_set": set(gold_hashes),
    }
    return row, suspicious_rows, index_sets


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate stage5.2 extraction dataset quality and leakage risk.")
    ap.add_argument("--schema-json", type=str, required=True)
    ap.add_argument("--dataset-jsonl", type=str, default="")
    ap.add_argument("--train-jsonl", type=str, default="")
    ap.add_argument("--val-jsonl", type=str, default="")
    ap.add_argument("--test-jsonl", type=str, default="")
    ap.add_argument("--max-suspicious-rows", type=int, default=200)
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    if args.max_suspicious_rows <= 0:
        die("--max-suspicious-rows must be > 0")

    schema_path = Path(args.schema_json).expanduser().resolve()
    if not schema_path.exists():
        die(f"schema not found: {schema_path}")
    schema = load_json(schema_path)
    if not isinstance(schema, dict):
        die("schema json must be object")
    required_fields, schema_fields = validate_schema(schema)
    datasets = parse_dataset_specs(args)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (
        Path.cwd() / "reports" / f"stage52_dataset_validation_{ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    per_split_rows: List[Dict[str, str]] = []
    suspicious_all: List[Dict[str, Any]] = []
    index_map: Dict[str, Dict[str, set]] = {}

    for label, path in datasets.items():
        rows = load_jsonl(path)
        split_row, suspicious_rows, idx = analyze_split(
            label=label,
            rows=rows,
            required_fields=required_fields,
            schema_fields=schema_fields,
            max_suspicious_rows=args.max_suspicious_rows,
        )
        split_row["dataset_jsonl"] = str(path)
        per_split_rows.append(split_row)
        index_map[label] = idx
        suspicious_all.extend(suspicious_rows)

    cross_rows: List[Dict[str, str]] = []
    labels = list(datasets.keys())
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a = labels[i]
            b = labels[j]
            xa = index_map[a]
            xb = index_map[b]
            cross_rows.append(
                {
                    "split_a": a,
                    "split_b": b,
                    "id_overlap": str(len(xa["id_set"] & xb["id_set"])),
                    "prompt_overlap": str(len(xa["prompt_set"] & xb["prompt_set"])),
                    "row_overlap": str(len(xa["row_set"] & xb["row_set"])),
                    "gold_overlap": str(len(xa["gold_set"] & xb["gold_set"])),
                }
            )

    per_split_csv = out_dir / "stage52_dataset_validation_per_split.csv"
    write_csv(per_split_csv, per_split_rows)
    cross_csv = out_dir / "stage52_dataset_validation_cross_split.csv"
    if cross_rows:
        write_csv(cross_csv, cross_rows)

    suspicious_jsonl = out_dir / "stage52_dataset_suspicious_rows.jsonl"
    with suspicious_jsonl.open("w", encoding="utf-8") as f:
        for r in suspicious_all[: args.max_suspicious_rows]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary_json = out_dir / "stage52_dataset_validation_summary.json"
    summary = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "schema_json": str(schema_path),
        "datasets": {k: str(v) for k, v in datasets.items()},
        "num_splits": len(datasets),
        "num_suspicious_rows_saved": min(len(suspicious_all), args.max_suspicious_rows),
        "per_split_csv": str(per_split_csv),
        "cross_split_csv": str(cross_csv) if cross_rows else "",
        "suspicious_jsonl": str(suspicious_jsonl),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary_md = out_dir / "stage52_dataset_validation_summary.md"
    lines = [
        "# Stage 5.2 Dataset Validation Summary",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Schema: `{schema_path}`",
        f"- Splits: `{', '.join(labels)}`",
        f"- Suspicious rows saved: `{summary['num_suspicious_rows_saved']}`",
        "",
        "## Per-split",
        "",
        "| split | rows | missing_keys | non_obj_gold | missing_required | type_mismatch | dup_id | dup_prompt | dup_row | p95_tokens_est | rule_ratio | tie_break_ratio | multi_record_ratio | leak_kw_ratio | exact_gold_json_ratio |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in per_split_rows:
        lines.append(
            f"| {r['split']} | {r['num_rows']} | {r['missing_key_rows']} | {r['non_obj_gold_rows']} | "
            f"{r['missing_required_rows']} | {r['type_mismatch_rows']} | {r['duplicate_id_rows']} | "
            f"{r['duplicate_prompt_rows']} | {r['duplicate_row_rows']} | {r['prompt_tokens_est_p95']} | "
            f"{r['rule_keyword_ratio']} | {r['tie_break_ratio']} | {r['multi_record_ratio']} | "
            f"{r['answer_leak_keyword_ratio']} | {r['exact_gold_json_in_prompt_ratio']} |"
        )
    if cross_rows:
        lines += [
            "",
            "## Cross-split overlap",
            "",
            "| split_a | split_b | id_overlap | prompt_overlap | row_overlap | gold_overlap |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for r in cross_rows:
            lines.append(
                f"| {r['split_a']} | {r['split_b']} | {r['id_overlap']} | {r['prompt_overlap']} | "
                f"{r['row_overlap']} | {r['gold_overlap']} |"
            )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("[done] stage5.2 dataset validation")
    print(f"- out_dir: {out_dir}")
    print(f"- per_split_csv: {per_split_csv}")
    if cross_rows:
        print(f"- cross_split_csv: {cross_csv}")
    print(f"- suspicious_jsonl: {suspicious_jsonl}")
    print(f"- summary_md: {summary_md}")


if __name__ == "__main__":
    main()

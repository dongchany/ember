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

    weights: Dict[str, float] = {}
    if args.weights_json.strip():
        w = load_json(Path(args.weights_json).expanduser().resolve())
        if not isinstance(w, dict):
            die("--weights-json must be an object")
        for k, v in w.items():
            weights[str(k)] = float(v)

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

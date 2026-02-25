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
    required_fields: List[str],
    reward_mode: str,
) -> Dict[str, Any]:
    pred_json = extract_first_json_object(pred_text)
    parse_ok = pred_json is not None
    if pred_json is None:
        pred_json = {}

    missing_required = sum(1 for f in required_fields if f not in pred_json)
    type_mismatch = 0
    exact_match = 0
    field_total = 0
    for f, expected_t in schema_fields.items():
        field_total += 1
        if f in pred_json:
            ok_t = type_ok(pred_json[f], expected_t)
            if not ok_t:
                type_mismatch += 1
            if ok_t and canonical_str(pred_json[f]) == canonical_str(gold_obj.get(f, None)):
                exact_match += 1

    schema_ok = (missing_required == 0 and type_mismatch == 0)
    field_acc = (exact_match / field_total) if field_total > 0 else 0.0
    exact_all = (field_total > 0 and exact_match == field_total and parse_ok and schema_ok)

    if reward_mode == "binary":
        reward = 1.0 if exact_all else 0.0
    elif reward_mode == "weighted":
        reward = field_acc
    elif reward_mode == "decomposed":
        reward = (0.2 if parse_ok else 0.0) + (0.2 if schema_ok else 0.0) + (0.6 * field_acc)
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
        "exact_all": 1 if exact_all else 0,
        "reward": reward,
    }


def build_prompt(tokenizer, prompt: str) -> str:
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
    if not isinstance(fields, dict) or not isinstance(required, list):
        die("schema invalid; need fields object and required list")
    schema_fields: Dict[str, str] = {str(k): str(v) for k, v in fields.items()}
    required_fields = [str(x) for x in required]

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

    for i, sample in enumerate(data):
        sid = str(sample["id"])
        prompt = str(sample["prompt"])
        gold = sample["gold"]
        full_prompt = build_prompt(tokenizer, prompt)
        enc = tokenizer(full_prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(args.device)
        attn = enc["attention_mask"].to(args.device)

        cand_scores: List[Tuple[int, Dict[str, Any], str]] = []
        for c in range(args.num_candidates):
            seed = args.seed + i * 1000 + c
            torch.manual_seed(seed)
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
                    "exact_all": str(ev["exact_all"]),
                    "parse_ok": str(ev["parse_ok"]),
                    "schema_ok": str(ev["schema_ok"]),
                    "missing_required": str(ev["missing_required"]),
                    "type_mismatch": str(ev["type_mismatch"]),
                    "output": text.replace("\n", "\\n"),
                }
            )

        first_ev = cand_scores[0][1]
        best = max(cand_scores, key=lambda x: (float(x[1]["reward"]), float(x[1]["field_acc"]), int(x[1]["exact_all"])))
        any_exact = any(int(x[1]["exact_all"]) == 1 for x in cand_scores)

        pass1 += int(first_ev["exact_all"])
        passn += 1 if any_exact else 0
        best_pass += int(best[1]["exact_all"])
        mean_reward_first += float(first_ev["reward"])
        mean_reward_best += float(best[1]["reward"])

        best_rows.append(
            {
                "id": sid,
                "best_candidate_id": str(best[0]),
                "best_reward": f"{float(best[1]['reward']):.6f}",
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
    ]
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("[done] stage5.2 best-of-n baseline")
    print(f"- candidates csv: {cand_csv}")
    print(f"- best csv: {best_csv}")
    print(f"- summary md: {summary_md}")
    print(f"- summary json: {summary_json}")


if __name__ == "__main__":
    main()

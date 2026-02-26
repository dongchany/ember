#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from common_train import die, load_jsonl


def parse_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def parse_int(v: str, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def load_csv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def choose_rejected(rows: List[Dict[str, str]]) -> Tuple[Dict[str, str], bool]:
    # Prefer the strongest non-exact candidate as hard negative.
    non_exact = [r for r in rows if parse_int(r.get("exact_all", "0")) == 0]
    if non_exact:
        ranked = sorted(
            non_exact,
            key=lambda r: (
                parse_float(r.get("reward", "0")),
                parse_float(r.get("field_acc", "0")),
                parse_int(r.get("parse_ok", "0")),
                parse_int(r.get("schema_ok", "0")),
            ),
            reverse=True,
        )
        return ranked[0], False
    # Fallback: if every candidate is exact, use the weakest exact as rejected.
    ranked = sorted(
        rows,
        key=lambda r: (
            parse_float(r.get("reward", "0")),
            parse_float(r.get("field_acc", "0")),
        ),
    )
    return ranked[0], True


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build DPO pairs with oracle chosen (gold JSON) and hard rejected from candidate outputs."
    )
    ap.add_argument("--dataset-jsonl", type=str, required=True, help='rows: {"id","prompt","gold"}')
    ap.add_argument("--candidates-csv", type=str, required=True, help="stage52_candidates.csv")
    ap.add_argument("--min-margin", type=float, default=0.05, help="minimum (1.0 - rejected_reward) margin")
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    ds_path = Path(args.dataset_jsonl).expanduser().resolve()
    cand_path = Path(args.candidates_csv).expanduser().resolve()
    for p in [ds_path, cand_path]:
        if not p.exists():
            die(f"missing file: {p}")

    ds = load_jsonl(ds_path)
    cands = load_csv(cand_path)

    prompt_by_id: Dict[str, str] = {}
    gold_by_id: Dict[str, Dict[str, Any]] = {}
    for r in ds:
        rid = str(r.get("id", ""))
        prompt = str(r.get("prompt", ""))
        gold = r.get("gold")
        if not rid or not prompt or not isinstance(gold, dict):
            continue
        prompt_by_id[rid] = prompt
        gold_by_id[rid] = gold

    by_id: Dict[str, List[Dict[str, str]]] = {}
    for r in cands:
        rid = str(r.get("id", ""))
        if not rid:
            continue
        by_id.setdefault(rid, []).append(r)

    pair_rows: List[Dict[str, Any]] = []
    skipped_missing_meta = 0
    skipped_no_candidates = 0
    skipped_low_margin = 0
    used_exact_fallback = 0
    margins: List[float] = []

    for rid, gold in gold_by_id.items():
        prompt = prompt_by_id.get(rid, "")
        rows = by_id.get(rid, [])
        if not prompt:
            skipped_missing_meta += 1
            continue
        if not rows:
            skipped_no_candidates += 1
            continue

        rejected, is_fallback = choose_rejected(rows)
        rejected_reward = parse_float(rejected.get("reward", "0.0"))
        chosen_reward = 1.0
        margin = chosen_reward - rejected_reward
        if margin < args.min_margin:
            skipped_low_margin += 1
            continue
        if is_fallback:
            used_exact_fallback += 1

        pair_rows.append(
            {
                "id": rid,
                "prompt": prompt,
                "chosen": json.dumps(gold, ensure_ascii=False),
                "rejected": str(rejected.get("output", "")).replace("\\n", "\n"),
                "reward_chosen": chosen_reward,
                "reward_rejected": rejected_reward,
                "reward_margin": margin,
                "chosen_exact_all": 1,
                "rejected_exact_all": parse_int(rejected.get("exact_all", "0")),
                "source": "oracle_exact_hard_negative",
            }
        )
        margins.append(margin)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (
        Path.cwd() / "reports" / f"stage52_dpo_pairs_oracle_exact_{ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "stage52_dpo_pairs.jsonl"
    out_md = out_dir / "stage52_dpo_pairs_summary.md"

    write_jsonl(out_jsonl, pair_rows)

    num_pairs = len(pair_rows)
    avg_margin = (sum(margins) / num_pairs) if num_pairs > 0 else 0.0
    min_margin = min(margins) if margins else 0.0
    max_margin = max(margins) if margins else 0.0

    lines = [
        "# Stage 5.2 DPO Pair Builder (Oracle Exact)",
        "",
        f"- Generated at: `{dt.datetime.now().isoformat(timespec='seconds')}`",
        f"- Candidate rows: `{len(cands)}`",
        f"- Unique sample ids in dataset: `{len(gold_by_id)}`",
        f"- Unique sample ids in candidates: `{len(by_id)}`",
        f"- Valid pairs: `{num_pairs}`",
        f"- Skipped (missing prompt/meta): `{skipped_missing_meta}`",
        f"- Skipped (no candidates): `{skipped_no_candidates}`",
        f"- Skipped (low margin): `{skipped_low_margin}`",
        f"- Rejected exact fallback used: `{used_exact_fallback}`",
        f"- Margin threshold: `{args.min_margin:.6f}`",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| avg_reward_margin | {avg_margin:.6f} |",
        f"| min_reward_margin | {min_margin:.6f} |",
        f"| max_reward_margin | {max_margin:.6f} |",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("[done] stage5.2 oracle-exact dpo pair builder")
    print(f"- pairs jsonl: {out_jsonl}")
    print(f"- summary md: {out_md}")


if __name__ == "__main__":
    main()

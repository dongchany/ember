#!/usr/bin/env python3
import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List

from common_train import die, load_jsonl


def load_csv(path: Path) -> List[Dict[str, str]]:
    import csv

    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def main() -> None:
    ap = argparse.ArgumentParser(description="Build DPO chosen/rejected pairs from Best-of-N candidates.")
    ap.add_argument("--dataset-jsonl", type=str, required=True, help='rows: {"id","prompt",...}')
    ap.add_argument("--candidates-csv", type=str, required=True, help="stage52_candidates.csv")
    ap.add_argument("--min-margin", type=float, default=0.05, help="min reward margin for valid pair")
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
    for r in ds:
        rid = str(r.get("id", ""))
        if not rid:
            continue
        prompt_by_id[rid] = str(r.get("prompt", ""))

    by_id: Dict[str, List[Dict[str, str]]] = {}
    for r in cands:
        rid = str(r.get("id", ""))
        if not rid:
            continue
        by_id.setdefault(rid, []).append(r)

    pair_rows: List[Dict[str, Any]] = []
    skipped_low_margin = 0
    skipped_missing_prompt = 0
    margins: List[float] = []

    for rid, rows in by_id.items():
        if rid not in prompt_by_id:
            skipped_missing_prompt += 1
            continue
        if len(rows) < 2:
            continue
        rows_sorted = sorted(
            rows,
            key=lambda r: (
                parse_float(r.get("reward", "0")),
                parse_float(r.get("field_acc", "0")),
                int(r.get("exact_all", "0") or "0"),
            ),
            reverse=True,
        )
        best = rows_sorted[0]
        worst = rows_sorted[-1]
        rb = parse_float(best.get("reward", "0"))
        rw = parse_float(worst.get("reward", "0"))
        margin = rb - rw
        if margin < args.min_margin:
            skipped_low_margin += 1
            continue
        margins.append(margin)
        pair_rows.append(
            {
                "id": rid,
                "prompt": prompt_by_id[rid],
                "chosen": str(best.get("output", "")).replace("\\n", "\n"),
                "rejected": str(worst.get("output", "")).replace("\\n", "\n"),
                "reward_chosen": rb,
                "reward_rejected": rw,
                "reward_margin": margin,
                "chosen_exact_all": int(best.get("exact_all", "0") or "0"),
                "rejected_exact_all": int(worst.get("exact_all", "0") or "0"),
            }
        )

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (Path.cwd() / "reports" / f"stage52_dpo_pairs_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "stage52_dpo_pairs.jsonl"
    out_md = out_dir / "stage52_dpo_pairs_summary.md"
    write_jsonl(out_jsonl, pair_rows)

    num_pairs = len(pair_rows)
    avg_margin = (sum(margins) / num_pairs) if num_pairs > 0 else 0.0
    min_margin = min(margins) if margins else 0.0
    max_margin = max(margins) if margins else 0.0

    lines = [
        "# Stage 5.2 DPO Pair Builder",
        "",
        f"- Generated at: `{dt.datetime.now().isoformat(timespec='seconds')}`",
        f"- Candidate rows: `{len(cands)}`",
        f"- Unique sample ids: `{len(by_id)}`",
        f"- Valid pairs: `{num_pairs}`",
        f"- Skipped (low margin): `{skipped_low_margin}`",
        f"- Skipped (missing prompt): `{skipped_missing_prompt}`",
        f"- Margin threshold: `{args.min_margin:.6f}`",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| avg_reward_margin | {avg_margin:.6f} |",
        f"| min_reward_margin | {min_margin:.6f} |",
        f"| max_reward_margin | {max_margin:.6f} |",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("[done] stage5.2 dpo pair builder")
    print(f"- pairs jsonl: {out_jsonl}")
    print(f"- summary md: {out_md}")


if __name__ == "__main__":
    main()

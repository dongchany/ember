#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from common_train import die, load_jsonl


def choose_other(value: Any, pool: List[Any], rng: random.Random) -> Any:
    cands = [x for x in pool if x != value]
    if not cands:
        return value
    return rng.choice(cands)


def corrupt_gold(gold: Dict[str, Any], rng: random.Random) -> str:
    mode = rng.choice(
        [
            "flip_bool",
            "age_offset",
            "city_swap",
            "drop_field",
            "stringify_number",
            "wrong_date",
            "non_json",
        ]
    )
    g = dict(gold)

    if mode == "flip_bool" and "active" in g and isinstance(g["active"], bool):
        g["active"] = (not g["active"])
        return json.dumps(g, ensure_ascii=False)

    if mode == "age_offset" and "age" in g and isinstance(g["age"], int):
        g["age"] = int(g["age"]) + rng.choice([-3, -2, -1, 1, 2, 3])
        return json.dumps(g, ensure_ascii=False)

    if mode == "city_swap" and "city" in g:
        g["city"] = choose_other(g["city"], ["Shanghai", "Beijing", "Shenzhen", "Hangzhou", "Chengdu"], rng)
        return json.dumps(g, ensure_ascii=False)

    if mode == "drop_field":
        keys = list(g.keys())
        if keys:
            k = rng.choice(keys)
            g.pop(k, None)
        return json.dumps(g, ensure_ascii=False)

    if mode == "stringify_number" and "score" in g:
        g["score"] = str(g["score"])
        return json.dumps(g, ensure_ascii=False)

    if mode == "wrong_date" and "join_date" in g and isinstance(g["join_date"], str):
        g["join_date"] = "2099-01-01"
        return json.dumps(g, ensure_ascii=False)

    return "I cannot extract the fields from this text."


def main() -> None:
    ap = argparse.ArgumentParser(description="Build synthetic DPO pairs from gold labels (chosen=gold, rejected=corruption).")
    ap.add_argument("--dataset-jsonl", type=str, required=True, help='rows: {"id","prompt","gold"}')
    ap.add_argument("--num-rejected-per-sample", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    if args.num_rejected_per_sample <= 0:
        die("--num-rejected-per-sample must be > 0")

    ds_path = Path(args.dataset_jsonl).expanduser().resolve()
    if not ds_path.exists():
        die(f"dataset not found: {ds_path}")
    rows = load_jsonl(ds_path)
    if not rows:
        die("empty dataset")

    rng = random.Random(args.seed)
    pairs: List[Dict[str, Any]] = []
    for r in rows:
        rid = str(r.get("id", ""))
        prompt = str(r.get("prompt", ""))
        gold = r.get("gold", None)
        if not rid or not prompt or not isinstance(gold, dict):
            continue
        chosen = json.dumps(gold, ensure_ascii=False)
        for j in range(args.num_rejected_per_sample):
            rejected = corrupt_gold(gold, rng)
            pairs.append(
                {
                    "id": f"{rid}_neg{j+1}",
                    "source_id": rid,
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                }
            )

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (Path.cwd() / "reports" / f"stage52_synth_pairs_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "stage52_synth_pairs.jsonl"
    out_md = out_dir / "stage52_synth_pairs_summary.md"

    with out_jsonl.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    lines = [
        "# Stage 5.2 Synthetic DPO Pairs From Gold",
        "",
        f"- Generated at: `{dt.datetime.now().isoformat(timespec='seconds')}`",
        f"- Input rows: `{len(rows)}`",
        f"- num_rejected_per_sample: `{args.num_rejected_per_sample}`",
        f"- Output pairs: `{len(pairs)}`",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("[done] stage5.2 synthetic pairs")
    print(f"- pairs jsonl: {out_jsonl}")
    print(f"- summary md: {out_md}")


if __name__ == "__main__":
    main()

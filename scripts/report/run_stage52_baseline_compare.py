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

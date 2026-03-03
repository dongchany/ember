#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple


NUM_FIELDS = [
    "prefill_ms",
    "decode_per_token_ms",
    "rollout_total_ms",
    "rollout_tok_s",
]


def safe_float(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def load_compare_csv(path: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    out: Dict[Tuple[str, str], Dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(r.get("engine", ""), r.get("scenario", ""))] = r
    return out


def mean(vals: List[float]) -> float:
    return sum(vals) / float(len(vals)) if vals else float("nan")


def stddev(vals: List[float]) -> float:
    if len(vals) <= 1:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / float(len(vals) - 1))


def fmt(x: float, digits: int = 3) -> str:
    if math.isnan(x):
        return ""
    return f"{x:.{digits}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize repeated framework compare CSVs.")
    ap.add_argument("--inputs", type=str, nargs="+", required=True, help="framework_compare.csv paths")
    ap.add_argument("--out-dir", type=str, required=True)
    args = ap.parse_args()

    csv_paths = [Path(p).expanduser().resolve() for p in args.inputs]
    for p in csv_paths:
        if not p.exists():
            raise FileNotFoundError(f"missing input csv: {p}")

    runs = [load_compare_csv(p) for p in csv_paths]
    keys = sorted(set().union(*[set(r.keys()) for r in runs]))

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "framework_compare_repeat_summary.csv"
    summary_md = out_dir / "framework_compare_repeat_summary.md"
    per_run_csv = out_dir / "framework_compare_repeat_runs.csv"

    summary_rows: List[Dict[str, str]] = []
    per_run_rows: List[Dict[str, str]] = []

    for engine, scenario in keys:
        valid_rows: List[Dict[str, str]] = []
        for run_id, run in enumerate(runs, start=1):
            r = run.get((engine, scenario))
            if not r or r.get("status") != "ok":
                continue
            valid_rows.append(r)
            per_run_rows.append(
                {
                    "run_id": str(run_id),
                    "engine": engine,
                    "scenario": scenario,
                    "rollout_tok_s": r.get("rollout_tok_s", ""),
                    "rollout_total_ms": r.get("rollout_total_ms", ""),
                    "prefill_ms": r.get("prefill_ms", ""),
                    "decode_per_token_ms": r.get("decode_per_token_ms", ""),
                }
            )

        if not valid_rows:
            continue

        stats: Dict[str, float] = {}
        for f in NUM_FIELDS:
            vals = [safe_float(r.get(f, "")) for r in valid_rows]
            vals = [v for v in vals if not math.isnan(v)]
            stats[f + "_mean"] = mean(vals)
            stats[f + "_std"] = stddev(vals)
            stats[f + "_min"] = min(vals) if vals else float("nan")
            stats[f + "_max"] = max(vals) if vals else float("nan")

        tok_mean = stats["rollout_tok_s_mean"]
        tok_std = stats["rollout_tok_s_std"]
        tok_cv = (tok_std / tok_mean * 100.0) if tok_mean and not math.isnan(tok_mean) else float("nan")

        summary_rows.append(
            {
                "engine": engine,
                "scenario": scenario,
                "n_runs": str(len(valid_rows)),
                "rollout_tok_s_mean": fmt(tok_mean, 3),
                "rollout_tok_s_std": fmt(tok_std, 3),
                "rollout_tok_s_cv_pct": fmt(tok_cv, 2),
                "rollout_tok_s_min": fmt(stats["rollout_tok_s_min"], 3),
                "rollout_tok_s_max": fmt(stats["rollout_tok_s_max"], 3),
                "rollout_total_ms_mean": fmt(stats["rollout_total_ms_mean"], 3),
                "prefill_ms_mean": fmt(stats["prefill_ms_mean"], 3),
                "decode_per_token_ms_mean": fmt(stats["decode_per_token_ms_mean"], 3),
            }
        )

    summary_rows.sort(key=lambda r: (r["engine"], r["scenario"]))
    per_run_rows.sort(key=lambda r: (r["engine"], r["scenario"], int(r["run_id"])))

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "engine",
            "scenario",
            "n_runs",
            "rollout_tok_s_mean",
            "rollout_tok_s_std",
            "rollout_tok_s_cv_pct",
            "rollout_tok_s_min",
            "rollout_tok_s_max",
            "rollout_total_ms_mean",
            "prefill_ms_mean",
            "decode_per_token_ms_mean",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    with per_run_csv.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "run_id",
            "engine",
            "scenario",
            "rollout_tok_s",
            "rollout_total_ms",
            "prefill_ms",
            "decode_per_token_ms",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in per_run_rows:
            w.writerow(r)

    lines: List[str] = []
    lines.append("# Framework Compare Repeat Summary")
    lines.append("")
    lines.append("## Inputs")
    for i, p in enumerate(csv_paths, start=1):
        lines.append(f"- run{i}: `{p}`")
    lines.append("")
    lines.append("| engine | scenario | n | tok/s mean | tok/s std | tok/s cv% | tok/s min | tok/s max |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    r["engine"],
                    r["scenario"],
                    r["n_runs"],
                    r["rollout_tok_s_mean"],
                    r["rollout_tok_s_std"],
                    r["rollout_tok_s_cv_pct"],
                    r["rollout_tok_s_min"],
                    r["rollout_tok_s_max"],
                ]
            )
            + " |"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[done] summary_csv={summary_csv}")
    print(f"[done] per_run_csv={per_run_csv}")
    print(f"[done] summary_md={summary_md}")


if __name__ == "__main__":
    main()

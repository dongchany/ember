#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import subprocess
from pathlib import Path
from typing import Dict, List


def die(msg: str) -> None:
    raise SystemExit(f"error: {msg}")


def split_ints(s: str) -> List[int]:
    out: List[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out


def split_strs(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def run_cmd(cmd: List[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        if p.stdout:
            f.write(p.stdout)
        if p.stderr:
            f.write("\n[stderr]\n")
            f.write(p.stderr)
    if p.returncode != 0:
        die(f"command failed rc={p.returncode}: {' '.join(cmd)} (see {log_path})")


def read_single_row_csv(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        die(f"expected exactly one row in {path}, got {len(rows)}")
    return rows[0]


def to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def write_summary_md(
    path: Path,
    rows: List[Dict[str, str]],
    model: str,
    adapter: str,
    gpus: str,
    split: str,
) -> None:
    lines: List[str] = []
    lines.append("# Stage 3.1 LoRA Weight Merge Check")
    lines.append("")
    lines.append(f"- Generated at: `{dt.datetime.now().isoformat(timespec='seconds')}`")
    lines.append(f"- Model: `{model}`")
    lines.append(f"- Adapter: `{adapter}`")
    lines.append(f"- GPUs: `{gpus}`")
    lines.append(f"- Split: `{split}`")
    lines.append("")
    lines.append("| layer | proj | delta_max_abs_diff | delta_mean_abs_diff | rollback_max_abs_diff | rollback_mean_abs_diff |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in rows:
        lines.append(
            f"| {r['layer_idx']} | {r['proj']} | {r['delta_max_abs_diff']} | {r['delta_mean_abs_diff']} | "
            f"{r['rollback_max_abs_diff']} | {r['rollback_mean_abs_diff']} |"
        )

    if rows:
        worst = max(rows, key=lambda r: to_float(r.get("delta_max_abs_diff", "nan")))
        lines.append("")
        lines.append("## Key Point")
        lines.append(
            f"- Worst `delta_max_abs_diff`: layer `{worst['layer_idx']}` `{worst['proj']}` = "
            f"`{worst['delta_max_abs_diff']}`."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Stage 3.1 LoRA weight-merge numerical checks.")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--adapter", type=str, required=True)
    ap.add_argument("--bench-bin", type=str, default="build/ember_lora_weight_merge_check")
    ap.add_argument("--gpus", type=str, default="0")
    ap.add_argument("--split", type=str, default="")
    ap.add_argument("--layers", type=str, default="0")
    ap.add_argument("--projs", type=str, default="q_proj,k_proj,v_proj,o_proj")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    repo = Path.cwd()
    bench_bin = Path(args.bench_bin).expanduser().resolve()
    if not bench_bin.exists():
        die(f"missing benchmark binary: {bench_bin}")

    layers = split_ints(args.layers)
    projs = split_strs(args.projs)
    if not layers:
        die("--layers is empty")
    if not projs:
        die("--projs is empty")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (repo / "reports" / f"stage31_lora_weight_merge_check_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    for layer in layers:
        for proj in projs:
            case_tag = f"layer{layer}_{proj}"
            case_csv = out_dir / f"{case_tag}.csv"
            cmd = [
                str(bench_bin),
                "--model", args.model,
                "--adapter", args.adapter,
                "--gpus", args.gpus,
                "--layer", str(layer),
                "--proj", proj,
                "--scale", str(args.scale),
                "--csv", str(case_csv),
            ]
            if args.split.strip():
                cmd += ["--split", args.split]
            run_cmd(cmd=cmd, cwd=repo, log_path=logs_dir / f"{case_tag}.log")
            rows.append(read_single_row_csv(case_csv))

    # Stable sort for readability.
    rows.sort(key=lambda r: (int(r.get("layer_idx", "0")), r.get("proj", "")))
    summary_csv = out_dir / "stage31_lora_weight_merge_check.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        fields = list(rows[0].keys()) if rows else []
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    summary_md = out_dir / "stage31_summary.md"
    write_summary_md(
        path=summary_md,
        rows=rows,
        model=args.model,
        adapter=args.adapter,
        gpus=args.gpus,
        split=args.split or "(none)",
    )

    print(f"[done] out_dir={out_dir}")
    print(f"[done] summary_csv={summary_csv}")
    print(f"[done] summary_md={summary_md}")


if __name__ == "__main__":
    main()

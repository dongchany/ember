#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


def run_cmd(cmd: List[str], cwd: Path, log_path: Path) -> subprocess.CompletedProcess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        p = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
        f.write(p.stdout)
        if p.stderr:
            f.write("\n[stderr]\n")
            f.write(p.stderr)
    return p


def read_policy_csv(path: Path) -> List[Dict[str, str]]:
    lines: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            if "=" in ln and "," not in ln:
                continue
            lines.append(ln)
    if not lines:
        return []
    reader = csv.DictReader(lines)
    return list(reader)


def safe_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def safe_int(v: str, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return default


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 3.3 cache policy simulation report.")
    ap.add_argument("--num-layers", type=int, default=36)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--freeze-layers", type=int, default=18)
    ap.add_argument("--periodic-refresh-k", type=int, default=10)
    ap.add_argument("--policies", type=str, default="naive,update_locality,periodic_refresh")
    ap.add_argument("--build-dir", type=str, default="build")
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    if args.num_layers <= 0:
        die("--num-layers must be > 0")
    if args.rounds <= 0:
        die("--rounds must be > 0")
    if args.freeze_layers < 0 or args.freeze_layers > args.num_layers:
        die("--freeze-layers out of range")
    if args.periodic_refresh_k < 0:
        die("--periodic-refresh-k must be >= 0")

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    if not policies:
        die("--policies empty")

    repo = Path.cwd()
    sim_bin = (repo / args.build_dir / "ember_cache_policy_sim").resolve()
    if not sim_bin.exists():
        die(f"missing binary: {sim_bin}")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (repo / "reports" / f"stage33_cache_policy_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    merged_rows: List[Dict[str, str]] = []
    summary_rows: List[Dict[str, str]] = []

    for policy in policies:
        run_csv = out_dir / f"stage33_{policy}.csv"
        cmd = [
            str(sim_bin),
            "--policy",
            policy,
            "--num-layers",
            str(args.num_layers),
            "--rounds",
            str(args.rounds),
            "--freeze-layers",
            str(args.freeze_layers),
            "--periodic-refresh-k",
            str(args.periodic_refresh_k),
            "--csv",
            str(run_csv),
        ]
        p = run_cmd(cmd, cwd=repo, log_path=logs_dir / f"{policy}.log")
        if p.returncode != 0:
            die(f"policy sim failed for {policy}; see {logs_dir / (policy + '.log')}")

        rows = read_policy_csv(run_csv)
        if not rows:
            die(f"empty csv for policy={policy}: {run_csv}")
        merged_rows.extend(rows)

        avg_ratio = sum(safe_float(r.get("recompute_ratio", "0")) for r in rows) / float(len(rows))
        refreshes = sum(safe_int(r.get("full_refresh", "0")) for r in rows)
        first = rows[0]
        summary_rows.append(
            {
                "policy": policy,
                "rounds": str(len(rows)),
                "num_layers": first.get("num_layers", str(args.num_layers)),
                "freeze_layers": first.get("freeze_layers", str(args.freeze_layers)),
                "periodic_refresh_k": first.get("periodic_refresh_k", str(args.periodic_refresh_k)),
                "full_refreshes": str(refreshes),
                "avg_recompute_ratio": f"{avg_ratio:.6f}",
                "avg_reused_ratio": f"{(1.0 - avg_ratio):.6f}",
            }
        )

    merged_csv = out_dir / "stage33_policy_per_round.csv"
    summary_csv = out_dir / "stage33_policy_summary.csv"
    summary_md = out_dir / "stage33_summary.md"
    p1_input_md = out_dir / "stage33_p1_input.md"

    write_csv(merged_csv, merged_rows)
    write_csv(summary_csv, summary_rows)

    lines = [
        "# Stage 3.3 Cache Policy Simulation",
        "",
        f"- Generated at: `{dt.datetime.now().isoformat(timespec='seconds')}`",
        f"- num_layers={args.num_layers}, rounds={args.rounds}, freeze_layers={args.freeze_layers}, periodic_refresh_k={args.periodic_refresh_k}",
        "",
        "| policy | full_refreshes | avg_recompute_ratio | avg_reused_ratio |",
        "| --- | --- | --- | --- |",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['policy']} | {r['full_refreshes']} | {r['avg_recompute_ratio']} | {r['avg_reused_ratio']} |"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    best = min(summary_rows, key=lambda r: safe_float(r.get("avg_recompute_ratio", "1")))
    p1_lines = [
        f"# Stage3.3 P1 Input ({dt.date.today().isoformat()})",
        "",
        f"Setting: num_layers={args.num_layers}, rounds={args.rounds}, freeze_layers={args.freeze_layers}, periodic_refresh_k={args.periodic_refresh_k}",
        "",
        "## Policy summary",
    ]
    for r in summary_rows:
        p1_lines.append(
            f"- {r['policy']}: avg_recompute_ratio={r['avg_recompute_ratio']}, "
            f"avg_reused_ratio={r['avg_reused_ratio']}, full_refreshes={r['full_refreshes']}"
        )
    p1_lines += [
        "",
        f"Best compute-saving policy in this sweep: {best['policy']} (avg_recompute_ratio={best['avg_recompute_ratio']}).",
    ]
    p1_input_md.write_text("\n".join(p1_lines) + "\n", encoding="utf-8")

    print(f"[done] out_dir={out_dir}")
    print(f"- per-round: {merged_csv}")
    print(f"- summary: {summary_csv}")
    print(f"- md: {summary_md}")
    print(f"- p1 input: {p1_input_md}")


if __name__ == "__main__":
    main()


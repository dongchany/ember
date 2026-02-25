#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
from pathlib import Path
from typing import Dict, List


def die(msg: str) -> None:
    raise SystemExit(f"error: {msg}")


def safe_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def read_csv(path: Path) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.append(r)
    return out


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def find_stage12_row(
    rows: List[Dict[str, str]], split: str, mode: str, prompt_len: int, decode_steps: int
) -> Dict[str, str]:
    for r in rows:
        if (
            r.get("split", "") == split
            and r.get("mode", "") == mode
            and int(r.get("prompt_len", "0")) == prompt_len
            and int(r.get("decode_steps", "0")) == decode_steps
        ):
            return r
    die(
        "failed to find stage12 row for "
        f"split={split}, mode={mode}, prompt_len={prompt_len}, decode_steps={decode_steps}"
    )
    raise AssertionError("unreachable")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 4.2 locality sweep (simulation).")
    ap.add_argument("--stage12-summary", type=str, required=True)
    ap.add_argument("--split", type=str, default="9+27")
    ap.add_argument("--mode", type=str, default="overlap", choices=["overlap", "no_overlap"])
    ap.add_argument("--prompt-len", type=int, default=2048)
    ap.add_argument("--decode-steps", type=int, default=128)
    ap.add_argument("--num-prompts", type=int, default=100)
    ap.add_argument("--num-candidates", type=int, default=8)
    ap.add_argument("--num-rounds", type=int, default=30)
    ap.add_argument("--num-gpus", type=int, default=2)
    ap.add_argument("--recompute-ratios", type=str, default="1.0,0.75,0.5,0.25,0.0")
    ap.add_argument("--periodic-refresh-k", type=int, default=10)
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    if args.prompt_len <= 0 or args.decode_steps <= 0:
        die("--prompt-len and --decode-steps must be > 0")
    if args.num_prompts <= 0 or args.num_candidates <= 0:
        die("--num-prompts and --num-candidates must be > 0")
    if args.num_rounds <= 0:
        die("--num-rounds must be > 0")
    if args.num_gpus <= 0:
        die("--num-gpus must be > 0")
    if args.periodic_refresh_k < 0:
        die("--periodic-refresh-k must be >= 0")

    ratios: List[float] = []
    for tok in args.recompute_ratios.split(","):
        tok = tok.strip()
        if not tok:
            continue
        val = float(tok)
        if val < 0.0 or val > 1.0:
            die(f"invalid recompute ratio: {val}")
        ratios.append(val)
    if not ratios:
        die("--recompute-ratios is empty")

    stage12_path = Path(args.stage12_summary).expanduser().resolve()
    rows = read_csv(stage12_path)
    row = find_stage12_row(rows, args.split, args.mode, args.prompt_len, args.decode_steps)
    prefill_full_ms = safe_float(row.get("prefill_wall_ms", "0"))
    decode_per_token_ms = safe_float(row.get("decode_per_token_ms", "0"))
    decode_total_ms = decode_per_token_ms * float(args.decode_steps)

    requests_per_round = args.num_prompts * args.num_candidates
    # baseline is naive/full recompute every round.
    baseline_round_ms = float(requests_per_round) * (prefill_full_ms + decode_total_ms)
    baseline_total_ms = baseline_round_ms * float(args.num_rounds)

    out_rows: List[Dict[str, str]] = []
    for ratio in ratios:
        cumulative_ms = 0.0
        for rnd in range(1, args.num_rounds + 1):
            if rnd <= 1:
                prefill_this = prefill_full_ms
            else:
                refresh = (args.periodic_refresh_k > 0 and rnd % args.periodic_refresh_k == 0)
                prefill_this = prefill_full_ms if refresh else (prefill_full_ms * ratio)
            round_ms = float(requests_per_round) * (prefill_this + decode_total_ms)
            cumulative_ms += round_ms

        wall_h = cumulative_ms / 1000.0 / 3600.0
        gpu_h = wall_h * float(args.num_gpus)
        speedup_vs_naive = (baseline_total_ms / cumulative_ms) if cumulative_ms > 0.0 else 0.0
        reduction_vs_naive = (
            (baseline_total_ms - cumulative_ms) / baseline_total_ms * 100.0 if baseline_total_ms > 0.0 else 0.0
        )
        freeze_pct_proxy = (1.0 - ratio) * 100.0

        out_rows.append(
            {
                "recompute_ratio": f"{ratio:.3f}",
                "freeze_pct_proxy": f"{freeze_pct_proxy:.1f}",
                "cumulative_ms": f"{cumulative_ms:.3f}",
                "cumulative_wall_hours": f"{wall_h:.6f}",
                "cumulative_gpu_hours": f"{gpu_h:.6f}",
                "speedup_vs_naive_x": f"{speedup_vs_naive:.4f}",
                "reduction_vs_naive_pct": f"{reduction_vs_naive:.3f}",
            }
        )

    out_rows.sort(key=lambda r: safe_float(r["recompute_ratio"]), reverse=True)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (Path.cwd() / "reports" / f"stage42_locality_sweep_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "stage42_locality_sweep.csv"
    out_md = out_dir / "stage42_locality_sweep.md"
    out_p1 = out_dir / "stage42_p1_input.md"
    write_csv(out_csv, out_rows)

    lines = [
        "# Stage 4.2 Locality Sweep (Simulation)",
        "",
        f"- Generated at: `{dt.datetime.now().isoformat(timespec='seconds')}`",
        f"- Base row: split={args.split}, mode={args.mode}, prompt_len={args.prompt_len}, decode_steps={args.decode_steps}",
        f"- Requests/round: {requests_per_round}, rounds={args.num_rounds}, periodic_refresh_k={args.periodic_refresh_k}",
        "",
        "| recompute_ratio | freeze_pct_proxy | cumulative_gpu_hours | speedup_vs_naive_x | reduction_vs_naive_% |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in out_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    r["recompute_ratio"],
                    r["freeze_pct_proxy"],
                    r["cumulative_gpu_hours"],
                    r["speedup_vs_naive_x"],
                    r["reduction_vs_naive_pct"],
                ]
            )
            + " |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    best = max(out_rows, key=lambda r: safe_float(r["speedup_vs_naive_x"]))
    p1_lines = [
        f"# Stage4.2 P1 Input ({dt.date.today().isoformat()})",
        "",
        f"Setting: split={args.split}, mode={args.mode}, prompt_len={args.prompt_len}, decode_steps={args.decode_steps}",
        f"Rounds={args.num_rounds}, requests_per_round={requests_per_round}, periodic_refresh_k={args.periodic_refresh_k}",
        "",
        "## Best speedup point (simulation)",
        f"- recompute_ratio={best['recompute_ratio']} (freeze_proxy={best['freeze_pct_proxy']}%)",
        f"- speedup_vs_naive={best['speedup_vs_naive_x']}x, reduction_vs_naive={best['reduction_vs_naive_pct']}%",
    ]
    out_p1.write_text("\n".join(p1_lines) + "\n", encoding="utf-8")

    print("[done] stage4.2 locality sweep")
    print(f"- csv: {out_csv}")
    print(f"- md: {out_md}")
    print(f"- p1 input: {out_p1}")


if __name__ == "__main__":
    main()

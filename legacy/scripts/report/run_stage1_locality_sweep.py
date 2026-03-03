#!/usr/bin/env python3
import argparse
import datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple

from common_report import die, read_csv, safe_float, write_csv


def load_freeze_quality_points(path: Path) -> List[Tuple[int, float]]:
    rows = read_csv(path)
    pts: List[Tuple[int, float]] = []
    for r in rows:
        try:
            n = int(r.get("freeze_layers", "0"))
            q = float(r.get("frozen_prefix_max_delta", "0"))
        except Exception:
            continue
        pts.append((n, q))
    if not pts:
        die(f"no valid freeze quality points in {path}")
    pts.sort(key=lambda x: x[0])
    return pts


def interp_quality(freeze_layers: int, points: List[Tuple[int, float]]) -> Tuple[float, str]:
    # Piecewise linear interpolation over measured freeze points.
    if freeze_layers <= points[0][0]:
        return points[0][1], "clamp_low"
    if freeze_layers >= points[-1][0]:
        return points[-1][1], "clamp_high"
    for i in range(1, len(points)):
        l0, q0 = points[i - 1]
        l1, q1 = points[i]
        if freeze_layers == l1:
            return q1, "exact"
        if l0 <= freeze_layers <= l1:
            if l1 == l0:
                return q1, "exact"
            t = float(freeze_layers - l0) / float(l1 - l0)
            return q0 + (q1 - q0) * t, "interp"
    return points[-1][1], "clamp_high"


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
    ap.add_argument("--num-layers", type=int, default=36)
    ap.add_argument("--recompute-ratios", type=str, default="1.0,0.75,0.5,0.25,0.0")
    ap.add_argument("--periodic-refresh-k", type=int, default=10)
    ap.add_argument("--delta-freeze-summary", type=str, default="", help="stage31_lora_delta_freeze_summary.csv")
    ap.add_argument("--quality-threshold", type=float, default=0.300000)
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
    if args.num_layers <= 0:
        die("--num-layers must be > 0")
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

    quality_points: List[Tuple[int, float]] = []
    quality_note = "quality proxy disabled"
    if args.delta_freeze_summary.strip():
        qp = Path(args.delta_freeze_summary).expanduser().resolve()
        if not qp.exists():
            die(f"--delta-freeze-summary not found: {qp}")
        quality_points = load_freeze_quality_points(qp)
        quality_note = f"quality proxy from {qp.name}, threshold={args.quality_threshold:.6f}"

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
        freeze_layers_proxy = int(round((1.0 - ratio) * float(args.num_layers)))
        freeze_layers_proxy = max(0, min(args.num_layers, freeze_layers_proxy))
        quality_proxy = 0.0
        quality_source = "na"
        quality_ok = "na"
        if quality_points:
            quality_proxy, quality_source = interp_quality(freeze_layers_proxy, quality_points)
            quality_ok = "1" if quality_proxy <= args.quality_threshold else "0"

        out_rows.append(
            {
                "recompute_ratio": f"{ratio:.3f}",
                "freeze_pct_proxy": f"{freeze_pct_proxy:.1f}",
                "freeze_layers_proxy": str(freeze_layers_proxy),
                "cumulative_ms": f"{cumulative_ms:.3f}",
                "cumulative_wall_hours": f"{wall_h:.6f}",
                "cumulative_gpu_hours": f"{gpu_h:.6f}",
                "speedup_vs_naive_x": f"{speedup_vs_naive:.4f}",
                "reduction_vs_naive_pct": f"{reduction_vs_naive:.3f}",
                "quality_proxy_delta_max": f"{quality_proxy:.6f}",
                "quality_proxy_source": quality_source,
                "quality_ok": quality_ok,
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
        f"- {quality_note}",
        "",
        "| recompute_ratio | freeze_pct_proxy | freeze_layers_proxy | cumulative_gpu_hours | speedup_vs_naive_x | reduction_vs_naive_% | quality_proxy_delta_max | quality_ok |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in out_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    r["recompute_ratio"],
                    r["freeze_pct_proxy"],
                    r["freeze_layers_proxy"],
                    r["cumulative_gpu_hours"],
                    r["speedup_vs_naive_x"],
                    r["reduction_vs_naive_pct"],
                    r["quality_proxy_delta_max"],
                    r["quality_ok"],
                ]
            )
            + " |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if quality_points:
        feasible = [r for r in out_rows if r.get("quality_ok", "0") == "1"]
    else:
        feasible = out_rows
    best = max(feasible, key=lambda r: safe_float(r["speedup_vs_naive_x"])) if feasible else max(
        out_rows, key=lambda r: safe_float(r["speedup_vs_naive_x"])
    )
    p1_lines = [
        f"# Stage4.2 P1 Input ({dt.date.today().isoformat()})",
        "",
        f"Setting: split={args.split}, mode={args.mode}, prompt_len={args.prompt_len}, decode_steps={args.decode_steps}",
        f"Rounds={args.num_rounds}, requests_per_round={requests_per_round}, periodic_refresh_k={args.periodic_refresh_k}",
        quality_note,
        "",
        "## Recommended point (speedup under quality constraint)",
        f"- recompute_ratio={best['recompute_ratio']} (freeze_proxy={best['freeze_pct_proxy']}%)",
        f"- freeze_layers_proxy={best['freeze_layers_proxy']}, quality_proxy_delta_max={best['quality_proxy_delta_max']}, quality_ok={best['quality_ok']}",
        f"- speedup_vs_naive={best['speedup_vs_naive_x']}x, reduction_vs_naive={best['reduction_vs_naive_pct']}%",
    ]
    out_p1.write_text("\n".join(p1_lines) + "\n", encoding="utf-8")

    print("[done] stage4.2 locality sweep")
    print(f"- csv: {out_csv}")
    print(f"- md: {out_md}")
    print(f"- p1 input: {out_p1}")


if __name__ == "__main__":
    main()

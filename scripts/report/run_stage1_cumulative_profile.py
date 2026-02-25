#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional


def die(msg: str) -> None:
    raise SystemExit(f"error: {msg}")


def safe_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def read_csv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def find_stage12_row(
    rows: List[Dict[str, str]],
    split: str,
    mode: str,
    prompt_len: int,
    decode_steps: int,
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


def find_prefix_row(rows: List[Dict[str, str]], prefix_len: int) -> Optional[Dict[str, str]]:
    for r in rows:
        if int(r.get("prefix_len", "0")) == prefix_len:
            return r
    return None


def per_round_ms_naive(
    n_requests: int,
    prefill_full_ms: float,
    decode_total_ms: float,
) -> float:
    return float(n_requests) * (prefill_full_ms + decode_total_ms)


def per_round_ms_prefix_only(
    n_requests: int,
    prefix_once_ms: float,
    suffix_per_req_ms: float,
    decode_total_ms: float,
) -> float:
    return prefix_once_ms + float(n_requests) * (suffix_per_req_ms + decode_total_ms)


def per_round_ms_update_locality(
    n_requests: int,
    prefill_full_ms: float,
    decode_total_ms: float,
    locality_recompute_ratio: float,
    round_idx_1based: int,
    periodic_refresh_k: int,
) -> float:
    if round_idx_1based <= 1:
        prefill_this_round = prefill_full_ms
    else:
        refresh = (periodic_refresh_k > 0 and round_idx_1based % periodic_refresh_k == 0)
        prefill_this_round = prefill_full_ms if refresh else (prefill_full_ms * locality_recompute_ratio)
    return float(n_requests) * (prefill_this_round + decode_total_ms)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 1.4 cumulative rollout cost simulation.")
    ap.add_argument("--stage12-summary", type=str, required=True, help="stage12_split_summary.csv")
    ap.add_argument("--split", type=str, default="9+27")
    ap.add_argument("--mode", type=str, default="overlap", choices=["overlap", "no_overlap"])
    ap.add_argument("--prompt-len", type=int, default=2048)
    ap.add_argument("--decode-steps", type=int, default=128)
    ap.add_argument("--num-prompts", type=int, default=100)
    ap.add_argument("--num-candidates", type=int, default=8)
    ap.add_argument("--num-rounds", type=int, default=30)
    ap.add_argument("--num-gpus", type=int, default=2, help="used for GPU-hours conversion")
    ap.add_argument("--prefix-sweep", type=str, default="", help="stage13_prefix_cache_sweep.csv")
    ap.add_argument("--prefix-len", type=int, default=1024)
    ap.add_argument("--locality-recompute-ratio", type=float, default=0.5, help="0.5 means keep 50% prefill compute")
    ap.add_argument("--periodic-refresh-k", type=int, default=0, help="0 disables periodic full refresh")
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
    if not (0.0 <= args.locality_recompute_ratio <= 1.0):
        die("--locality-recompute-ratio must be in [0,1]")
    if args.periodic_refresh_k < 0:
        die("--periodic-refresh-k must be >= 0")

    stage12_path = Path(args.stage12_summary).expanduser().resolve()
    if not stage12_path.exists():
        die(f"missing stage12 summary: {stage12_path}")

    stage12_rows = read_csv(stage12_path)
    row = find_stage12_row(
        rows=stage12_rows,
        split=args.split,
        mode=args.mode,
        prompt_len=args.prompt_len,
        decode_steps=args.decode_steps,
    )
    prefill_full_ms = safe_float(row.get("prefill_wall_ms", "0"))
    decode_per_token_ms = safe_float(row.get("decode_per_token_ms", "0"))
    decode_total_ms = decode_per_token_ms * float(args.decode_steps)

    prefix_once_ms = 0.0
    suffix_per_req_ms = prefill_full_ms
    prefix_note = "prefix sweep unavailable; Prefix-only falls back to Naive prefill"
    if args.prefix_sweep.strip():
        prefix_path = Path(args.prefix_sweep).expanduser().resolve()
        if not prefix_path.exists():
            die(f"missing prefix sweep csv: {prefix_path}")
        prefix_rows = read_csv(prefix_path)
        prow = find_prefix_row(prefix_rows, args.prefix_len)
        if prow is None:
            die(f"prefix_len={args.prefix_len} not found in {prefix_path}")
        prefix_once_ms = safe_float(prow.get("cache_prefix_once_ms", "0"))
        num_docs_in_sweep = int(prow.get("num_docs", "0"))
        if num_docs_in_sweep <= 0:
            die("invalid num_docs in prefix sweep row")
        cache_suffix_total_ms = safe_float(prow.get("cache_suffix_total_ms", "0"))
        suffix_per_req_ms = cache_suffix_total_ms / float(num_docs_in_sweep)
        prefix_note = (
            f"prefix reuse constants from {prefix_path.name}: "
            f"prefix_once={prefix_once_ms:.3f}ms, suffix_per_req={suffix_per_req_ms:.3f}ms"
        )

    n_requests = args.num_prompts * args.num_candidates
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (Path.cwd() / "reports" / f"stage14_cumulative_profile_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    per_round_rows: List[Dict[str, str]] = []
    cumulative = {"naive": 0.0, "prefix_only": 0.0, "update_locality": 0.0}
    for rnd in range(1, args.num_rounds + 1):
        naive_ms = per_round_ms_naive(n_requests, prefill_full_ms, decode_total_ms)
        prefix_ms = per_round_ms_prefix_only(n_requests, prefix_once_ms, suffix_per_req_ms, decode_total_ms)
        locality_ms = per_round_ms_update_locality(
            n_requests=n_requests,
            prefill_full_ms=prefill_full_ms,
            decode_total_ms=decode_total_ms,
            locality_recompute_ratio=args.locality_recompute_ratio,
            round_idx_1based=rnd,
            periodic_refresh_k=args.periodic_refresh_k,
        )
        cumulative["naive"] += naive_ms
        cumulative["prefix_only"] += prefix_ms
        cumulative["update_locality"] += locality_ms

        for strategy, round_ms in [
            ("naive", naive_ms),
            ("prefix_only", prefix_ms),
            ("update_locality", locality_ms),
        ]:
            cum_ms = cumulative[strategy]
            wall_h = cum_ms / 1000.0 / 3600.0
            gpu_h = wall_h * float(args.num_gpus)
            per_round_rows.append(
                {
                    "round": str(rnd),
                    "strategy": strategy,
                    "round_ms": f"{round_ms:.3f}",
                    "cumulative_ms": f"{cum_ms:.3f}",
                    "cumulative_wall_hours": f"{wall_h:.6f}",
                    "cumulative_gpu_hours": f"{gpu_h:.6f}",
                }
            )

    per_round_csv = out_dir / "stage14_per_round.csv"
    summary_csv = out_dir / "stage14_summary.csv"
    summary_md = out_dir / "stage14_summary.md"
    p1_input_md = out_dir / "stage14_p1_input.md"
    write_csv(per_round_csv, per_round_rows)

    end_naive = cumulative["naive"]
    end_prefix = cumulative["prefix_only"]
    end_locality = cumulative["update_locality"]

    def reduction_pct(base: float, other: float) -> float:
        if base <= 0.0:
            return 0.0
        return (base - other) / base * 100.0

    summary_rows = [
        {
            "num_rounds": str(args.num_rounds),
            "strategy": "naive",
            "cumulative_ms": f"{end_naive:.3f}",
            "cumulative_wall_hours": f"{(end_naive / 1000.0 / 3600.0):.6f}",
            "cumulative_gpu_hours": f"{(end_naive / 1000.0 / 3600.0 * args.num_gpus):.6f}",
            "reduction_vs_naive_pct": "0.000",
        },
        {
            "num_rounds": str(args.num_rounds),
            "strategy": "prefix_only",
            "cumulative_ms": f"{end_prefix:.3f}",
            "cumulative_wall_hours": f"{(end_prefix / 1000.0 / 3600.0):.6f}",
            "cumulative_gpu_hours": f"{(end_prefix / 1000.0 / 3600.0 * args.num_gpus):.6f}",
            "reduction_vs_naive_pct": f"{reduction_pct(end_naive, end_prefix):.3f}",
        },
        {
            "num_rounds": str(args.num_rounds),
            "strategy": "update_locality",
            "cumulative_ms": f"{end_locality:.3f}",
            "cumulative_wall_hours": f"{(end_locality / 1000.0 / 3600.0):.6f}",
            "cumulative_gpu_hours": f"{(end_locality / 1000.0 / 3600.0 * args.num_gpus):.6f}",
            "reduction_vs_naive_pct": f"{reduction_pct(end_naive, end_locality):.3f}",
        },
    ]
    write_csv(summary_csv, summary_rows)

    lines = [
        "# Stage 1.4 Cumulative Cost Simulation",
        "",
        f"- Generated at: `{dt.datetime.now().isoformat(timespec='seconds')}`",
        f"- Base row: split={args.split}, mode={args.mode}, prompt_len={args.prompt_len}, decode_steps={args.decode_steps}",
        f"- Requests per round: num_prompts={args.num_prompts}, num_candidates={args.num_candidates}, total={n_requests}",
        f"- Locality assumptions: recompute_ratio={args.locality_recompute_ratio}, periodic_refresh_k={args.periodic_refresh_k}",
        f"- Prefix assumptions: {prefix_note}",
        "",
        "| strategy | cumulative_ms | cumulative_wall_hours | cumulative_gpu_hours | reduction_vs_naive_% |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    r["strategy"],
                    r["cumulative_ms"],
                    r["cumulative_wall_hours"],
                    r["cumulative_gpu_hours"],
                    r["reduction_vs_naive_pct"],
                ]
            )
            + " |"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    p1_lines = [
        f"# Stage1.4 P1 Input ({dt.date.today().isoformat()})",
        "",
        f"Setting: rounds={args.num_rounds}, prompts={args.num_prompts}, candidates={args.num_candidates}, split={args.split}, mode={args.mode}",
        f"Base timing: prefill_full_ms={prefill_full_ms:.3f}, decode_per_token_ms={decode_per_token_ms:.3f}, decode_total_ms={decode_total_ms:.3f}",
        f"Prefix constants: prefix_once_ms={prefix_once_ms:.3f}, suffix_per_req_ms={suffix_per_req_ms:.3f}",
        f"Locality constants: recompute_ratio={args.locality_recompute_ratio:.3f}, periodic_refresh_k={args.periodic_refresh_k}",
        "",
        "## 30-round cumulative (or configured rounds)",
    ]
    for r in summary_rows:
        p1_lines.append(
            f"- {r['strategy']}: cumulative_gpu_hours={r['cumulative_gpu_hours']}, "
            f"reduction_vs_naive={r['reduction_vs_naive_pct']}%"
        )
    p1_input_md.write_text("\n".join(p1_lines) + "\n", encoding="utf-8")

    print("[done] stage1.4 cumulative profile")
    print(f"- per-round csv: {per_round_csv}")
    print(f"- summary csv: {summary_csv}")
    print(f"- summary md: {summary_md}")
    print(f"- p1 input md: {p1_input_md}")


if __name__ == "__main__":
    main()

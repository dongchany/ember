#!/usr/bin/env python3
import argparse
import datetime as dt
from pathlib import Path
from typing import Dict, List, Tuple

from common_report import die, read_csv, safe_float, write_csv


def find_row(rows: List[Dict[str, str]], strategy: str) -> Dict[str, str]:
    for r in rows:
        if r.get("strategy", "") == strategy:
            return r
    die(f"strategy='{strategy}' missing")
    raise AssertionError("unreachable")


def load_policy_summary(path: Path) -> Dict[str, Dict[str, str]]:
    rows = read_csv(path)
    out: Dict[str, Dict[str, str]] = {}
    for r in rows:
        name = r.get("policy", "")
        if name:
            out[name] = r
    if not out:
        die(f"empty policy summary: {path}")
    return out


def load_locality_quality_points(path: Path) -> List[Tuple[float, float]]:
    rows = read_csv(path)
    pts: List[Tuple[float, float]] = []
    for r in rows:
        rr = safe_float(r.get("recompute_ratio", ""), -1.0)
        q = safe_float(r.get("quality_proxy_delta_max", ""), -1.0)
        if rr < 0.0 or q < 0.0:
            continue
        pts.append((rr, q))
    if not pts:
        die(f"no quality points in {path}")
    pts.sort(key=lambda x: x[0])
    return pts


def interp_quality_by_ratio(recompute_ratio: float, points: List[Tuple[float, float]]) -> float:
    if recompute_ratio <= points[0][0]:
        return points[0][1]
    if recompute_ratio >= points[-1][0]:
        return points[-1][1]
    for i in range(1, len(points)):
        r0, q0 = points[i - 1]
        r1, q1 = points[i]
        if abs(recompute_ratio - r1) < 1e-9:
            return q1
        if r0 <= recompute_ratio <= r1:
            if abs(r1 - r0) < 1e-9:
                return q1
            t = (recompute_ratio - r0) / (r1 - r0)
            return q0 + (q1 - q0) * t
    return points[-1][1]


def add_row(
    out_rows: List[Dict[str, str]],
    *,
    scenario: str,
    strategy: str,
    row: Dict[str, str],
    source: str,
    recompute_ratio: float,
    cache_hit_rate: float,
    quality_points: List[Tuple[float, float]],
    quality_threshold: float,
    with_quality: bool = True,
) -> None:
    quality = ""
    quality_ok = ""
    if with_quality and recompute_ratio >= 0.0:
        qv = interp_quality_by_ratio(recompute_ratio, quality_points)
        quality = f"{qv:.6f}"
        quality_ok = "1" if qv <= quality_threshold else "0"

    out_rows.append(
        {
            "scenario": scenario,
            "strategy": strategy,
            "cumulative_gpu_hours": row.get("cumulative_gpu_hours", ""),
            "reduction_vs_naive_pct": row.get("reduction_vs_naive_pct", ""),
            "avg_recompute_ratio": "" if recompute_ratio < 0.0 else f"{recompute_ratio:.6f}",
            "cache_hit_rate_proxy": "" if cache_hit_rate < 0.0 else f"{cache_hit_rate:.6f}",
            "quality_proxy_delta_max": quality,
            "quality_ok": quality_ok,
            "source_summary": source,
        }
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 4.3 strategy table aggregator.")
    ap.add_argument("--summary-2048-update", type=str, required=True)
    ap.add_argument("--summary-2048-periodic", type=str, required=True)
    ap.add_argument("--summary-4096-update", type=str, required=True)
    ap.add_argument("--summary-4096-periodic", type=str, required=True)
    ap.add_argument("--policy-summary", type=str, required=True)
    ap.add_argument("--locality-sweep", type=str, required=True)
    ap.add_argument("--quality-threshold", type=float, default=0.300000)
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    p_2048_u = Path(args.summary_2048_update).expanduser().resolve()
    p_2048_p = Path(args.summary_2048_periodic).expanduser().resolve()
    p_4096_u = Path(args.summary_4096_update).expanduser().resolve()
    p_4096_p = Path(args.summary_4096_periodic).expanduser().resolve()
    p_policy = Path(args.policy_summary).expanduser().resolve()
    p_locality = Path(args.locality_sweep).expanduser().resolve()
    for p in [p_2048_u, p_2048_p, p_4096_u, p_4096_p, p_policy, p_locality]:
        if not p.exists():
            die(f"missing file: {p}")

    rows_2048_u = read_csv(p_2048_u)
    rows_2048_p = read_csv(p_2048_p)
    rows_4096_u = read_csv(p_4096_u)
    rows_4096_p = read_csv(p_4096_p)
    policy = load_policy_summary(p_policy)
    quality_points = load_locality_quality_points(p_locality)

    recompute_naive = safe_float(policy.get("naive", {}).get("avg_recompute_ratio", "1"), 1.0)
    hit_naive = safe_float(policy.get("naive", {}).get("cache_hit_rate_proxy", "0"), 0.0)
    recompute_update = safe_float(policy.get("update_locality", {}).get("avg_recompute_ratio", ""), -1.0)
    hit_update = safe_float(policy.get("update_locality", {}).get("cache_hit_rate_proxy", ""), -1.0)
    recompute_periodic = safe_float(policy.get("periodic_refresh", {}).get("avg_recompute_ratio", ""), -1.0)
    hit_periodic = safe_float(policy.get("periodic_refresh", {}).get("cache_hit_rate_proxy", ""), -1.0)

    out_rows: List[Dict[str, str]] = []
    for scenario, rows_u, rows_p, src_u, src_p in [
        ("2048/128", rows_2048_u, rows_2048_p, p_2048_u.name, p_2048_p.name),
        ("4096/64", rows_4096_u, rows_4096_p, p_4096_u.name, p_4096_p.name),
    ]:
        add_row(
            out_rows,
            scenario=scenario,
            strategy="naive",
            row=find_row(rows_u, "naive"),
            source=src_u,
            recompute_ratio=recompute_naive,
            cache_hit_rate=hit_naive,
            quality_points=quality_points,
            quality_threshold=args.quality_threshold,
            with_quality=False,
        )
        add_row(
            out_rows,
            scenario=scenario,
            strategy="prefix_only",
            row=find_row(rows_u, "prefix_only"),
            source=src_u,
            recompute_ratio=-1.0,
            cache_hit_rate=-1.0,
            quality_points=quality_points,
            quality_threshold=args.quality_threshold,
            with_quality=False,
        )
        add_row(
            out_rows,
            scenario=scenario,
            strategy="update_locality",
            row=find_row(rows_u, "update_locality"),
            source=src_u,
            recompute_ratio=recompute_update,
            cache_hit_rate=hit_update,
            quality_points=quality_points,
            quality_threshold=args.quality_threshold,
        )
        add_row(
            out_rows,
            scenario=scenario,
            strategy="periodic_refresh",
            row=find_row(rows_p, "update_locality"),
            source=src_p,
            recompute_ratio=recompute_periodic,
            cache_hit_rate=hit_periodic,
            quality_points=quality_points,
            quality_threshold=args.quality_threshold,
        )
        add_row(
            out_rows,
            scenario=scenario,
            strategy="hybrid",
            row=find_row(rows_u, "hybrid_update_locality"),
            source=src_u,
            recompute_ratio=recompute_update,
            cache_hit_rate=hit_update,
            quality_points=quality_points,
            quality_threshold=args.quality_threshold,
        )

    order = {"naive": 0, "prefix_only": 1, "update_locality": 2, "periodic_refresh": 3, "hybrid": 4}
    out_rows.sort(key=lambda r: (r["scenario"], order.get(r["strategy"], 99)))

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (Path.cwd() / "reports" / f"stage43_strategy_table_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "stage43_strategy_table.csv"
    out_md = out_dir / "stage43_strategy_table.md"
    out_p1 = out_dir / "stage43_p1_input.md"
    write_csv(out_csv, out_rows)

    md_lines = [
        "# Stage 4.3 Strategy Table (Simulation Aggregate)",
        "",
        f"- Generated at: `{dt.datetime.now().isoformat(timespec='seconds')}`",
        f"- Quality threshold (proxy): `{args.quality_threshold:.6f}`",
        "",
        "| scenario | strategy | cumulative_gpu_hours | reduction_vs_naive_% | avg_recompute_ratio | cache_hit_rate_proxy | quality_proxy_delta_max | quality_ok |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in out_rows:
        md_lines.append(
            "| "
            + " | ".join(
                [
                    r["scenario"],
                    r["strategy"],
                    r["cumulative_gpu_hours"],
                    r["reduction_vs_naive_pct"],
                    r["avg_recompute_ratio"],
                    r["cache_hit_rate_proxy"],
                    r["quality_proxy_delta_max"],
                    r["quality_ok"],
                ]
            )
            + " |"
        )
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    p1_lines = [f"# Stage4.3 P1 Input ({dt.date.today().isoformat()})", ""]
    for scenario in ["2048/128", "4096/64"]:
        rows_s = [r for r in out_rows if r["scenario"] == scenario]
        feasible = [r for r in rows_s if r.get("quality_ok", "") in ("", "1")]
        if not feasible:
            feasible = rows_s
        best = min(feasible, key=lambda r: safe_float(r.get("cumulative_gpu_hours", "1e18"), 1e18))
        p1_lines.append(f"## {scenario}")
        p1_lines.append(
            f"- Best strategy (under quality constraint): {best['strategy']}, "
            f"gpu_hours={best['cumulative_gpu_hours']}, reduction_vs_naive={best['reduction_vs_naive_pct']}%"
        )
    out_p1.write_text("\n".join(p1_lines) + "\n", encoding="utf-8")

    print("[done] stage4.3 strategy table")
    print(f"- csv: {out_csv}")
    print(f"- md: {out_md}")
    print(f"- p1 input: {out_p1}")


if __name__ == "__main__":
    main()

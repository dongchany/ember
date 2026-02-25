#!/usr/bin/env python3
import argparse
import datetime as dt
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from common_report import die, read_csv, run_cmd, safe_float, write_csv


def hf_hub_root() -> Path:
    hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE", "").strip()
    if hub_cache:
        return Path(hub_cache).expanduser().resolve()
    hf_home = os.environ.get("HF_HOME", "").strip()
    if hf_home:
        return (Path(hf_home).expanduser().resolve() / "hub")
    return (Path.home() / ".cache" / "huggingface" / "hub").resolve()


def resolve_snapshot_dir(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    if (path / "config.json").exists() and list(path.glob("*.safetensors")):
        return path
    snap_root = path / "snapshots"
    if not snap_root.exists():
        return None
    cands = [p for p in snap_root.iterdir() if p.is_dir() and (p / "config.json").exists() and list(p.glob("*.safetensors"))]
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def resolve_model_dir(model_arg: str) -> Path:
    raw = model_arg.strip()
    if not raw:
        die("--model is empty")
    p = Path(raw).expanduser().resolve()
    resolved = resolve_snapshot_dir(p)
    if resolved is not None:
        return resolved
    hub_root = hf_hub_root()
    model_cache_dir = hub_root / ("models--" + raw.replace("/", "--"))
    resolved = resolve_snapshot_dir(model_cache_dir)
    if resolved is not None:
        return resolved
    die(
        "failed to resolve model from local cache: "
        f"{raw}. Checked path='{p}' and HF cache='{model_cache_dir}'."
    )
    raise AssertionError("unreachable")


def resolve_adapter_dir(adapter_arg: str) -> Path:
    p = Path(adapter_arg).expanduser().resolve()
    if p.is_file():
        p = p.parent
    if not p.exists():
        die(f"adapter path not found: {p}")
    if not list(p.glob("*.safetensors")):
        die(f"no .safetensors in adapter dir: {p}")
    return p


def size_of_safetensors_bytes(dir_path: Path) -> int:
    total = 0
    for p in dir_path.glob("*.safetensors"):
        if p.is_file():
            total += p.stat().st_size
    return total


def sync_ms_from_bytes(num_bytes: int, bandwidth_gib_per_s: float) -> float:
    return float(num_bytes) / (bandwidth_gib_per_s * (1024.0 ** 3)) * 1000.0


def one_run(
    *,
    bench_bin: Path,
    repo: Path,
    logs_dir: Path,
    mode_name: str,
    model_dir: Path,
    adapter_dir: Path,
    gpus: str,
    split: str,
    prompt_len: int,
    gen_len: int,
    num_candidates: int,
    rounds: int,
    warmup: int,
    chunk_len: int,
    overlap: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    scale: float,
    replace_existing: bool,
    update_mode: str,
    simulate_sync_ms: float,
    out_dir: Path,
) -> Tuple[Path, Path]:
    summary_csv = out_dir / f"stage53_{mode_name}.csv"
    per_round_csv = out_dir / f"stage53_{mode_name}_per_round.csv"
    cmd = [
        str(bench_bin),
        "--model",
        str(model_dir),
        "--adapter",
        str(adapter_dir),
        "--update-mode",
        update_mode,
        "--simulate-sync-ms",
        f"{simulate_sync_ms:.8f}",
        "--gpus",
        gpus,
        "--split",
        split,
        "--prompt-len",
        str(prompt_len),
        "--gen-len",
        str(gen_len),
        "--num-candidates",
        str(num_candidates),
        "--rounds",
        str(rounds),
        "--warmup",
        str(warmup),
        "--chunk-len",
        str(chunk_len),
        "--temperature",
        str(temperature),
        "--top-p",
        str(top_p),
        "--top-k",
        str(top_k),
        "--seed",
        str(seed),
        "--scale",
        str(scale),
        "--csv",
        str(summary_csv),
        "--per-round-csv",
        str(per_round_csv),
    ]
    if overlap:
        cmd.append("--overlap")
    else:
        cmd.append("--no-overlap")
    if not replace_existing:
        cmd.append("--no-replace-existing")
    p = run_cmd(cmd, cwd=repo, log_path=logs_dir / f"{mode_name}.log", check=False)
    if p.returncode != 0:
        die(f"benchmark failed for mode={mode_name}; see {logs_dir / (mode_name + '.log')}")
    if not summary_csv.exists():
        die(f"missing output csv for mode={mode_name}: {summary_csv}")
    return summary_csv, per_round_csv


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 5.3 measured e2e rollout+update loop comparison.")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--adapter", type=str, required=True)
    ap.add_argument("--gpus", type=str, default="0,1")
    ap.add_argument("--split", type=str, default="18,18")
    ap.add_argument("--prompt-len", type=int, default=1024)
    ap.add_argument("--gen-len", type=int, default=128)
    ap.add_argument("--num-candidates", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--chunk-len", type=int, default=512)
    ap.add_argument("--overlap", action="store_true", default=True)
    ap.add_argument("--no-overlap", dest="overlap", action="store_false")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--replace-existing", action="store_true", default=True)
    ap.add_argument("--no-replace-existing", dest="replace_existing", action="store_false")
    ap.add_argument("--sync-bandwidth-gib-per-s", type=float, default=24.0)
    ap.add_argument("--build-dir", type=str, default="build")
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    if args.prompt_len <= 0 or args.gen_len <= 0 or args.num_candidates <= 0:
        die("prompt/gen/candidates must be > 0")
    if args.rounds <= 0 or args.warmup < 0:
        die("--rounds must be > 0 and --warmup >= 0")
    if args.sync_bandwidth_gib_per_s <= 0:
        die("--sync-bandwidth-gib-per-s must be > 0")

    repo = Path.cwd()
    model_dir = resolve_model_dir(args.model)
    adapter_dir = resolve_adapter_dir(args.adapter)
    bench_bin = (repo / args.build_dir / "ember_rollout_update_loop_benchmark").resolve()
    if not bench_bin.exists():
        die(f"missing benchmark binary: {bench_bin}")

    model_bytes = size_of_safetensors_bytes(model_dir)
    adapter_bytes = size_of_safetensors_bytes(adapter_dir)
    if model_bytes <= 0:
        die(f"no model safetensors under {model_dir}")
    if adapter_bytes <= 0:
        die(f"no adapter safetensors under {adapter_dir}")
    full_sync_ms = sync_ms_from_bytes(model_bytes, args.sync_bandwidth_gib_per_s)
    lora_sync_ms = sync_ms_from_bytes(adapter_bytes, args.sync_bandwidth_gib_per_s)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (
        repo / "reports" / f"stage53_e2e_loop_compare_{ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        ("unified_apply", "apply", 0.0),
        ("dual_fullsync_sim", "skip", full_sync_ms),
        ("dual_lora_sync_sim", "skip", lora_sync_ms),
    ]

    rows: List[Dict[str, str]] = []
    for idx, (name, update_mode, sync_ms) in enumerate(runs):
        summary_csv, _ = one_run(
            bench_bin=bench_bin,
            repo=repo,
            logs_dir=logs_dir,
            mode_name=name,
            model_dir=model_dir,
            adapter_dir=adapter_dir,
            gpus=args.gpus,
            split=args.split,
            prompt_len=args.prompt_len,
            gen_len=args.gen_len,
            num_candidates=args.num_candidates,
            rounds=args.rounds,
            warmup=args.warmup,
            chunk_len=args.chunk_len,
            overlap=args.overlap,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed + idx,
            scale=args.scale,
            replace_existing=args.replace_existing,
            update_mode=update_mode,
            simulate_sync_ms=sync_ms,
            out_dir=out_dir,
        )
        r = read_csv(summary_csv)
        if not r:
            die(f"empty summary csv: {summary_csv}")
        row = dict(r[0])
        row["scenario"] = name
        row["sync_ms_assumed"] = f"{sync_ms:.6f}"
        rows.append(row)

    out_csv = out_dir / "stage53_e2e_compare.csv"
    out_md = out_dir / "stage53_e2e_compare.md"

    # normalized summary table
    summary_rows: List[Dict[str, str]] = []
    for r in rows:
        summary_rows.append(
            {
                "scenario": r.get("scenario", ""),
                "update_mode": r.get("update_mode", ""),
                "simulate_sync_ms": r.get("simulate_sync_ms", ""),
                "update_ms_ext_avg": r.get("update_ms_ext_avg", ""),
                "rollout_ms_avg": r.get("rollout_ms_avg", ""),
                "round_ms_avg": r.get("round_ms_avg", ""),
                "rollout_tok_s": r.get("rollout_tok_s", ""),
                "e2e_tok_s": r.get("e2e_tok_s", ""),
                "tokens_per_round": r.get("tokens_per_round", ""),
                "total_tokens_measured": r.get("total_tokens_measured", ""),
            }
        )
    write_csv(out_csv, summary_rows)

    by_name = {r["scenario"]: r for r in summary_rows}
    u = by_name.get("unified_apply")
    f = by_name.get("dual_fullsync_sim")
    l = by_name.get("dual_lora_sync_sim")
    speedup_vs_full = 0.0
    speedup_vs_lora = 0.0
    if u and f:
        f_ms = safe_float(f.get("round_ms_avg", "0"))
        u_ms = safe_float(u.get("round_ms_avg", "0"))
        speedup_vs_full = (f_ms / u_ms) if u_ms > 0 else 0.0
    if u and l:
        l_ms = safe_float(l.get("round_ms_avg", "0"))
        u_ms = safe_float(u.get("round_ms_avg", "0"))
        speedup_vs_lora = (l_ms / u_ms) if u_ms > 0 else 0.0

    lines: List[str] = []
    lines.append("# Stage 5.3 Measured E2E Loop Compare")
    lines.append("")
    lines.append(f"- Generated at: `{dt.datetime.now().isoformat(timespec='seconds')}`")
    lines.append(f"- Model: `{model_dir}`")
    lines.append(f"- Adapter: `{adapter_dir}`")
    lines.append(f"- rounds={args.rounds}, warmup={args.warmup}, prompt_len={args.prompt_len}, gen_len={args.gen_len}, num_candidates={args.num_candidates}")
    lines.append(f"- sync bandwidth assumption: `{args.sync_bandwidth_gib_per_s:.3f} GiB/s`")
    lines.append(f"- full_sync_ms_est=`{full_sync_ms:.6f}`, lora_sync_ms_est=`{lora_sync_ms:.6f}`")
    lines.append("")
    lines.append("| scenario | update_mode | simulate_sync_ms | update_ms_ext_avg | rollout_ms_avg | round_ms_avg | rollout_tok_s | e2e_tok_s |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in summary_rows:
        lines.append(
            f"| {r['scenario']} | {r['update_mode']} | {r['simulate_sync_ms']} | "
            f"{r['update_ms_ext_avg']} | {r['rollout_ms_avg']} | {r['round_ms_avg']} | "
            f"{r['rollout_tok_s']} | {r['e2e_tok_s']} |"
        )
    lines.append("")
    lines.append("## Key Point")
    lines.append(f"- Unified vs dual_fullsync(sim): speedup `{speedup_vs_full:.6f}x` (round_ms ratio).")
    lines.append(f"- Unified vs dual_lora_sync(sim): speedup `{speedup_vs_lora:.6f}x` (round_ms ratio).")
    lines.append("")
    lines.append("## Notes")
    lines.append("- `unified_apply` is measured in-process `apply_lora_adapter + rollout`.")
    lines.append("- dual-stack rows are simulated by adding per-round sync sleep with measured rollout path.")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[done] out_dir={out_dir}")
    print(f"- compare_csv: {out_csv}")
    print(f"- compare_md: {out_md}")


if __name__ == "__main__":
    main()

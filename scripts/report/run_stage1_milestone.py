#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from common_report import die, read_csv, safe_float, split_ints, write_csv


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
    candidates = [
        p for p in snap_root.iterdir()
        if p.is_dir() and (p / "config.json").exists() and list(p.glob("*.safetensors"))
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def resolve_model_dir(model_arg: str) -> Path:
    raw = model_arg.strip()
    if not raw:
        die("--model is empty")

    p = Path(raw).expanduser().resolve()
    resolved = resolve_snapshot_dir(p)
    if resolved is not None:
        return resolved

    # Treat as HF model id (e.g., Qwen/Qwen3-8B), resolve from local cache only.
    hub_root = hf_hub_root()
    model_cache_dir = hub_root / ("models--" + raw.replace("/", "--"))
    resolved = resolve_snapshot_dir(model_cache_dir)
    if resolved is not None:
        return resolved

    die(
        "failed to resolve model from local cache: "
        f"{raw}. Checked path='{p}' and HF cache='{model_cache_dir}'. "
        "Set HF_HOME/HUGGINGFACE_HUB_CACHE correctly, or pass a local snapshot path."
    )
    raise AssertionError("unreachable")


def run_cmd(cmd: List[str], cwd: Path, log_path: Path) -> Tuple[int, str, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        p = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
        f.write(p.stdout)
        if p.stderr:
            f.write("\n[stderr]\n")
            f.write(p.stderr)
    return p.returncode, p.stdout, p.stderr


def write_failures_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    fieldnames = [
        "run_id",
        "prompt_len",
        "decode_steps",
        "overlap",
        "chunk_len",
        "return_code",
        "error_hint",
        "log_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_int(v: str, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def looks_like_oom(text: str) -> bool:
    t = text.lower()
    keys = [
        "out of memory",
        "cuda_error_out_of_memory",
        "cuda error: out of memory",
        "cudaMalloc".lower(),
    ]
    return any(k in t for k in keys)


def looks_like_runtime_unavailable(text: str) -> bool:
    t = text.lower()
    keys = [
        "cuda runtime not available",
        "no cuda device available",
        "cuda_error_no_device",
        "cuda_error_system_not_ready",
        "cudasetdevice",
        "operation not supported on this os",
        "os call failed",
    ]
    return any(k in t for k in keys)


def detect_error_hint(text: str) -> str:
    if looks_like_oom(text):
        return "oom"
    if looks_like_runtime_unavailable(text):
        return "runtime_unavailable"
    return "runtime_error"


def summarize(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    grouped: Dict[Tuple[str, str, str, str, str, str], Dict[str, Dict[str, str]]] = {}
    for row in rows:
        key = (
            row.get("gpus", ""),
            row.get("split", ""),
            row.get("mode", ""),
            row.get("decode_sampling", ""),
            row.get("prompt_len", ""),
            row.get("decode_steps", ""),
        )
        grouped.setdefault(key, {})[row.get("phase", "")] = row

    out: List[Dict[str, str]] = []
    for key, phases in sorted(grouped.items()):
        pre = phases.get("prefill")
        dec = phases.get("decode_per_token")
        if pre is None:
            continue
        prompt_len = pre.get("prompt_len", "0")
        decode_steps = pre.get("decode_steps", "0")
        pre_ms = safe_float(pre.get("wall_ms", "0"))
        dec_tok_ms = safe_float(dec.get("wall_ms", "0")) if dec else 0.0
        total_ms = pre_ms + dec_tok_ms * safe_float(decode_steps, 0.0)
        prefill_share = (pre_ms / total_ms * 100.0) if total_ms > 0.0 else 0.0

        out.append(
            {
                "gpus": key[0],
                "split": key[1],
                "mode": pre.get("mode", ""),
                "decode_sampling": pre.get("decode_sampling", ""),
                "prompt_len": prompt_len,
                "decode_steps": decode_steps,
                "prefill_wall_ms": f"{pre_ms:.3f}",
                "decode_per_token_ms": f"{dec_tok_ms:.3f}",
                "rollout_total_ms_est": f"{total_ms:.3f}",
                "prefill_share_pct": f"{prefill_share:.2f}",
                "prefill_attention_ms": pre.get("attention_ms", "0"),
                "prefill_ffn_ms": pre.get("ffn_ms", "0"),
                "prefill_memcpy_h2d_ms": pre.get("memcpy_h2d_ms", "0"),
                "decode_memcpy_d2h_ms": (dec or {}).get("memcpy_d2h_ms", "0"),
                "decode_sampling_ms": (dec or {}).get("sampling_ms", "0"),
            }
        )
    return out


def write_markdown(path: Path, model: str, rows: List[Dict[str, str]]) -> None:
    lines: List[str] = []
    lines.append("# Stage 1.1 Milestone Summary")
    lines.append("")
    lines.append(f"- Model: `{model}`")
    lines.append(f"- Generated at: `{dt.datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")
    if not rows:
        lines.append("_No data rows were generated._")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.append("| gpus | split | mode | prompt_len | decode_steps | prefill_ms | decode_per_token_ms | prefill_share_% |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    r["gpus"],
                    r["split"],
                    r["mode"],
                    r["prompt_len"],
                    r["decode_steps"],
                    r["prefill_wall_ms"],
                    r["decode_per_token_ms"],
                    r["prefill_share_pct"],
                ]
            )
            + " |"
        )

    best = max(rows, key=lambda x: safe_float(x.get("prefill_share_pct", "0")))
    lines.append("")
    lines.append("## Key Point")
    lines.append(
        f"- Highest prefill share: `{best['prefill_share_pct']}%` "
        f"(prompt_len={best['prompt_len']}, decode_steps={best['decode_steps']}, gpus={best['gpus']}, mode={best['mode']})."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_mainline_ready_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    out: List[Dict[str, str]] = []
    for r in rows:
        row = dict(r)
        row["coverage_note"] = "ok"
        out.append(row)
    out.sort(
        key=lambda x: (
            x.get("gpus", ""),
            x.get("split", ""),
            x.get("mode", ""),
            safe_int(x.get("decode_steps", "0")),
            safe_int(x.get("prompt_len", "0")),
        )
    )
    write_csv(path, out)


def write_mainline_ready_md(path: Path, usable_rows: List[Dict[str, str]], failed_rows: List[Dict[str, str]]) -> None:
    oom_rows = [r for r in failed_rows if r.get("error_hint", "") == "oom"]
    runtime_rows = [r for r in failed_rows if r.get("error_hint", "") == "runtime_unavailable"]
    missing = sorted(
        failed_rows,
        key=lambda x: (
            safe_int(x.get("prompt_len", "0")),
            safe_int(x.get("decode_steps", "0")),
            x.get("overlap", ""),
            safe_int(x.get("chunk_len", "0")),
        ),
    )
    lines: List[str] = []
    lines.append("# Stage 1.1 Mainline Readiness")
    lines.append("")
    lines.append(f"- usable summary rows: {len(usable_rows)}")
    lines.append(f"- failed combos: {len(failed_rows)}")
    lines.append(f"- oom combos: {len(oom_rows)}")
    lines.append("")
    lines.append("## Missing Combos")
    if not missing:
        lines.append("- none")
    else:
        for r in missing:
            lines.append(
                "- "
                f"prompt_len={r.get('prompt_len','')}, "
                f"decode_steps={r.get('decode_steps','')}, "
                f"overlap={r.get('overlap','')}, "
                f"chunk_len={r.get('chunk_len','')}, "
                f"hint={r.get('error_hint','')}"
            )
    lines.append("")
    lines.append("## Recommendation")
    lines.append("- Use current 8B results for Stage 1.1 mainline plots/tables.")
    if oom_rows:
        lines.append("- Mark OOM combinations as boundary points in analysis.")
    if runtime_rows:
        lines.append("- Current rerun hit transient CUDA runtime availability errors on some missing combinations.")
    if missing:
        lines.append("- For strict full-grid coverage, rerun missing combinations after CUDA environment is stable.")
    else:
        lines.append("- Full grid coverage achieved for current Stage 1.1 matrix.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_prefill_share_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    out: List[Dict[str, str]] = []
    for r in rows:
        out.append(
            {
                "prompt_len": r.get("prompt_len", ""),
                "decode_steps": r.get("decode_steps", ""),
                "mode": r.get("mode", ""),
                "prefill_share_pct": r.get("prefill_share_pct", ""),
                "prefill_wall_ms": r.get("prefill_wall_ms", ""),
                "decode_per_token_ms": r.get("decode_per_token_ms", ""),
            }
        )
    out.sort(
        key=lambda x: (
            x.get("mode", ""),
            safe_int(x.get("decode_steps", "0")),
            safe_int(x.get("prompt_len", "0")),
        )
    )
    write_csv(path, out)


def write_stage_latency_components_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    phase_order = {"prefill": 0, "decode_per_token": 1}
    out: List[Dict[str, str]] = []
    for r in rows:
        out.append(
            {
                "run_id": r.get("run_id", ""),
                "phase": r.get("phase", ""),
                "mode": r.get("mode", ""),
                "prompt_len": r.get("prompt_len", ""),
                "decode_steps": r.get("decode_steps", ""),
                "wall_ms": r.get("wall_ms", ""),
                "embedding_ms": r.get("embedding_ms", ""),
                "rmsnorm_ms": r.get("rmsnorm_ms", ""),
                "attention_ms": r.get("attention_ms", ""),
                "ffn_ms": r.get("ffn_ms", ""),
                "p2p_ms": r.get("p2p_ms", ""),
                "memcpy_h2d_ms": r.get("memcpy_h2d_ms", ""),
                "memcpy_d2h_ms": r.get("memcpy_d2h_ms", ""),
                "sampling_ms": r.get("sampling_ms", ""),
                "lm_head_ms": r.get("lm_head_ms", ""),
                "profile_total_ms": r.get("profile_total_ms", ""),
            }
        )
    out.sort(
        key=lambda x: (
            safe_int(x.get("run_id", "0")),
            phase_order.get(x.get("phase", ""), 99),
        )
    )
    write_csv(path, out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Stage 1.1 profiling milestone in one command.")
    ap.add_argument(
        "--model",
        type=str,
        default=os.environ.get("MODEL_PATH", ""),
        help="model path or HF model id (default: MODEL_PATH)",
    )
    ap.add_argument("--gpus", type=str, default="0", help="GPU ids, e.g. 0 or 0,1")
    ap.add_argument("--split", type=str, default="", help="2-GPU layer split A,B")
    ap.add_argument("--prompt-lens", type=str, default="512,1024,2048,4096")
    ap.add_argument("--decode-steps", type=str, default="64,128,256")
    ap.add_argument("--chunk-len", type=int, default=512)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--decode-with-sampling", action="store_true", default=True)
    ap.add_argument("--decode-no-sampling", dest="decode_with_sampling", action="store_false")
    ap.add_argument("--retry-oom", action="store_true", default=True, help="retry a failed run with smaller chunk_len when OOM")
    ap.add_argument("--no-retry-oom", dest="retry_oom", action="store_false")
    ap.add_argument("--min-chunk-len", type=int, default=64, help="minimum chunk_len when retrying OOM")
    ap.add_argument("--continue-on-error", action="store_true", default=True, help="continue matrix on non-OOM/non-retriable failures")
    ap.add_argument("--stop-on-error", dest="continue_on_error", action="store_false")
    ap.add_argument("--retry-runtime-unavailable", type=int, default=2, help="retry count when CUDA runtime is temporarily unavailable")
    ap.add_argument(
        "--pipeline-2gpu",
        action="store_true",
        default=True,
        help="force 2-GPU prefill to use chunked pipeline path for both overlap=0/1 (default: on)",
    )
    ap.add_argument("--no-pipeline-2gpu", dest="pipeline_2gpu", action="store_false")
    ap.add_argument("--skip-existing", action="store_true", default=True, help="reuse non-empty existing stage_raw_*.csv if present")
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--build-dir", type=str, default="build")
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    if not args.model:
        die("--model is required (or set MODEL_PATH)")
    model_dir = resolve_model_dir(args.model)

    prompt_lens = split_ints(args.prompt_lens)
    decode_steps_list = split_ints(args.decode_steps)
    if not prompt_lens:
        die("--prompt-lens is empty")
    if not decode_steps_list:
        die("--decode-steps is empty")
    if args.min_chunk_len <= 0:
        die("--min-chunk-len must be > 0")
    if args.retry_runtime_unavailable < 0:
        die("--retry-runtime-unavailable must be >= 0")

    repo = Path.cwd()
    build_dir = (repo / args.build_dir).resolve()
    bin_stage = build_dir / "ember_stage_breakdown"
    if not bin_stage.exists():
        die(f"binary not found: {bin_stage} (build first)")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (repo / "reports" / f"stage1_milestone_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if not gpus:
        die("--gpus is empty")
    gpus_str = ",".join(gpus)
    overlap_modes = [0]
    if len(gpus) == 2:
        overlap_modes = [0, 1]

    all_rows: List[Dict[str, str]] = []
    failed_by_id: Dict[str, Dict[str, str]] = {}
    failures_csv = out_dir / "stage1_failures.csv"
    if failures_csv.exists() and failures_csv.stat().st_size > 0:
        for row in read_csv(failures_csv):
            rid = row.get("run_id", "")
            if rid:
                failed_by_id[rid] = row
    run_idx = 0

    for pl in prompt_lens:
        for ds in decode_steps_list:
            for ov in overlap_modes:
                run_idx += 1
                run_csv = out_dir / f"stage_raw_{run_idx:03d}.csv"
                if args.skip_existing and run_csv.exists() and run_csv.stat().st_size > 0:
                    print(f"[skip {run_idx}] reuse existing {run_csv.name}")
                    rows = read_csv(run_csv)
                    for row in rows:
                        row["model_dir"] = str(model_dir)
                        row["run_id"] = str(run_idx)
                    all_rows.extend(rows)
                    failed_by_id.pop(str(run_idx), None)
                    continue
                failed_by_id.pop(str(run_idx), None)

                cmd = [
                    str(bin_stage),
                    "--model",
                    str(model_dir),
                    "--gpus",
                    gpus_str,
                    "--prompt-len",
                    str(pl),
                    "--decode-steps",
                    str(ds),
                    "--iters",
                    str(args.iters),
                    "--warmup",
                    str(args.warmup),
                    "--csv",
                    str(run_csv),
                ]
                if args.split:
                    cmd += ["--split", args.split]
                if len(gpus) == 2 and args.pipeline_2gpu:
                    cmd += ["--pipeline"]
                if ov == 1:
                    cmd += ["--overlap"]
                else:
                    cmd += ["--no-overlap"]
                if args.decode_with_sampling:
                    cmd += ["--decode-with-sampling"]
                else:
                    cmd += ["--decode-no-sampling"]

                chunk_len = args.chunk_len
                attempt = 0
                success = False
                saw_oom = False
                saw_runtime_unavailable = False
                min_oom_chunk: Optional[int] = None
                while True:
                    attempt += 1
                    run_log = logs_dir / f"run_{run_idx:03d}_try{attempt}.log"
                    run_cmdline = list(cmd) + ["--chunk-len", str(chunk_len)]
                    print(f"[run {run_idx}] pl={pl} ds={ds} overlap={ov} chunk={chunk_len} try={attempt}")
                    rc, out, err = run_cmd(run_cmdline, cwd=repo, log_path=run_log)
                    if rc == 0:
                        success = True
                        break
                    merged = (out or "") + "\n" + (err or "")
                    attempt_hint = detect_error_hint(merged)
                    if attempt_hint == "oom":
                        saw_oom = True
                        min_oom_chunk = chunk_len if min_oom_chunk is None else min(min_oom_chunk, chunk_len)
                    elif attempt_hint == "runtime_unavailable":
                        saw_runtime_unavailable = True

                    if attempt_hint == "runtime_unavailable" and attempt <= args.retry_runtime_unavailable:
                        print(f"[retry {run_idx}] CUDA runtime unavailable, retrying in 2s (try={attempt})")
                        time.sleep(2.0)
                        continue
                    if args.retry_oom and attempt_hint == "oom" and chunk_len > args.min_chunk_len:
                        next_chunk = max(args.min_chunk_len, chunk_len // 2)
                        if next_chunk == chunk_len:
                            break
                        print(f"[retry {run_idx}] OOM detected, chunk_len {chunk_len} -> {next_chunk}")
                        chunk_len = next_chunk
                        continue
                    final_hint = attempt_hint
                    final_chunk = chunk_len
                    if saw_oom:
                        final_hint = "oom"
                        if min_oom_chunk is not None:
                            final_chunk = min_oom_chunk
                    elif saw_runtime_unavailable:
                        final_hint = "runtime_unavailable"
                    cur_failed = {
                        "run_id": str(run_idx),
                        "prompt_len": str(pl),
                        "decode_steps": str(ds),
                        "overlap": str(ov),
                        "chunk_len": str(final_chunk),
                        "return_code": str(rc),
                        "error_hint": final_hint,
                        "log_path": str(run_log),
                    }
                    failed_by_id[str(run_idx)] = cur_failed
                    if not args.continue_on_error:
                        die(
                            "run failed: "
                            + " ".join(run_cmdline)
                            + f" (log: {run_log})"
                        )
                    print(f"[warn] run {run_idx} failed; continue-on-error enabled")
                    break

                if success:
                    rows = read_csv(run_csv)
                    for row in rows:
                        row["model_dir"] = str(model_dir)
                        row["run_id"] = str(run_idx)
                    all_rows.extend(rows)
                    failed_by_id.pop(str(run_idx), None)

    failed_rows: List[Dict[str, str]] = sorted(
        failed_by_id.values(),
        key=lambda r: safe_int(r.get("run_id", "0")),
    )

    raw_csv = out_dir / "stage1_raw_rows.csv"
    write_csv(raw_csv, all_rows)

    summary_rows = summarize(all_rows)
    summary_csv = out_dir / "stage1_summary.csv"
    write_csv(summary_csv, summary_rows)

    summary_md = out_dir / "stage1_summary.md"
    write_markdown(summary_md, str(model_dir), summary_rows)

    write_failures_csv(failures_csv, failed_rows)

    mainline_csv = out_dir / "stage1_mainline_ready.csv"
    write_mainline_ready_csv(mainline_csv, summary_rows)

    mainline_md = out_dir / "stage1_mainline_ready.md"
    write_mainline_ready_md(mainline_md, summary_rows, failed_rows)

    p1_csv = out_dir / "p1_fig2_prefill_share.csv"
    write_prefill_share_csv(p1_csv, summary_rows)

    p2_csv = out_dir / "p2_stage_latency_components.csv"
    write_stage_latency_components_csv(p2_csv, all_rows)

    print("")
    print("[done] stage1 milestone completed")
    print(f"- raw rows: {raw_csv}")
    print(f"- summary csv: {summary_csv}")
    print(f"- summary md: {summary_md}")
    print(f"- mainline csv: {mainline_csv}")
    print(f"- mainline md: {mainline_md}")
    print(f"- p1 fig2 csv: {p1_csv}")
    print(f"- p2 stage csv: {p2_csv}")
    if failed_rows:
        print(f"- failures: {failures_csv} ({len(failed_rows)} failed runs)")
    print(f"- logs: {logs_dir}")


if __name__ == "__main__":
    main()

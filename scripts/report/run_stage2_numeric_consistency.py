#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


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
        p
        for p in snap_root.iterdir()
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


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            out.append(json.loads(ln))
    return out


def to_int_list(v: object) -> List[int]:
    if not isinstance(v, list):
        return []
    out: List[int] = []
    for x in v:
        out.append(int(x))
    return out


def to_float_list(v: object) -> List[float]:
    if not isinstance(v, list):
        return []
    out: List[float] = []
    for x in v:
        out.append(float(x))
    return out


def compare_candidates(
    a: List[Dict[str, object]],
    b: List[Dict[str, object]],
    tol: float,
) -> Tuple[bool, Dict[str, str]]:
    if len(a) != len(b):
        return False, {
            "num_candidates_a": str(len(a)),
            "num_candidates_b": str(len(b)),
            "token_mismatch_candidates": "0",
            "finish_reason_mismatch_candidates": "0",
            "max_abs_logprob_diff": "nan",
            "mean_abs_logprob_diff": "nan",
        }

    token_mismatch = 0
    finish_mismatch = 0
    total_abs = 0.0
    count_abs = 0
    max_abs = 0.0

    for ca, cb in zip(a, b):
        ta = to_int_list(ca.get("tokens", []))
        tb = to_int_list(cb.get("tokens", []))
        if ta != tb:
            token_mismatch += 1

        fa = str(ca.get("finish_reason", ""))
        fb = str(cb.get("finish_reason", ""))
        if fa != fb:
            finish_mismatch += 1

        la = to_float_list(ca.get("token_logprobs", []))
        lb = to_float_list(cb.get("token_logprobs", []))
        n = min(len(la), len(lb))
        for i in range(n):
            d = abs(la[i] - lb[i])
            total_abs += d
            count_abs += 1
            if d > max_abs:
                max_abs = d
        if len(la) != len(lb):
            token_mismatch += 1

    mean_abs = (total_abs / float(count_abs)) if count_abs > 0 else 0.0
    ok = (token_mismatch == 0 and finish_mismatch == 0 and max_abs <= tol)
    return ok, {
        "num_candidates_a": str(len(a)),
        "num_candidates_b": str(len(b)),
        "token_mismatch_candidates": str(token_mismatch),
        "finish_reason_mismatch_candidates": str(finish_mismatch),
        "max_abs_logprob_diff": f"{max_abs:.8f}",
        "mean_abs_logprob_diff": f"{mean_abs:.8f}",
    }


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 2.2 deterministic numeric consistency check.")
    ap.add_argument("--model", type=str, required=True, help="model path or HF model id")
    ap.add_argument("--prompt", type=str, default="")
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--gen-len", type=int, default=64)
    ap.add_argument("--num-candidates", type=int, default=4)
    ap.add_argument("--gpus", type=str, default="0,1")
    ap.add_argument("--split", type=str, default="9,27")
    ap.add_argument("--chunk-len", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--build-dir", type=str, default="build")
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    if args.prompt_len <= 0 or args.gen_len <= 0 or args.num_candidates <= 0:
        die("prompt/gen/candidates must be > 0")
    if args.tol < 0.0:
        die("--tol must be >= 0")

    repo = Path.cwd()
    model_dir = resolve_model_dir(args.model)
    bench_bin = (repo / args.build_dir / "ember_multi_candidate_rollout").resolve()
    if not bench_bin.exists():
        die(f"missing binary: {bench_bin}")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (repo / "reports" / f"stage22_numeric_consistency_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    run_paths = {
        "a": out_dir / "run_a_candidates.jsonl",
        "b": out_dir / "run_b_candidates.jsonl",
        "c": out_dir / "run_c_seed_plus_1_candidates.jsonl",
    }

    def run_once(tag: str, seed: int) -> None:
        cmd = [
            str(bench_bin),
            "--model",
            str(model_dir),
            "--prompt-len",
            str(args.prompt_len),
            "--gen-len",
            str(args.gen_len),
            "--num-candidates",
            str(args.num_candidates),
            "--gpus",
            args.gpus,
            "--split",
            args.split,
            "--chunk-len",
            str(args.chunk_len),
            "--temperature",
            str(args.temperature),
            "--top-p",
            str(args.top_p),
            "--top-k",
            str(args.top_k),
            "--seed",
            str(seed),
            "--candidates-jsonl",
            str(run_paths[tag]),
            "--no-decode-text",
        ]
        if args.prompt:
            cmd += ["--prompt", args.prompt]
        p = run_cmd(cmd, cwd=repo, log_path=logs_dir / f"run_{tag}.log")
        if p.returncode != 0:
            die(f"benchmark run_{tag} failed rc={p.returncode}; see {logs_dir / ('run_' + tag + '.log')}")

    run_once("a", args.seed)
    run_once("b", args.seed)
    run_once("c", args.seed + 1)

    a = load_jsonl(run_paths["a"])
    b = load_jsonl(run_paths["b"])
    c = load_jsonl(run_paths["c"])

    same_ok, same_stats = compare_candidates(a, b, tol=args.tol)
    diff_ok, diff_stats = compare_candidates(a, c, tol=args.tol)
    # For seed sensitivity we expect "not same".
    seed_sensitive = not diff_ok

    summary_rows = [
        {
            "check": "same_seed_repro",
            "ok": "1" if same_ok else "0",
            **same_stats,
            "tol": f"{args.tol:.8f}",
            "seed_a": str(args.seed),
            "seed_b": str(args.seed),
        },
        {
            "check": "seed_plus_1_diff",
            "ok": "1" if seed_sensitive else "0",
            **diff_stats,
            "tol": f"{args.tol:.8f}",
            "seed_a": str(args.seed),
            "seed_b": str(args.seed + 1),
        },
    ]

    summary_csv = out_dir / "stage22_numeric_consistency.csv"
    summary_md = out_dir / "stage22_summary.md"
    write_csv(summary_csv, summary_rows)

    lines: List[str] = []
    lines.append("# Stage 2.2 Numeric Consistency")
    lines.append("")
    lines.append(f"- Model: `{model_dir}`")
    lines.append(
        f"- Setting: prompt_len={args.prompt_len}, gen_len={args.gen_len}, "
        f"num_candidates={args.num_candidates}, gpus={args.gpus}, split={args.split}"
    )
    lines.append(f"- Generated at: `{dt.datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")
    lines.append("| check | ok | token_mismatch_candidates | finish_reason_mismatch_candidates | max_abs_logprob_diff | mean_abs_logprob_diff |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in summary_rows:
        lines.append(
            f"| {r['check']} | {r['ok']} | {r['token_mismatch_candidates']} | "
            f"{r['finish_reason_mismatch_candidates']} | {r['max_abs_logprob_diff']} | {r['mean_abs_logprob_diff']} |"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[done] out_dir={out_dir}")
    print(f"- csv: {summary_csv}")
    print(f"- md: {summary_md}")
    print(f"- same_seed_ok={int(same_ok)} seed_plus_1_diff_ok={int(seed_sensitive)}")


if __name__ == "__main__":
    main()

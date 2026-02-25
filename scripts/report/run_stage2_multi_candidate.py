#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


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


def read_csv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def safe_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def load_candidates(path: Path) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            out.append(json.loads(ln))
    return out


def write_summary_md(path: Path, model_dir: Path, row: Dict[str, str], cand: List[Dict[str, object]]) -> None:
    lines: List[str] = []
    lines.append("# Stage 2.1 Multi-Candidate Rollout")
    lines.append("")
    lines.append(f"- Model: `{model_dir}`")
    lines.append(f"- Generated at: `{dt.datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    for key in [
        "prompt_len",
        "gen_len",
        "num_candidates",
        "num_stop_sequences",
        "gpus",
        "split",
        "prefill_ms",
        "decode_ms",
        "total_ms",
        "total_gen_tokens",
        "gen_tok_s",
        "temperature",
        "top_p",
        "top_k",
    ]:
        lines.append(f"| {key} | `{row.get(key, '')}` |")

    if cand:
        avg_lp = sum(float(x.get("avg_logprob", 0.0)) for x in cand) / float(len(cand))
        best = max(cand, key=lambda x: float(x.get("sum_logprob", -1e9)))
        lines.append("")
        lines.append("## Candidate Stats")
        lines.append(f"- mean(avg_logprob): `{avg_lp:.6f}`")
        lines.append(f"- best candidate id: `{best.get('candidate_id')}`")
        lines.append(f"- best sum_logprob: `{float(best.get('sum_logprob', 0.0)):.6f}`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_p1_input(path: Path, row: Dict[str, str], cand: List[Dict[str, object]]) -> None:
    lines: List[str] = []
    lines.append("# P1/P4 Input — Multi-Candidate Rollout")
    lines.append("")
    lines.append(
        f"- Rollout throughput: `{row.get('gen_tok_s','')}` tok/s "
        f"(total_gen_tokens={row.get('total_gen_tokens','')}, total_ms={row.get('total_ms','')})."
    )
    lines.append(
        f"- Multi-candidate setting: num_candidates=`{row.get('num_candidates','')}`, "
        f"gen_len=`{row.get('gen_len','')}`."
    )
    if cand:
        avg_lp = sum(float(x.get("avg_logprob", 0.0)) for x in cand) / float(len(cand))
        lines.append(f"- Mean per-candidate avg_logprob: `{avg_lp:.6f}`.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Stage 2.1 multi-candidate rollout benchmark.")
    ap.add_argument("--model", type=str, required=True, help="model path or HF model id")
    ap.add_argument("--prompt", type=str, default="")
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--gen-len", type=int, default=128)
    ap.add_argument("--num-candidates", type=int, default=8)
    ap.add_argument("--gpus", type=str, default="0,1")
    ap.add_argument("--split", type=str, default="9,27")
    ap.add_argument("--chunk-len", type=int, default=512)
    ap.add_argument("--overlap", action="store_true", default=True)
    ap.add_argument("--no-overlap", dest="overlap", action="store_false")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--stop-seqs", type=str, default="", help="stop sequences joined by ||, e.g. '<|im_end|>||###'")
    ap.add_argument("--strip-stop", action="store_true", default=True)
    ap.add_argument("--no-strip-stop", dest="strip_stop", action="store_false")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--decode-text", action="store_true", default=True)
    ap.add_argument("--no-decode-text", dest="decode_text", action="store_false")
    ap.add_argument("--build-dir", type=str, default="build")
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    if args.prompt_len <= 0 or args.gen_len <= 0 or args.num_candidates <= 0:
        die("prompt/gen/candidates must be > 0")

    model_dir = resolve_model_dir(args.model)
    repo = Path.cwd()
    bench_bin = (repo / args.build_dir / "ember_multi_candidate_rollout").resolve()
    if not bench_bin.exists():
        die(f"missing binary: {bench_bin}")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (repo / "reports" / f"stage21_multi_candidate_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "stage21_multi_candidate.csv"
    cand_path = out_dir / "stage21_candidates.jsonl"

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
        str(args.seed),
        "--csv",
        str(csv_path),
        "--candidates-jsonl",
        str(cand_path),
    ]
    if args.stop_seqs.strip():
        stop_list = [x for x in args.stop_seqs.split("||") if x]
        for s in stop_list:
            cmd += ["--stop-seq", s]
    if not args.strip_stop:
        cmd += ["--no-strip-stop"]
    if args.prompt:
        cmd += ["--prompt", args.prompt]
    if args.overlap:
        cmd += ["--overlap"]
    else:
        cmd += ["--no-overlap"]
    if not args.decode_text:
        cmd += ["--no-decode-text"]

    p = run_cmd(cmd, cwd=repo, log_path=logs_dir / "run.log")
    if p.returncode != 0:
        die(f"benchmark failed rc={p.returncode}; see {logs_dir / 'run.log'}")

    rows = read_csv(csv_path)
    if not rows:
        die(f"empty csv: {csv_path}")
    row = rows[0]
    cand = load_candidates(cand_path)

    write_summary_md(out_dir / "stage21_summary.md", model_dir, row, cand)
    write_p1_input(out_dir / "stage21_p1_input.md", row, cand)

    print(f"[done] out_dir={out_dir}")
    print(
        f"[result] candidates={row.get('num_candidates','')} "
        f"gen_tok_s={safe_float(row.get('gen_tok_s','0')):.3f} "
        f"total_ms={safe_float(row.get('total_ms','0')):.3f}"
    )


if __name__ == "__main__":
    main()

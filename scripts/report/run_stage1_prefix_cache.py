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


def split_ints(text: str) -> List[int]:
    out: List[int] = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


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


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_md(path: Path, model_dir: Path, rows: List[Dict[str, str]], prompt_len: int, num_docs: int) -> None:
    lines: List[str] = []
    lines.append("# Stage 1.3 Prefix Cache Sweep")
    lines.append("")
    lines.append(f"- Model: `{model_dir}`")
    lines.append(f"- Prompt length: `{prompt_len}`")
    lines.append(f"- Docs per run: `{num_docs}`")
    lines.append(f"- Generated at: `{dt.datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")
    lines.append("| prefix_len | suffix_len | no_cache_total_ms | with_cache_total_ms | speedup_x | savings_% | theoretical_savings_% |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    r.get("prefix_len", ""),
                    r.get("suffix_len", ""),
                    r.get("no_cache_total_ms", ""),
                    r.get("with_cache_total_ms", ""),
                    r.get("speedup_x", ""),
                    r.get("savings_pct", ""),
                    r.get("theoretical_savings_pct", ""),
                ]
            )
            + " |"
        )

    if rows:
        best = max(rows, key=lambda x: safe_float(x.get("savings_pct", "0")))
        mid = None
        for r in rows:
            if int(r.get("prefix_len", "0")) == 1000:
                mid = r
                break
        if mid is None:
            for r in rows:
                if int(r.get("prefix_len", "0")) == 1024:
                    mid = r
                    break
        lines.append("")
        lines.append("## Key Point")
        lines.append(
            f"- Best measured savings: prefix_len={best.get('prefix_len','')} "
            f"-> `{best.get('savings_pct','')}%` (`{best.get('speedup_x','')}x`)."
        )
        if mid is not None:
            lines.append(
                f"- Shared-prefix ~1k tokens result: savings `{mid.get('savings_pct','')}%`, "
                f"speedup `{mid.get('speedup_x','')}x`."
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Stage 1.3 prefix cache sweep.")
    ap.add_argument("--model", type=str, required=True, help="model path or HF model id")
    ap.add_argument("--gpus", type=str, default="0,1", help="GPU ids, e.g. 0,1")
    ap.add_argument("--split", type=str, default="9,27", help="2-GPU split A,B")
    ap.add_argument("--prompt-len", type=int, default=2048)
    ap.add_argument("--prefix-lens", type=str, default="0,256,512,768,1024,1280,1536")
    ap.add_argument("--num-docs", type=int, default=100)
    ap.add_argument("--chunk-len", type=int, default=512)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--overlap", action="store_true", default=True)
    ap.add_argument("--no-overlap", dest="overlap", action="store_false")
    ap.add_argument("--pipeline", action="store_true", default=False)
    ap.add_argument("--build-dir", type=str, default="build")
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    if args.prompt_len <= 0:
        die("--prompt-len must be > 0")
    if args.num_docs <= 0:
        die("--num-docs must be > 0")
    if args.iters <= 0:
        die("--iters must be > 0")
    if args.warmup < 0:
        die("--warmup must be >= 0")

    prefix_lens = split_ints(args.prefix_lens)
    if not prefix_lens:
        die("--prefix-lens is empty")
    for p in prefix_lens:
        if p < 0 or p > args.prompt_len:
            die(f"prefix_len out of range: {p}")

    model_dir = resolve_model_dir(args.model)
    repo = Path.cwd()
    bench_bin = (repo / args.build_dir / "ember_prefix_cache_benchmark").resolve()
    if not bench_bin.exists():
        die(f"missing binary: {bench_bin}")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (repo / "reports" / f"stage1_prefix_cache_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    failed: List[Dict[str, str]] = []

    for idx, prefix_len in enumerate(prefix_lens, start=1):
        run_csv = out_dir / f"stage13_raw_{idx:03d}_prefix_{prefix_len}.csv"
        cmd = [
            str(bench_bin),
            "--model",
            str(model_dir),
            "--gpus",
            args.gpus,
            "--split",
            args.split,
            "--prompt-len",
            str(args.prompt_len),
            "--prefix-len",
            str(prefix_len),
            "--num-docs",
            str(args.num_docs),
            "--chunk-len",
            str(args.chunk_len),
            "--iters",
            str(args.iters),
            "--warmup",
            str(args.warmup),
            "--seed",
            str(args.seed),
            "--csv",
            str(run_csv),
        ]
        if args.overlap:
            cmd.append("--overlap")
        else:
            cmd.append("--no-overlap")
        if args.pipeline:
            cmd.append("--pipeline")
        else:
            cmd.append("--no-pipeline")

        print(f"[run {idx}] prefix_len={prefix_len}")
        p = run_cmd(cmd, cwd=repo, log_path=logs_dir / f"run_{idx:03d}.log")
        if p.returncode != 0:
            failed.append(
                {
                    "prefix_len": str(prefix_len),
                    "return_code": str(p.returncode),
                    "log_path": str(logs_dir / f"run_{idx:03d}.log"),
                }
            )
            print(f"[fail {idx}] rc={p.returncode}")
            continue

        run_rows = read_csv(run_csv)
        if not run_rows:
            failed.append(
                {
                    "prefix_len": str(prefix_len),
                    "return_code": "empty_csv",
                    "log_path": str(logs_dir / f"run_{idx:03d}.log"),
                }
            )
            print(f"[fail {idx}] empty csv")
            continue
        rows.append(run_rows[0])

    rows.sort(key=lambda r: int(r.get("prefix_len", "0")))
    summary_csv = out_dir / "stage13_prefix_cache_sweep.csv"
    summary_md = out_dir / "stage13_prefix_cache_summary.md"
    failures_csv = out_dir / "stage13_failures.csv"
    p1_input_md = out_dir / "stage13_p1_input.md"

    if rows:
        write_csv(summary_csv, rows)
        write_md(summary_md, model_dir=model_dir, rows=rows, prompt_len=args.prompt_len, num_docs=args.num_docs)
        key_1k = None
        for r in rows:
            if int(r.get("prefix_len", "0")) in (1000, 1024):
                key_1k = r
                break
        best = max(rows, key=lambda x: safe_float(x.get("savings_pct", "0")))
        lines = [
            f"# Stage1.3 P1 Input ({dt.date.today().isoformat()})",
            "",
            f"Model: {model_dir}",
            f"Setting: prompt_len={args.prompt_len}, num_docs={args.num_docs}, mode={'overlap' if args.overlap else 'no_overlap'}",
            "",
            "## Best point",
            f"- prefix_len={best.get('prefix_len','')}, savings={best.get('savings_pct','')}%, speedup={best.get('speedup_x','')}x",
        ]
        if key_1k is not None:
            lines += [
                "",
                "## Shared prefix ~1k tokens",
                f"- prefix_len={key_1k.get('prefix_len','')}, savings={key_1k.get('savings_pct','')}%, speedup={key_1k.get('speedup_x','')}x",
                f"- no_cache_total_ms={key_1k.get('no_cache_total_ms','')}, with_cache_total_ms={key_1k.get('with_cache_total_ms','')}",
            ]
        p1_input_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with failures_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["prefix_len", "return_code", "log_path"])
        w.writeheader()
        for r in failed:
            w.writerow(r)

    print("[done] stage1.3 prefix cache sweep")
    if rows:
        print(f"- sweep csv: {summary_csv}")
        print(f"- summary md: {summary_md}")
        print(f"- p1 input md: {p1_input_md}")
    else:
        print("- no successful runs")
    print(f"- failures: {failures_csv} ({len(failed)} failed runs)")
    print(f"- logs: {logs_dir}")


if __name__ == "__main__":
    main()

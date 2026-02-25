#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
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


def resolve_adapter_dir(adapter_arg: str) -> Path:
    p = Path(adapter_arg).expanduser().resolve()
    if p.is_file():
        p = p.parent
    if not p.exists():
        die(f"adapter path not found: {p}")
    if not list(p.glob("*.safetensors")):
        die(f"no .safetensors found in adapter dir: {p}")
    return p


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


def write_summary_md(path: Path, model_dir: Path, adapter_dir: Path, row: Dict[str, str]) -> None:
    lines: List[str] = []
    lines.append("# Stage 3.1 LoRA Hot Update")
    lines.append("")
    lines.append(f"- Model: `{model_dir}`")
    lines.append(f"- Adapter: `{adapter_dir}`")
    lines.append(f"- Generated at: `{dt.datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    lines.append(f"| gpus | `{row.get('gpus','')}` |")
    lines.append(f"| split | `{row.get('split','')}` |")
    lines.append(f"| scale | `{row.get('scale','')}` |")
    lines.append(f"| replace_existing | `{row.get('replace_existing','')}` |")
    lines.append(f"| effective_scale | `{row.get('effective_scale','')}` |")
    lines.append(f"| updated_matrices | `{row.get('updated_matrices','')}` |")
    lines.append(f"| skipped_matrices | `{row.get('skipped_matrices','')}` |")
    lines.append(f"| apply_ms_ext | `{row.get('apply_ms_ext','')}` |")
    lines.append(f"| apply_ms_inner | `{row.get('apply_ms_inner','')}` |")
    lines.append("")
    lines.append("## Key Point")
    lines.append(
        "- Attention q/k/v/o matrices can be merged in-place from PEFT LoRA adapter "
        "without reloading base model weights."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_p1_input(path: Path, row: Dict[str, str]) -> None:
    lines: List[str] = []
    lines.append("# P1 Input — LoRA Hot Update")
    lines.append("")
    lines.append(
        f"- LoRA hot update latency (external wall): `{row.get('apply_ms_ext','')}` ms; "
        f"runtime internal: `{row.get('apply_ms_inner','')}` ms."
    )
    lines.append(
        f"- Updated matrices: `{row.get('updated_matrices','')}`, "
        f"skipped: `{row.get('skipped_matrices','')}`."
    )
    lines.append(
        f"- Effective scale used in merge: `{row.get('effective_scale','')}` "
        f"(user scale `{row.get('scale','')}`)."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Stage 3.1 LoRA hot update benchmark.")
    ap.add_argument("--model", type=str, required=True, help="model path or HF model id")
    ap.add_argument("--adapter", type=str, required=True, help="LoRA adapter dir")
    ap.add_argument("--gpus", type=str, default="0,1")
    ap.add_argument("--split", type=str, default="9,27")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--replace-existing", action="store_true", default=False)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--build-dir", type=str, default="build")
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    if args.iters <= 0:
        die("--iters must be > 0")
    if args.warmup < 0:
        die("--warmup must be >= 0")
    _ = split_ints(args.gpus)
    if args.split.strip():
        s = split_ints(args.split)
        if len(s) != 2:
            die("--split expects A,B")

    model_dir = resolve_model_dir(args.model)
    adapter_dir = resolve_adapter_dir(args.adapter)

    repo = Path.cwd()
    bench_bin = (repo / args.build_dir / "ember_lora_hot_update_benchmark").resolve()
    if not bench_bin.exists():
        die(f"missing binary: {bench_bin}")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (repo / "reports" / f"stage31_lora_hot_update_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "stage31_lora_hot_update.csv"
    cmd = [
        str(bench_bin),
        "--model",
        str(model_dir),
        "--adapter",
        str(adapter_dir),
        "--gpus",
        args.gpus,
        "--split",
        args.split,
        "--scale",
        str(args.scale),
        "--iters",
        str(args.iters),
        "--warmup",
        str(args.warmup),
        "--csv",
        str(csv_path),
    ]
    if args.replace_existing:
        cmd.append("--replace-existing")

    p = run_cmd(cmd, cwd=repo, log_path=logs_dir / "run.log")
    if p.returncode != 0:
        die(f"benchmark failed rc={p.returncode}; see {logs_dir / 'run.log'}")

    rows = read_csv(csv_path)
    if not rows:
        die(f"empty csv: {csv_path}")
    row = rows[0]

    write_summary_md(out_dir / "stage31_summary.md", model_dir, adapter_dir, row)
    write_p1_input(out_dir / "stage31_p1_input.md", row)

    print(f"[done] out_dir={out_dir}")
    print(
        f"[result] updated={row.get('updated_matrices','')} "
        f"apply_ms_ext={safe_float(row.get('apply_ms_ext','0')):.3f} "
        f"apply_ms_inner={safe_float(row.get('apply_ms_inner','0')):.3f}"
    )


if __name__ == "__main__":
    main()

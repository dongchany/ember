#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def run(cmd: List[str],
        cwd: Path,
        log_path: Path,
        check: bool = True,
        progress: bool = False,
        env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if progress:
        print(f"[run] {' '.join(cmd)}", flush=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        p = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, env=merged_env)
        f.write(p.stdout)
        if p.stderr:
            f.write("\n[stderr]\n")
            f.write(p.stderr)
    if progress:
        dt_s = time.time() - t0
        if p.returncode == 0:
            tail = ""
            lines = [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]
            if lines:
                tail = lines[-1]
            msg = f"[ok ] rc=0 ({dt_s:.1f}s)"
            if tail:
                msg += f" | {tail}"
            print(msg, flush=True)
        else:
            err_tail = ""
            elines = [ln.strip() for ln in p.stderr.splitlines() if ln.strip()]
            if elines:
                err_tail = elines[-1]
            msg = f"[err] rc={p.returncode} ({dt_s:.1f}s)"
            if err_tail:
                msg += f" | {err_tail}"
            msg += f" | log={log_path}"
            print(msg, flush=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)} (see {log_path})")
    return p


def parse_csv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def format_md_table(headers: List[str], rows: List[List[str]]) -> str:
    if not rows:
        return "_(no data)_\n"
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out) + "\n"


def load_model_config(model_dir: Path) -> Dict:
    cfg_path = model_dir / "config.json"
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)

def resolve_snapshot_dir(model_or_snapshot_dir: Path) -> Path:
    """
    Accepts either:
      - a snapshot dir (contains config.json)
      - a HF cache model dir (contains snapshots/<hash>/config.json)
    Returns the resolved snapshot dir.
    """
    d = model_or_snapshot_dir.expanduser().resolve()
    if (d / "config.json").exists():
        # Ensure this snapshot is compatible with Ember (expects safetensors weights).
        if not list(d.glob("*.safetensors")):
            raise FileNotFoundError(f"no *.safetensors found in snapshot dir: {d}")
        return d
    snap_root = d / "snapshots"
    if snap_root.exists() and snap_root.is_dir():
        snaps = [p for p in snap_root.iterdir() if p.is_dir() and (p / "config.json").exists()]
        if not snaps:
            raise FileNotFoundError(f"no snapshots with config.json under: {snap_root}")
        snaps.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        # Ensure this snapshot is compatible with Ember (expects safetensors weights).
        if not list(snaps[0].glob("*.safetensors")):
            raise FileNotFoundError(f"no *.safetensors found in snapshot dir: {snaps[0]}")
        return snaps[0]
    raise FileNotFoundError(f"cannot resolve model dir (need config.json or snapshots/*/config.json): {d}")

def resolve_model_arg(arg: str, hub_root: Optional[Path]) -> Path:
    """
    Resolve a model argument to a snapshot directory.
    - If arg is a valid path: resolve_snapshot_dir(arg).
    - Else if hub_root is provided: search under hub_root for dirs containing arg as substring.
    """
    p = Path(arg).expanduser()
    if p.exists():
        return resolve_snapshot_dir(p)
    if hub_root is None:
        raise FileNotFoundError(f"model path not found: {p}")
    root = hub_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"hub root not found: {root}")
    candidates = []
    needle = arg.lower()
    for d in root.iterdir():
        if not d.is_dir():
            continue
        if needle not in d.name.lower():
            continue
        try:
            snap = resolve_snapshot_dir(d)
        except Exception:
            continue
        candidates.append(snap)
    if not candidates:
        raise FileNotFoundError(f"cannot resolve '{arg}' under hub root: {root}")
    candidates.sort(key=lambda p2: p2.stat().st_mtime, reverse=True)
    return candidates[0]


def _prepend_env_path(env: Dict[str, str], key: str, prefix: str) -> Dict[str, str]:
    cur = env.get(key, os.environ.get(key, ""))
    if cur:
        env[key] = prefix + ":" + cur
    else:
        env[key] = prefix
    return env


def resolve_gguf_in_hub(hub_root: Path, filename_substr: str) -> Path:
    """
    Find a GGUF file under a HF cache hub_root (models--*/snapshots/*/*.gguf).
    Prefers the common "unsloth" layout when present.
    """
    root = hub_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"hub root not found: {root}")

    unsloth = root / "models--unsloth--Qwen3-4B-GGUF"
    if unsloth.exists():
        hits = sorted(unsloth.glob("snapshots/*/*.gguf"))
        hits = [p for p in hits if filename_substr.lower() in p.name.lower()]
        if hits:
            return hits[0]

    hits = sorted(root.glob("models--*/snapshots/*/*.gguf"))
    hits = [p for p in hits if filename_substr.lower() in p.name.lower()]
    if not hits:
        raise FileNotFoundError(f"no gguf found in {root} matching '{filename_substr}'")
    return hits[0]


def run_llama_bench(repo: Path,
                    logs_dir: Path,
                    progress: bool,
                    llama_bin_dir: Path,
                    gguf: Path,
                    prompt_len: int,
                    gen_len: int,
                    device: str,
                    split_mode: str,
                    tensor_split: str) -> Tuple[float, float, float, float]:
    """
    Runs llama-bench and returns:
      prefill_ms, prefill_tok_s, decode_ms, decode_tok_s
    """
    llama_bench = llama_bin_dir / "llama-bench"
    if not llama_bench.exists():
        raise FileNotFoundError(f"missing llama-bench: {llama_bench}")
    if not gguf.exists():
        raise FileNotFoundError(f"missing gguf: {gguf}")

    cmd = [
        str(llama_bench),
        "-m", str(gguf),
        "-p", str(prompt_len),
        "-n", str(gen_len),
        "-r", "1",
        # llama-bench expects an integer for -ngl/--n-gpu-layers (unlike some other llama.cpp tools
        # which accept 'auto'/'all'). Use a large number to effectively offload all layers.
        "-ngl", "999",
        "-o", "json",
    ]
    if device:
        cmd += ["-dev", device]
    if split_mode:
        cmd += ["-sm", split_mode]
    if tensor_split:
        cmd += ["-ts", tensor_split]

    env: Dict[str, str] = {}
    _prepend_env_path(env, "LD_LIBRARY_PATH", str(llama_bin_dir))

    p = run(cmd,
            cwd=repo,
            log_path=logs_dir / f"E_llama_bench_{gguf.stem}_{device.replace('/', '-') or 'auto'}.log",
            check=False,
            progress=progress,
            env=env)
    if p.returncode != 0:
        raise RuntimeError(f"llama-bench failed rc={p.returncode}")

    data = json.loads(p.stdout)
    if not isinstance(data, list) or not data:
        raise RuntimeError("unexpected llama-bench json output (empty/non-list)")

    prefill = None
    decode = None
    for item in data:
        try:
            n_prompt = int(item.get("n_prompt", 0))
            n_gen = int(item.get("n_gen", 0))
        except Exception:
            continue
        if n_prompt > 0 and n_gen == 0:
            prefill = item
        if n_gen > 0 and n_prompt == 0:
            decode = item

    if prefill is None or decode is None:
        raise RuntimeError("unexpected llama-bench json output (missing prompt/gen entries)")

    prefill_ms = float(prefill.get("avg_ns", 0.0)) / 1e6
    prefill_tok_s = float(prefill.get("avg_ts", 0.0))
    decode_ms = float(decode.get("avg_ns", 0.0)) / 1e6
    decode_tok_s = float(decode.get("avg_ts", 0.0))
    return prefill_ms, prefill_tok_s, decode_ms, decode_tok_s


def llama_list_devices(repo: Path, logs_dir: Path, progress: bool, llama_bin_dir: Path) -> List[str]:
    llama_bench = llama_bin_dir / "llama-bench"
    if not llama_bench.exists():
        raise FileNotFoundError(f"missing llama-bench: {llama_bench}")

    env: Dict[str, str] = {}
    _prepend_env_path(env, "LD_LIBRARY_PATH", str(llama_bin_dir))
    p = run([str(llama_bench), "--list-devices"],
            cwd=repo,
            log_path=logs_dir / "E_llama_list_devices.log",
            check=False,
            progress=progress,
            env=env)
    if p.returncode != 0:
        return []

    ids: List[str] = []
    for ln in p.stdout.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.lower().startswith("available devices"):
            continue
        if s.startswith("(") and "none" in s.lower():
            continue
        tok = s.split()[0].strip()
        if tok.endswith(":"):
            tok = tok[:-1]
        # Common formats include "0:" / "cuda0:" / "CUDA0:"; keep as-is.
        if tok:
            ids.append(tok)
    return ids


def pick_llama_device(dev_ids: List[str], idx: int) -> str:
    """
    Pick a llama.cpp device id for GPU `idx`.

    llama-bench commonly exposes devices like "CUDA0"/"CUDA1" (not plain "0"/"1").
    Keep the original token from `--list-devices` when possible.
    """
    if not dev_ids:
        return ""

    # Prefer exact matches (case-insensitive) for common formats.
    preferred = [f"CUDA{idx}", f"cuda{idx}", str(idx)]
    for p in preferred:
        for d in dev_ids:
            if d == p or d.lower() == p.lower():
                return d

    # Next, accept numeric suffix matches (e.g., "gpu0", "cuda0", "0").
    for d in dev_ids:
        m = re.search(r"(\d+)$", d)
        if m and int(m.group(1)) == idx:
            return d

    # Fall back to list order.
    return dev_ids[idx] if idx < len(dev_ids) else dev_ids[0]


def get_num_layers(cfg: Dict) -> int:
    for k in ("num_hidden_layers", "num_layers", "n_layer"):
        v = cfg.get(k)
        if isinstance(v, int) and v > 0:
            return v
    return 0


def auto_split_for(cfg: Dict, gpus: List[int]) -> str:
    if len(gpus) != 2:
        return ""
    n = get_num_layers(cfg)
    if n <= 0:
        return ""
    a = n // 2
    b = n - a
    if a <= 0 or b <= 0:
        return ""
    return f"{a},{b}"


def nearest_p2p_time_us(p2p_rows: List[Dict[str, str]], bytes_needed: int, direction: str) -> Optional[float]:
    # Prefer cudaMemcpyPeerAsync rows for the requested direction.
    candidates: List[Tuple[int, float]] = []
    for r in p2p_rows:
        if r.get("method") != "cudaMemcpyPeerAsync":
            continue
        if r.get("direction") != direction:
            continue
        try:
            b = int(r["data_size_bytes"])
            us = float(r["transfer_time_us"])
        except Exception:
            continue
        candidates.append((b, us))
    if not candidates:
        return None
    candidates.sort(key=lambda x: abs(x[0] - bytes_needed))
    return candidates[0][1]


@dataclass
class PipelineSimResult:
    total_us: float
    bubble_ratio: float
    gpu0_util: float
    gpu1_util: float


def simulate_2stage(stage0_us: float, stage1_us: float, transfer_us: float, micro_batches: int) -> PipelineSimResult:
    t0 = 0.0
    t1 = 0.0
    busy0 = 0.0
    busy1 = 0.0
    for _ in range(micro_batches):
        start0 = t0
        end0 = start0 + stage0_us
        busy0 += stage0_us
        t0 = end0

        arrive1 = end0 + transfer_us
        start1 = max(t1, arrive1)
        end1 = start1 + stage1_us
        busy1 += stage1_us
        t1 = end1

    total = max(t0, t1)
    util0 = busy0 / total if total > 0 else 0.0
    util1 = busy1 / total if total > 0 else 0.0
    avg_util = 0.5 * (util0 + util1)
    bubble = max(0.0, 1.0 - avg_util)
    return PipelineSimResult(total_us=total, bubble_ratio=bubble, gpu0_util=util0, gpu1_util=util1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Ember A/B/C/D experiments and generate CSV + Markdown report.")
    ap.add_argument("--repo", type=str, default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--build-dir", type=str, default="build")
    ap.add_argument("--out-root", type=str, default="reports")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--hub-root", type=str, default="", help="optional HF cache root (e.g. ~/xilinx/huggingface/hub); allows --model-* to be a name substring")
    ap.add_argument("--progress", dest="progress", action="store_true", default=True, help="print per-step progress to terminal (default: enabled)")
    ap.add_argument("--no-progress", dest="progress", action="store_false", help="disable per-step progress prints")

    ap.add_argument("--gpus", type=str, default="0,1", help="e.g. 0,1")
    ap.add_argument("--split", type=str, default="", help="(legacy) split for all phases; prefer --split-b/--split-d")
    ap.add_argument("--split-a", type=str, default="", help="split for A bubble sim (defaults to even split for model-a)")
    ap.add_argument("--split-b", type=str, default="", help="split for B (defaults to even split for model-b)")
    ap.add_argument("--split-c", type=str, default="", help="split for C serve sim (defaults to even split for model-b)")
    ap.add_argument("--split-d", type=str, default="", help="split for D (defaults to even split for model-d)")
    ap.add_argument("--chunk-len", type=int, default=128)
    ap.add_argument("--prompt-lens", type=str, default="128,512,1024,2048")
    ap.add_argument("--decode-steps", type=int, default=100)
    ap.add_argument("--p2p-sizes", type=str, default="1k,10k,100k,1m,10m,100m")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--bench-prompt-len", type=int, default=1024)
    ap.add_argument("--bench-gen-len", type=int, default=100)
    ap.add_argument("--decode-batches", type=str, default="1,4,8")

    ap.add_argument("--model-a", type=str, default="", help="model dir or name substring for A.1 phase analysis (e.g. Qwen3-4B)")
    ap.add_argument("--model-b", type=str, default="", help="model dir or name substring for B/C (e.g. Qwen3-8B)")
    ap.add_argument("--model-d", type=str, default="", help="model dir or name substring for D main eval (optional; can be 'auto')")
    ap.add_argument("--fallback-d-to-b", dest="fallback_d_to_b", action="store_true", default=True, help="(default) if D fails (OOM etc.), fallback to model-b")
    ap.add_argument("--no-fallback-d-to-b", dest="fallback_d_to_b", action="store_false", help="disable D->B fallback")
    ap.add_argument("--include-phase-aware", action="store_true", help="also run PhaseAwareScheduler variant in B/D")
    ap.add_argument("--c-source",
                    type=str,
                    default="serve",
                    choices=["serve", "benchmark", "both"],
                    help="how to generate C results: serve (default) uses ember_serve_benchmark; benchmark uses ember_benchmark decode-batch sweep; both runs both")
    ap.add_argument("--extra-single-vs-dual",
                    dest="extra_single_vs_dual",
                    action="store_true",
                    default=True,
                    help="(default) run extra single-GPU vs dual-GPU comparison on model-a (if possible)")
    ap.add_argument("--no-extra-single-vs-dual",
                    dest="extra_single_vs_dual",
                    action="store_false",
                    help="disable extra single-vs-dual experiment")
    ap.add_argument("--split-sweep",
                    type=str,
                    default="",
                    help="optional split sweep list for model-b, e.g. '12,24;18,18;24,12' (must sum to num_layers)")
    ap.add_argument("--prompt-sweep",
                    type=str,
                    default="",
                    help="optional prompt length sweep for model-b on 2 GPUs, e.g. '128,512,1024,2048'")

    ap.add_argument("--include-llama", action="store_true", help="also run llama.cpp baseline (llama-bench) and include in Extras")
    ap.add_argument("--include-llama-quantized",
                    action="store_true",
                    help="also run a quantized llama.cpp baseline (not apples-to-apples; included as a practical reference)")
    ap.add_argument("--llama-bin-dir", type=str, default="", help="path to llama.cpp build/bin (must contain llama-bench)")
    ap.add_argument("--llama-gguf-bf16", type=str, default="auto", help="GGUF path for BF16 baseline, or 'auto' (requires --hub-root)")
    ap.add_argument("--llama-gguf-q4", type=str, default="auto", help="GGUF path for quantized baseline (e.g. Q4_K_M), or 'auto' (requires --hub-root)")
    ap.add_argument("--llama-dual-split-mode",
                    type=str,
                    default="row",
                    choices=["none", "layer", "row"],
                    help="llama.cpp split mode for multi-GPU (default: row)")
    ap.add_argument("--llama-tensor-split", type=str, default="0.5/0.5", help="llama.cpp tensor split ratios for multi-GPU, e.g. 0.5/0.5")

    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--skip-b", action="store_true")
    ap.add_argument("--skip-c", action="store_true")
    ap.add_argument("--skip-d", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    build_dir = (repo / args.build_dir).resolve()
    gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    if len(gpus) < 1:
        raise SystemExit("--gpus is empty")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = args.tag.strip()
    name = ts + (f"_{tag}" if tag else "")
    out_dir = (repo / args.out_root / name).resolve()
    logs_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build.
    if not args.skip_build:
        run(["cmake", "-S", ".", "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release"], cwd=repo, log_path=logs_dir / "build_configure.log", progress=args.progress)
        run(["cmake", "--build", str(build_dir), "-j"], cwd=repo, log_path=logs_dir / "build_build.log", progress=args.progress)

    # Resolve binaries.
    bin_phase = build_dir / "ember_phase_analysis"
    bin_p2p = build_dir / "ember_p2p_bandwidth"
    bin_bench = build_dir / "ember_benchmark"
    bin_serve = build_dir / "ember_serve_benchmark"
    for b in [bin_p2p, bin_bench, bin_serve]:
        if not b.exists():
            raise SystemExit(f"missing binary: {b} (build first)")

    # Collect system info best-effort.
    sysinfo_md = []
    try:
        p = run(["nvidia-smi", "-L"], cwd=repo, log_path=logs_dir / "nvidia_smi_L.log", check=False, progress=args.progress)
        if p.returncode == 0:
            sysinfo_md.append("```\n" + p.stdout.strip() + "\n```\n")
    except Exception:
        pass
    try:
        p = run(["nvidia-smi", "topo", "-m"], cwd=repo, log_path=logs_dir / "nvidia_smi_topo.log", check=False, progress=args.progress)
        if p.returncode == 0:
            sysinfo_md.append("```\n" + p.stdout.strip() + "\n```\n")
    except Exception:
        pass

    # A: phase analysis + p2p bandwidth + bubble model.
    a_phase_csv = out_dir / "A_phase_analysis.csv"
    a_p2p_csv = out_dir / "A_p2p_bandwidth.csv"
    a_bubble_csv = out_dir / "A_pipeline_bubble.csv"

    model_a = Path(args.model_a).expanduser() if args.model_a else None
    model_b = Path(args.model_b).expanduser() if args.model_b else None
    hub_root = Path(args.hub_root).expanduser().resolve() if args.hub_root else None

    # Auto-pick model-b and model-a if user provided hub_root but omitted them.
    if model_b is None and hub_root is not None:
        try:
            model_b = Path(resolve_model_arg("Qwen3-8B", hub_root))
        except Exception:
            model_b = None
    if model_a is None and hub_root is not None:
        # Prefer 4B instruct (better representative), fallback to 0.6B.
        try:
            model_a = Path(resolve_model_arg("Qwen3-4B", hub_root))
        except Exception:
            try:
                model_a = Path(resolve_model_arg("Qwen3-0.6B", hub_root))
            except Exception:
                model_a = None

    # Use model-b as fallback for A if user didn't provide model-a.
    if model_a is None and model_b is not None:
        model_a = model_b

    model_a_cfg = None
    if model_a is not None:
        model_a = resolve_model_arg(str(model_a), hub_root) if not model_a.exists() else resolve_snapshot_dir(model_a)
        model_a_cfg = load_model_config(model_a)
    model_b_cfg = None
    if model_b is not None:
        model_b = resolve_model_arg(str(model_b), hub_root) if not model_b.exists() else resolve_snapshot_dir(model_b)
        model_b_cfg = load_model_config(model_b)

    # D model resolution (optional).
    model_d: Optional[Path] = None
    model_d_cfg: Optional[Dict] = None
    if args.model_d:
        if args.model_d.strip().lower() != "auto":
            model_d = resolve_model_arg(args.model_d, hub_root)
            model_d_cfg = load_model_config(model_d)
        else:
            # "auto": prefer 14B if present under hub_root, otherwise fallback to model-b.
            if hub_root is not None:
                try:
                    model_d = resolve_model_arg("Qwen3-14B", hub_root)
                    model_d_cfg = load_model_config(model_d)
                except Exception:
                    model_d = None
                    model_d_cfg = None
            if model_d is None and model_b is not None:
                model_d = model_b
                model_d_cfg = model_b_cfg

    # Pre-compute chosen splits for reporting (may still be overridden per phase below).
    split_a_chosen = args.split_a or args.split or (auto_split_for(model_a_cfg or {}, gpus) if model_a_cfg else "")
    split_b_chosen = args.split_b or args.split or (auto_split_for(model_b_cfg or {}, gpus) if model_b_cfg else "")
    split_c_chosen = args.split_c or args.split_b or args.split or (auto_split_for(model_b_cfg or {}, gpus) if model_b_cfg else "")
    split_d_chosen = args.split_d or args.split or (auto_split_for(model_d_cfg or {}, gpus) if model_d_cfg else "")

    # Ensure chunk_len is included in phase analysis prompt lens for better bubble estimation.
    plens = [int(x) for x in args.prompt_lens.split(",") if x.strip() != ""]
    if args.chunk_len not in plens:
        plens.append(args.chunk_len)
    plens = sorted(set(plens))

    if not args.skip_a:
        if model_a is None:
            print("A skipped: --model-a (or --model-b as fallback) not provided", file=sys.stderr)
        else:
            if not bin_phase.exists():
                raise SystemExit(f"missing binary: {bin_phase} (build first)")
            run(
                [
                    str(bin_phase),
                    "--model",
                    str(model_a),
                    "--prompt-lens",
                    ",".join(str(x) for x in plens),
                    "--decode-steps",
                    str(args.decode_steps),
                    "--output",
                    str(a_phase_csv),
                ],
                cwd=repo,
                log_path=logs_dir / "A_phase_analysis.log",
                progress=args.progress,
            )
            run(
                [
                    str(bin_p2p),
                    "--gpus",
                    args.gpus,
                    "--sizes",
                    args.p2p_sizes,
                    "--method",
                    "both",
                    "--direction",
                    "both",
                    "--csv",
                    str(a_p2p_csv),
                ],
                cwd=repo,
                log_path=logs_dir / "A_p2p_bandwidth.log",
                progress=args.progress,
            )

            # Bubble model (very small 2-stage sim). Requires split + model hidden size.
            split_a = args.split_a or args.split or auto_split_for(model_a_cfg, gpus)
            if model_a_cfg and len(gpus) >= 2 and split_a:
                p2p_rows = parse_csv(a_p2p_csv)
                phase_rows = parse_csv(a_phase_csv)

                # stage split
                a_split, b_split = [int(x) for x in split_a.split(",")]
                if a_split + b_split != get_num_layers(model_a_cfg):
                    # if mismatch, still write bubble csv as "n/a"
                    pass

                # Aggregate per-layer ms at given prompt_len.
                # Map: prompt_len -> (prefill_sum_ms_stage0, prefill_sum_ms_stage1, decode_sum_ms_stage0, decode_sum_ms_stage1)
                agg: Dict[int, Dict[str, float]] = {}
                for r in phase_rows:
                    try:
                        plen = int(r["prompt_len"])
                        lid = int(r["layer_id"])
                        pre_ms = float(r["prefill_time_ms"])
                        dec_ms = float(r["decode_time_ms"])
                    except Exception:
                        continue
                    if plen not in agg:
                        agg[plen] = {"pre0": 0.0, "pre1": 0.0, "dec0": 0.0, "dec1": 0.0}
                    if lid < a_split:
                        agg[plen]["pre0"] += pre_ms
                        agg[plen]["dec0"] += dec_ms
                    else:
                        agg[plen]["pre1"] += pre_ms
                        agg[plen]["dec1"] += dec_ms

                hidden = int(model_a_cfg.get("hidden_size", 0))
                elem = 2  # assume fp16/bf16 activation size for transfer
                direction = f"gpu{gpus[0]}_to_gpu{gpus[1]}"

                with a_bubble_csv.open("w", encoding="utf-8", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["phase", "prompt_len", "micro_batches", "transfer_bytes", "transfer_us", "total_ms", "bubble_ratio", "gpu0_util", "gpu1_util"])

                    # Prefill: micro_batches = ceil(prompt_len/chunk_len), compute per micro_batch from prompt_len=chunk_len measurement.
                    base_plen = args.chunk_len
                    if base_plen in agg:
                        stage0_us = agg[base_plen]["pre0"] * 1000.0
                        stage1_us = agg[base_plen]["pre1"] * 1000.0
                        xfer_bytes = args.chunk_len * hidden * elem
                        xfer_us = nearest_p2p_time_us(p2p_rows, xfer_bytes, direction) or 0.0
                        for plen in [p for p in plens if p >= args.chunk_len]:
                            mb = (plen + args.chunk_len - 1) // args.chunk_len
                            sim = simulate_2stage(stage0_us, stage1_us, xfer_us, mb)
                            w.writerow(["prefill", plen, mb, xfer_bytes, f"{xfer_us:.3f}", f"{sim.total_us/1000.0:.3f}", f"{sim.bubble_ratio:.4f}", f"{sim.gpu0_util:.4f}", f"{sim.gpu1_util:.4f}"])

                    # Decode: per token compute is from prompt_len bucket (avg over decode-steps in phase_analysis).
                    xfer_bytes = hidden * elem
                    xfer_us = nearest_p2p_time_us(p2p_rows, xfer_bytes, direction) or 0.0
                    for plen in [p for p in plens if p in agg]:
                        stage0_us = agg[plen]["dec0"] * 1000.0
                        stage1_us = agg[plen]["dec1"] * 1000.0
                        sim = simulate_2stage(stage0_us, stage1_us, xfer_us, micro_batches=1)
                        w.writerow(["decode", plen, 1, xfer_bytes, f"{xfer_us:.3f}", f"{sim.total_us/1000.0:.3f}", f"{sim.bubble_ratio:.4f}", f"{sim.gpu0_util:.4f}", f"{sim.gpu1_util:.4f}"])

    # B/C/D: e2e bench sweeps (writes aggregated CSV; raw stdout in logs).
    results_csv = out_dir / "benchmark_results.csv"
    serve_csv = out_dir / "serve_results.csv"
    extra_csv = out_dir / "extra_results.csv"
    rows_out: List[List[str]] = []
    header = ["phase", "policy", "mode", "model", "prompt_len", "gen_len", "chunk_len", "batch_size", "ttft_ms", "prefill_ms", "decode_ms", "decode_tok_s", "split", "gpus"]

    def bench_once(phase: str,
                   policy: str,
                   model_dir: Path,
                   mode_flag: str,
                   decode_batch: int,
                   split: str,
                   gpus_str: str,
                   prompt_len: Optional[int] = None,
                   gen_len: Optional[int] = None,
                   extra: Optional[List[str]] = None) -> None:
        pl = args.bench_prompt_len if prompt_len is None else prompt_len
        gl = args.bench_gen_len if gen_len is None else gen_len
        cmd = [str(bin_bench), "--model", str(model_dir), "--gpus", gpus_str, "--prompt-len", str(pl), "--gen-len", str(gl), "--chunk-len", str(args.chunk_len), "--iters", str(args.iters), "--decode-batch", str(decode_batch)]
        if split:
            cmd += ["--split", split]
        if extra:
            cmd += extra
        cmd.append(mode_flag)
        p = run(cmd, cwd=repo, log_path=logs_dir / f"{phase}_{model_dir.name}_{mode_flag.replace('--','')}_b{decode_batch}.log", progress=args.progress)
        line = p.stdout.strip().splitlines()[-1].strip() if p.stdout.strip() else ""
        # Expected: mode,prompt_len,gen_len,chunk_len,batch_size,ttft_ms,prefill_ms,decode_ms,decode_tok_s
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 9:
            raise RuntimeError(f"unexpected ember_benchmark output: {line}")
        mode, prompt_len, gen_len, chunk_len, batch_size, ttft_ms, pre_ms, dec_ms, tok_s = parts
        rows_out.append([phase, policy, mode, model_dir.name, prompt_len, gen_len, chunk_len, batch_size, ttft_ms, pre_ms, dec_ms, tok_s, split or "", gpus_str])

    if not args.skip_b:
        if model_b is None:
            print("B skipped: --model-b not provided", file=sys.stderr)
        else:
            split_b = args.split_b or args.split or (auto_split_for(model_b_cfg or {}, gpus) if model_b_cfg else "")
            bench_once("B", "naive", model_b, "--no-overlap", decode_batch=1, split=split_b, gpus_str=args.gpus)
            bench_once("B", "naive", model_b, "--overlap", decode_batch=1, split=split_b, gpus_str=args.gpus)
            if args.include_phase_aware:
                bench_once("B", "phase_aware", model_b, "--overlap", decode_batch=1, split=split_b, gpus_str=args.gpus, extra=["--phase-aware"])

    if not args.skip_c and args.c_source in ("benchmark", "both"):
        if model_b is None:
            print("C skipped: --model-b not provided", file=sys.stderr)
        else:
            batches = [int(x) for x in args.decode_batches.split(",") if x.strip() != ""]
            split_c = args.split_c or args.split_b or args.split or (auto_split_for(model_b_cfg or {}, gpus) if model_b_cfg else "")
            for b in batches:
                # NOTE: ember_benchmark's batch>1 prefill uses full-seq attention (O(batch * prompt_len^2)) and can OOM
                # on 12GB GPUs for prompt=1024, batch=8. We treat failures as non-fatal.
                try:
                    bench_once("C", "naive", model_b, "--no-overlap", decode_batch=b, split=split_c, gpus_str=args.gpus)
                except Exception as ex:
                    print(f"[warn] C/benchmark batch={b} failed; skipping. Reason: {ex}", file=sys.stderr, flush=True)

    if not args.skip_d:
        if model_d is not None:
            split_d = args.split_d or args.split or (auto_split_for(model_d_cfg or {}, gpus) if model_d_cfg else "")
            # Main D: do both modes (batch=1), and a decode batch sweep for the same model.
            try:
                bench_once("D", "naive", model_d, "--no-overlap", decode_batch=1, split=split_d, gpus_str=args.gpus)
                bench_once("D", "naive", model_d, "--overlap", decode_batch=1, split=split_d, gpus_str=args.gpus)
                if args.include_phase_aware:
                    bench_once("D", "phase_aware", model_d, "--overlap", decode_batch=1, split=split_d, gpus_str=args.gpus, extra=["--phase-aware"])
                batches = [int(x) for x in args.decode_batches.split(",") if x.strip() != ""]
                for b in batches:
                    if b == 1:
                        # Already measured above (batch=1 for both overlap modes).
                        continue
                    bench_once("D", "naive", model_d, "--no-overlap", decode_batch=b, split=split_d, gpus_str=args.gpus)
            except Exception as ex:
                if args.fallback_d_to_b and model_b is not None and model_d != model_b:
                    print(f"D failed for {model_d} ({ex}); falling back to model-b {model_b}", file=sys.stderr)
                    split_d2 = args.split_b or args.split or (auto_split_for(model_b_cfg or {}, gpus) if model_b_cfg else "")
                    bench_once("D", "naive", model_b, "--no-overlap", decode_batch=1, split=split_d2, gpus_str=args.gpus)
                    bench_once("D", "naive", model_b, "--overlap", decode_batch=1, split=split_d2, gpus_str=args.gpus)
                    if args.include_phase_aware:
                        bench_once("D", "phase_aware", model_b, "--overlap", decode_batch=1, split=split_d2, gpus_str=args.gpus, extra=["--phase-aware"])
                else:
                    print(f"D skipped due to failure: {ex}", file=sys.stderr)
        else:
            print("D skipped: --model-d not provided", file=sys.stderr)

    with results_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows_out:
            w.writerow(r)

    # C: continuous batching / per-slot positions (PhaseAwareBatchScheduler).
    serve_rows: List[List[str]] = []
    serve_header = ["phase", "model", "mode", "num_reqs", "batch_size", "prompt_len", "gen_len", "vary_gen", "prefill_ms", "decode_ms", "gen_tokens", "decode_tok_s", "split", "gpus"]
    if not args.skip_c and model_b is not None and args.c_source in ("serve", "both"):
        batches = [int(x) for x in args.decode_batches.split(",") if x.strip() != ""]
        split_c = args.split_c or args.split_b or args.split or (auto_split_for(model_b_cfg or {}, gpus) if model_b_cfg else "")
        for b in batches:
            num_reqs = max(b * 4, b)
            cmd = [str(bin_serve),
                   "--model", str(model_b),
                   "--gpus", args.gpus,
                   "--batch-size", str(b),
                   "--num-req", str(num_reqs),
                   "--prompt-len", str(args.bench_prompt_len),
                   "--gen-len", str(args.bench_gen_len),
                   "--prefill-chunk-len", str(args.chunk_len)]
            if split_c:
                cmd += ["--split", split_c]
            try:
                p = run(cmd, cwd=repo, log_path=logs_dir / f"C_serve_{model_b.name}_b{b}.log", progress=args.progress)
                line = p.stdout.strip().splitlines()[-1].strip() if p.stdout.strip() else ""
                parts = [x.strip() for x in line.split(",")]
                if len(parts) != 10:
                    raise RuntimeError(f"unexpected ember_serve_benchmark output: {line}")
                mode, num_reqs_s, batch_s, plen_s, glen_s, vary_s, pre_ms, dec_ms, gen_tok, tok_s = parts
                serve_rows.append(["C", model_b.name, mode, num_reqs_s, batch_s, plen_s, glen_s, vary_s, pre_ms, dec_ms, gen_tok, tok_s, split_c or "", args.gpus])
                # Also append a normalized row into benchmark_results.csv so the main table contains C.
                rows_out.append(["C", "phase_aware_batch", mode, model_b.name, plen_s, glen_s, str(args.chunk_len), batch_s, "", pre_ms, dec_ms, tok_s, split_c or "", args.gpus])
            except Exception as ex:
                print(f"[warn] C/serve batch={b} failed; skipping. Reason: {ex}", file=sys.stderr, flush=True)

    with serve_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(serve_header)
        for r in serve_rows:
            w.writerow(r)

    # Extra experiments (optional): single vs dual GPU on model-a; split/prompt sweeps on model-b.
    extra_rows: List[List[str]] = []
    extra_header = ["name", "model", "gpus", "split", "prompt_len", "gen_len", "batch_size", "ttft_ms", "prefill_ms", "decode_ms", "decode_tok_s", "note"]

    if args.extra_single_vs_dual and model_a is not None:
        try:
            single_gpu = str(gpus[0])
            dual_gpu = args.gpus if len(gpus) >= 2 else single_gpu
            split_a_bench = args.split_a or args.split or (auto_split_for(model_a_cfg or {}, gpus) if model_a_cfg else "")
            # Single GPU baseline.
            bench_once("E", "single_gpu", model_a, "--no-overlap", decode_batch=1, split="", gpus_str=single_gpu)
            # Dual GPU pipeline (no-overlap) to highlight bubble.
            if len(gpus) >= 2:
                bench_once("E", "dual_gpu", model_a, "--no-overlap", decode_batch=1, split=split_a_bench, gpus_str=dual_gpu)
        except Exception as ex:
            print(f"[warn] extra single-vs-dual failed; skipping. Reason: {ex}", file=sys.stderr, flush=True)

    if args.split_sweep and model_b is not None and len(gpus) >= 2:
        split_items = [x.strip() for x in args.split_sweep.split(";") if x.strip()]
        for s in split_items:
            try:
                bench_once("E_split", "naive", model_b, "--no-overlap", decode_batch=1, split=s, gpus_str=args.gpus)
            except Exception as ex:
                print(f"[warn] split sweep {s} failed; skipping. Reason: {ex}", file=sys.stderr, flush=True)

    if args.prompt_sweep and model_b is not None and len(gpus) >= 2:
        lens = [int(x) for x in args.prompt_sweep.split(",") if x.strip() != ""]
        split_b = args.split_b or args.split or (auto_split_for(model_b_cfg or {}, gpus) if model_b_cfg else "")
        for plen in lens:
            try:
                bench_once("E_prompt", "naive", model_b, "--overlap", decode_batch=1, split=split_b, gpus_str=args.gpus, prompt_len=plen, gen_len=args.bench_gen_len)
            except Exception as ex:
                print(f"[warn] prompt sweep {plen} failed; skipping. Reason: {ex}", file=sys.stderr, flush=True)

    # Optional llama.cpp baseline (best-effort; does not fail the report).
    if args.include_llama:
        try:
            if not args.llama_bin_dir.strip():
                raise ValueError("--llama-bin-dir is required with --include-llama")
            llama_bin_dir = Path(args.llama_bin_dir).expanduser().resolve()
            if not llama_bin_dir.exists():
                raise FileNotFoundError(f"llama bin dir does not exist: {llama_bin_dir}")

            if args.llama_gguf_bf16.strip().lower() == "auto":
                if hub_root is None:
                    raise ValueError("--llama-gguf-bf16=auto requires --hub-root")
                gguf_bf16 = resolve_gguf_in_hub(hub_root, "Qwen3-4B-BF16")
            else:
                gguf_bf16 = Path(args.llama_gguf_bf16).expanduser().resolve()

            dev_ids = llama_list_devices(repo, logs_dir, args.progress, llama_bin_dir)
            if not dev_ids:
                raise RuntimeError("llama-bench reports no available devices (CUDA init failed?); skip llama baseline")

            dev0 = pick_llama_device(dev_ids, 0)
            dev1 = pick_llama_device(dev_ids, 1) if len(dev_ids) >= 2 else ""

            # Single GPU baseline.
            pre_ms, pre_tps, dec_ms, dec_tps = run_llama_bench(repo,
                                                               logs_dir,
                                                               args.progress,
                                                               llama_bin_dir,
                                                               gguf_bf16,
                                                               prompt_len=args.bench_prompt_len,
                                                               gen_len=args.bench_gen_len,
                                                               device=dev0,
                                                               split_mode="none",
                                                               tensor_split="")
            extra_rows.append([
                "E_llama:bf16_single",
                gguf_bf16.name,
                dev0,
                "",
                str(args.bench_prompt_len),
                str(args.bench_gen_len),
                "1",
                "",
                f"{pre_ms:.3f}",
                f"{dec_ms:.3f}",
                f"{dec_tps:.3f}",
                f"llama-bench BF16 (GGUF); prompt_tok_s={pre_tps:.3f}",
            ])

            # Dual GPU baseline.
            if len(gpus) >= 2 and dev1:
                pre_ms, pre_tps, dec_ms, dec_tps = run_llama_bench(repo,
                                                                   logs_dir,
                                                                   args.progress,
                                                                   llama_bin_dir,
                                                                   gguf_bf16,
                                                                   prompt_len=args.bench_prompt_len,
                                                                   gen_len=args.bench_gen_len,
                                                                   device=f"{dev0}/{dev1}",
                                                                   split_mode=args.llama_dual_split_mode,
                                                                   tensor_split=args.llama_tensor_split)
                extra_rows.append([
                    "E_llama:bf16_dual",
                    gguf_bf16.name,
                    f"{dev0}/{dev1}",
                    f"{args.llama_dual_split_mode} ts={args.llama_tensor_split}",
                    str(args.bench_prompt_len),
                    str(args.bench_gen_len),
                    "1",
                    "",
                    f"{pre_ms:.3f}",
                    f"{dec_ms:.3f}",
                    f"{dec_tps:.3f}",
                    f"llama-bench BF16 (GGUF); prompt_tok_s={pre_tps:.3f}",
                ])

            # Optional quantized baseline (not apples-to-apples).
            if args.include_llama_quantized:
                if args.llama_gguf_q4.strip().lower() == "auto":
                    if hub_root is None:
                        raise ValueError("--llama-gguf-q4=auto requires --hub-root")
                    try:
                        gguf_q4 = resolve_gguf_in_hub(hub_root, "Qwen3-4B-Q4_K_M")
                    except Exception:
                        gguf_q4 = resolve_gguf_in_hub(hub_root, "Qwen3-4B-Q4_0")
                else:
                    gguf_q4 = Path(args.llama_gguf_q4).expanduser().resolve()

                pre_ms, pre_tps, dec_ms, dec_tps = run_llama_bench(repo,
                                                                   logs_dir,
                                                                   args.progress,
                                                                   llama_bin_dir,
                                                                   gguf_q4,
                                                                   prompt_len=args.bench_prompt_len,
                                                                   gen_len=args.bench_gen_len,
                                                                   device=dev0,
                                                                   split_mode="none",
                                                                   tensor_split="")
                extra_rows.append([
                    "E_llama:q4_single",
                    gguf_q4.name,
                    dev0,
                    "",
                    str(args.bench_prompt_len),
                    str(args.bench_gen_len),
                    "1",
                    "",
                    f"{pre_ms:.3f}",
                    f"{dec_ms:.3f}",
                    f"{dec_tps:.3f}",
                    f"llama-bench quantized (not apples-to-apples); prompt_tok_s={pre_tps:.3f}",
                ])
        except Exception as ex:
            print(f"[warn] llama.cpp baseline failed; skipping. Reason: {ex}", file=sys.stderr, flush=True)

    # Write extra_results.csv by filtering rows_out for phases starting with "E".
    for r in rows_out:
        ph = r[0]
        if not ph.startswith("E"):
            continue
        # rows_out: phase,policy,mode,model,prompt_len,gen_len,chunk_len,batch_size,ttft_ms,prefill_ms,decode_ms,decode_tok_s,split,gpus
        extra_rows.append([
            ph + ":" + r[1],
            r[3],
            r[13],
            r[12],
            r[4],
            r[5],
            r[7],
            r[8],
            r[9],
            r[10],
            r[11],
            "from ember_benchmark",
        ])

    with extra_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(extra_header)
        for r in extra_rows:
            w.writerow(r)

    # Render Markdown report.
    md_path = out_dir / "report.md"
    md = []
    md.append(f"# Ember Report ({name})\n")
    md.append("## System\n")
    if sysinfo_md:
        md.extend(sysinfo_md)
    else:
        md.append("_(nvidia-smi not available)_\n")

    md.append("## Artifacts\n")
    md.append(f"- `A_phase_analysis.csv`\n")
    md.append(f"- `A_p2p_bandwidth.csv`\n")
    md.append(f"- `A_pipeline_bubble.csv`\n")
    md.append(f"- `benchmark_results.csv`\n")
    md.append(f"- `serve_results.csv`\n")
    md.append(f"- `extra_results.csv`\n")
    md.append(f"- `logs/`\n")

    md.append("\n## Models\n")
    md.append(f"- A: `{str(model_a) if model_a is not None else ''}` (split `{split_a_chosen or 'auto'}`)\n")
    md.append(f"- B/C: `{str(model_b) if model_b is not None else ''}` (split `{split_b_chosen or 'auto'}`)\n")
    md.append(f"- D: `{str(model_d) if model_d is not None else ''}` (split `{split_d_chosen or 'auto'}`; fallback_to_B `{1 if args.fallback_d_to_b else 0}`)\n")
    md.append("\n## Notes\n")
    md.append("- 不同模型的层数不同（例如 8B=36 layers，14B=40 layers），所以 pipeline split 必须按各自层数计算；脚本会为 A/B/C/D 分别选择 split（优先使用 --split-<phase>，否则自动均分）。\n")
    md.append("- `model-d` 如果太大导致 OOM，默认会自动回退到 `model-b`（可用 `--no-fallback-d-to-b` 关闭），仍然生成完整报告。\n")
    md.append("- C 阶段默认用 `ember_serve_benchmark`（continuous batching，PhaseAwareBatchScheduler），避免 `ember_benchmark` 在 batch>1 + 长 prompt 下因为 full-seq attention（显存随 `batch * prompt_len^2` 增长）而 OOM。可用 `--c-source benchmark`/`both` 复现旧路径。\n")
    md.append("- 如果启用 `--include-llama`，脚本会调用 `llama-bench` 生成 llama.cpp 的 BF16 GGUF 基线并写入 Extras；`--include-llama-quantized` 会额外跑量化 GGUF（不与 FP16 严格可比，仅作实用参考）。\n")

    # A summaries
    md.append("## A. Motivation Study\n")
    if a_phase_csv.exists():
        phase_rows = parse_csv(a_phase_csv)
        # Summarize by prompt_len: average over layers of prefill/decode ms.
        by_plen: Dict[int, List[Tuple[float, float]]] = {}
        for r in phase_rows:
            try:
                plen = int(r["prompt_len"])
                pre = float(r["prefill_time_ms"])
                dec = float(r["decode_time_ms"])
            except Exception:
                continue
            by_plen.setdefault(plen, []).append((pre, dec))
        rows = []
        for plen in sorted(by_plen.keys()):
            pre_avg = sum(x[0] for x in by_plen[plen]) / len(by_plen[plen])
            dec_avg = sum(x[1] for x in by_plen[plen]) / len(by_plen[plen])
            rows.append([str(plen), f"{pre_avg:.4f}", f"{dec_avg:.4f}"])
        md.append("### A.1 Prefill vs Decode（逐层平均耗时）\n")
        md.append(format_md_table(["prompt_len", "avg_prefill_layer_ms", "avg_decode_layer_ms"], rows))
    else:
        md.append("### A.1 Prefill vs Decode\n_(skipped)_\n")

    if a_p2p_csv.exists():
        p2p_rows = parse_csv(a_p2p_csv)
        # Summarize cudaMemcpyPeerAsync only.
        rows = []
        for r in p2p_rows:
            if r.get("method") != "cudaMemcpyPeerAsync":
                continue
            rows.append([r["data_size_bytes"], r["transfer_time_us"], r["bandwidth_gbps"], r["direction"]])
        md.append("### A.2 PCIe P2P Bandwidth（cudaMemcpyPeerAsync）\n")
        md.append(format_md_table(["bytes", "us", "GB/s", "direction"], rows[:24]))
        if len(rows) > 24:
            md.append(f"_(truncated; see `A_p2p_bandwidth.csv` for full data)_\n")
    else:
        md.append("### A.2 PCIe P2P Bandwidth\n_(skipped)_\n")

    if a_bubble_csv.exists():
        bubble_rows = parse_csv(a_bubble_csv)
        # Show compact table: prefill prompt=1024/2048 and decode prompt=1024
        show = []
        for r in bubble_rows:
            if r["phase"] == "prefill" and r["prompt_len"] in {"1024", "2048"}:
                show.append([r["phase"], r["prompt_len"], r["micro_batches"], f"{float(r['bubble_ratio'])*100.0:.2f}%"])
            if r["phase"] == "decode" and r["prompt_len"] in {"1024"}:
                show.append([r["phase"], r["prompt_len"], r["micro_batches"], f"{float(r['bubble_ratio'])*100.0:.2f}%"])
        md.append("### A.3 Naive Pipeline Bubble（简化 2-stage 模型，估算）\n")
        md.append(format_md_table(["phase", "prompt_len", "micro_batches", "bubble_ratio"], show))
    else:
        md.append("### A.3 Naive Pipeline Bubble\n_(skipped)_\n")

    # B/C/D summaries
    md.append("## B/C/D. Benchmarks\n")
    bench_rows = parse_csv(results_csv) if results_csv.exists() else []
    rows = []
    for r in bench_rows:
        rows.append([
            r["phase"],
            r["model"],
            r["policy"],
            r["mode"],
            r["prompt_len"],
            r["gen_len"],
            r["batch_size"],
            r.get("ttft_ms", ""),
            r["prefill_ms"],
            r["decode_ms"],
            r["decode_tok_s"],
            r["split"],
        ])
    md.append(format_md_table(["phase", "model", "policy", "mode", "prompt", "gen", "batch", "ttft_ms", "prefill_ms", "decode_ms", "decode_tok_s", "split"], rows))

    md.append("## C. Phase-Aware Decode (Continuous Batching)\n")
    serve_rows_md = parse_csv(serve_csv) if serve_csv.exists() else []
    srows = []
    for r in serve_rows_md:
        srows.append([
            r["model"],
            r["batch_size"],
            r["num_reqs"],
            r["prompt_len"],
            r["gen_len"],
            r["vary_gen"],
            r["prefill_ms"],
            r["decode_ms"],
            r["gen_tokens"],
            r["decode_tok_s"],
            r["split"],
        ])
    md.append(format_md_table(["model", "batch", "num_reqs", "prompt", "gen", "vary", "prefill_ms", "decode_ms", "gen_tokens", "decode_tok_s", "split"], srows))

    md.append("## E. Extras\n")
    if extra_csv.exists():
        extra_rows_md = parse_csv(extra_csv)
        erows = []
        for r in extra_rows_md:
            erows.append([
                r["name"],
                r["model"],
                r["gpus"],
                r["split"],
                r["prompt_len"],
                r["gen_len"],
                r["batch_size"],
                r.get("ttft_ms", ""),
                r["prefill_ms"],
                r["decode_ms"],
                r["decode_tok_s"],
                r["note"],
            ])
        md.append(format_md_table(["name", "model", "gpus", "split", "prompt", "gen", "batch", "ttft_ms", "prefill_ms", "decode_ms", "tok/s", "note"], erows))
    else:
        md.append("_(no extra experiments)_\n")

    md_path.write_text("\n".join(md), encoding="utf-8")

    print(str(md_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

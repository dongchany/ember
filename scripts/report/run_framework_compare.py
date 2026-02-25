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


def safe_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def run_cmd(
    cmd: List[str],
    cwd: Path,
    log_path: Path,
    env: Optional[Dict[str, str]] = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        p = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, env=merged_env)
        f.write(p.stdout)
        if p.stderr:
            f.write("\n[stderr]\n")
            f.write(p.stderr)
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed rc={p.returncode}: {' '.join(cmd)} (see {log_path})")
    return p


def run_cmd_json(
    cmd: List[str],
    cwd: Path,
    log_path: Path,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    p = run_cmd(cmd=cmd, cwd=cwd, log_path=log_path, env=env, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"command failed rc={p.returncode}: {' '.join(cmd)} (see {log_path})")
    payload = (p.stdout or "").strip()
    if not payload:
        payload = "{}"
    # Benchmark scripts may emit framework logs around the final JSON payload.
    # Extract the last parseable JSON object from mixed stdout.
    dec = json.JSONDecoder()
    best_obj: Optional[Dict[str, object]] = None
    if payload and not payload.startswith("{"):
        starts = [i for i, ch in enumerate(payload) if ch == "{"]
        for i in reversed(starts):
            try:
                obj, _ = dec.raw_decode(payload[i:])
            except Exception:
                continue
            if isinstance(obj, dict):
                best_obj = obj
                break
    try:
        if best_obj is not None:
            return best_obj
        return json.loads(payload)
    except Exception as ex:
        raise RuntimeError(f"failed to parse JSON output from {' '.join(cmd)}: {ex}") from ex


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_csv(path: Path) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.append(r)
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


def extract_json_array_from_mixed_output(text: str) -> List[Dict[str, object]]:
    lb = text.find("[")
    rb = text.rfind("]")
    if lb < 0 or rb < 0 or rb <= lb:
        raise ValueError("failed to locate JSON payload")
    payload = text[lb:rb + 1]
    data = json.loads(payload)
    if not isinstance(data, list):
        raise ValueError("unexpected JSON payload (non-list)")
    return data


def parse_stage_breakdown_csv(csv_path: Path, decode_steps: int) -> Tuple[float, float, float, float]:
    rows = read_csv(csv_path)
    by_phase: Dict[str, Dict[str, str]] = {}
    for r in rows:
        by_phase[r.get("phase", "")] = r
    pre = by_phase.get("prefill")
    dec = by_phase.get("decode_per_token")
    if pre is None or dec is None:
        raise RuntimeError(f"missing prefill/decode_per_token rows in {csv_path}")
    prefill_ms = safe_float(pre.get("wall_ms", "0"))
    decode_per_token_ms = safe_float(dec.get("wall_ms", "0"))
    rollout_total_ms = prefill_ms + decode_per_token_ms * float(decode_steps)
    rollout_tok_s = (float(decode_steps) * 1000.0 / rollout_total_ms) if rollout_total_ms > 0.0 else 0.0
    return prefill_ms, decode_per_token_ms, rollout_total_ms, rollout_tok_s


def check_python_module(python_bin: str, module_name: str) -> bool:
    p = subprocess.run(
        [
            python_bin,
            "-c",
            (
                "import importlib.util,sys; "
                f"sys.exit(0 if importlib.util.find_spec('{module_name}') else 1)"
            ),
        ],
        text=True,
        capture_output=True,
    )
    return p.returncode == 0


def run_ember_case(
    repo: Path,
    logs_dir: Path,
    stage_bin: Path,
    model_dir: Path,
    prompt_len: int,
    decode_steps: int,
    iters: int,
    warmup: int,
    chunk_len: int,
    gpus: str,
    split: str,
    overlap: bool,
    out_csv: Path,
) -> Tuple[float, float, float, float]:
    cmd = [
        str(stage_bin),
        "--model",
        str(model_dir),
        "--gpus",
        gpus,
        "--prompt-len",
        str(prompt_len),
        "--decode-steps",
        str(decode_steps),
        "--chunk-len",
        str(chunk_len),
        "--iters",
        str(iters),
        "--warmup",
        str(warmup),
        "--csv",
        str(out_csv),
        "--decode-with-sampling",
    ]
    if split.strip():
        cmd += ["--split", split]
    if overlap:
        cmd += ["--overlap"]
    else:
        cmd += ["--no-overlap"]
    if "," in gpus:
        cmd += ["--pipeline"]

    run_cmd(cmd, cwd=repo, log_path=logs_dir / f"ember_{gpus.replace(',', '_')}_{'ov' if overlap else 'noov'}.log")
    return parse_stage_breakdown_csv(out_csv, decode_steps=decode_steps)


def run_llama_case(
    repo: Path,
    logs_dir: Path,
    llama_bin_dir: Path,
    gguf_path: Path,
    prompt_len: int,
    decode_steps: int,
    device: str,
    split_mode: str,
) -> Tuple[float, float, float, float]:
    llama_bench = llama_bin_dir / "llama-bench"
    if not llama_bench.exists():
        raise FileNotFoundError(f"missing llama-bench: {llama_bench}")
    if not gguf_path.exists():
        raise FileNotFoundError(f"missing gguf: {gguf_path}")

    cmd = [
        str(llama_bench),
        "-m",
        str(gguf_path),
        "-p",
        str(prompt_len),
        "-n",
        str(decode_steps),
        "-r",
        "1",
        "-ngl",
        "999",
        "-o",
        "json",
        "-dev",
        device,
        "-sm",
        split_mode,
    ]

    env = os.environ.copy()
    ld = env.get("LD_LIBRARY_PATH", "")
    prefix = str(llama_bin_dir)
    env["LD_LIBRARY_PATH"] = f"{prefix}:{ld}" if ld else prefix

    p = run_cmd(
        cmd,
        cwd=repo,
        log_path=logs_dir / f"llama_{device.replace('/', '_')}.log",
        env=env,
        check=False,
    )
    if p.returncode != 0:
        raise RuntimeError(f"llama-bench failed rc={p.returncode}")

    arr = extract_json_array_from_mixed_output((p.stdout or "") + "\n" + (p.stderr or ""))
    pre = None
    dec = None
    for item in arr:
        n_prompt = int(item.get("n_prompt", 0))
        n_gen = int(item.get("n_gen", 0))
        if n_prompt > 0 and n_gen == 0:
            pre = item
        if n_prompt == 0 and n_gen > 0:
            dec = item
    if pre is None or dec is None:
        raise RuntimeError("llama-bench output missing prompt/gen entries")

    prefill_ms = float(pre.get("avg_ns", 0.0)) / 1e6
    decode_total_ms = float(dec.get("avg_ns", 0.0)) / 1e6
    decode_per_token_ms = decode_total_ms / float(decode_steps) if decode_steps > 0 else 0.0
    rollout_total_ms = prefill_ms + decode_total_ms
    rollout_tok_s = (float(decode_steps) * 1000.0 / rollout_total_ms) if rollout_total_ms > 0.0 else 0.0
    return prefill_ms, decode_per_token_ms, rollout_total_ms, rollout_tok_s


def run_transformers_case(
    repo: Path,
    logs_dir: Path,
    python_bin: str,
    model_dir: Path,
    prompt_len: int,
    decode_steps: int,
    iters: int,
    warmup: int,
) -> Tuple[float, float, float, float]:
    bench_script = repo / "scripts" / "report" / "bench_transformers_rollout.py"
    if not bench_script.exists():
        raise FileNotFoundError(f"missing script: {bench_script}")
    cmd = [
        python_bin,
        str(bench_script),
        "--model",
        str(model_dir),
        "--prompt-len",
        str(prompt_len),
        "--decode-steps",
        str(decode_steps),
        "--device",
        "cuda:0",
        "--dtype",
        "float16",
        "--iters",
        str(iters),
        "--warmup",
        str(warmup),
    ]
    out = run_cmd_json(
        cmd=cmd,
        cwd=repo,
        log_path=logs_dir / "transformers.log",
    )
    return (
        float(out.get("prefill_ms", 0.0)),
        float(out.get("decode_per_token_ms", 0.0)),
        float(out.get("rollout_total_ms", 0.0)),
        float(out.get("rollout_tok_s", 0.0)),
    )


def run_vllm_case(
    repo: Path,
    logs_dir: Path,
    python_bin: str,
    model_dir: Path,
    prompt_len: int,
    decode_steps: int,
    iters: int,
    warmup: int,
    tp_size: int,
    gpu_memory_utilization: float,
    max_num_seqs: int,
    enforce_eager: bool,
) -> Tuple[float, float, float, float]:
    bench_script = repo / "scripts" / "report" / "bench_vllm_rollout.py"
    if not bench_script.exists():
        raise FileNotFoundError(f"missing script: {bench_script}")
    cmd = [
        python_bin,
        str(bench_script),
        "--model",
        str(model_dir),
        "--prompt-len",
        str(prompt_len),
        "--decode-steps",
        str(decode_steps),
        "--dtype",
        "float16",
        "--tensor-parallel-size",
        str(tp_size),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-num-seqs",
        str(max_num_seqs),
        "--iters",
        str(iters),
        "--warmup",
        str(warmup),
    ]
    if enforce_eager:
        cmd.append("--enforce-eager")
    out = run_cmd_json(
        cmd=cmd,
        cwd=repo,
        log_path=logs_dir / "vllm.log",
    )
    return (
        float(out.get("prefill_ms", 0.0)),
        float(out.get("decode_per_token_ms", 0.0)),
        float(out.get("rollout_total_ms", 0.0)),
        float(out.get("rollout_tok_s", 0.0)),
    )


def run_sglang_case(
    repo: Path,
    logs_dir: Path,
    python_bin: str,
    model_dir: Path,
    prompt_len: int,
    decode_steps: int,
    iters: int,
    warmup: int,
    tp_size: int,
) -> Tuple[float, float, float, float]:
    bench_script = repo / "scripts" / "report" / "bench_sglang_rollout.py"
    if not bench_script.exists():
        raise FileNotFoundError(f"missing script: {bench_script}")
    cmd = [
        python_bin,
        str(bench_script),
        "--model",
        str(model_dir),
        "--prompt-len",
        str(prompt_len),
        "--decode-steps",
        str(decode_steps),
        "--dtype",
        "float16",
        "--tp-size",
        str(tp_size),
        "--iters",
        str(iters),
        "--warmup",
        str(warmup),
    ]
    out = run_cmd_json(
        cmd=cmd,
        cwd=repo,
        log_path=logs_dir / "sglang.log",
    )
    return (
        float(out.get("prefill_ms", 0.0)),
        float(out.get("decode_per_token_ms", 0.0)),
        float(out.get("rollout_total_ms", 0.0)),
        float(out.get("rollout_tok_s", 0.0)),
    )


def build_row(
    engine: str,
    scenario: str,
    status: str,
    prefill_ms: float = 0.0,
    decode_per_token_ms: float = 0.0,
    rollout_total_ms: float = 0.0,
    rollout_tok_s: float = 0.0,
    notes: str = "",
) -> Dict[str, str]:
    return {
        "engine": engine,
        "scenario": scenario,
        "status": status,
        "prefill_ms": f"{prefill_ms:.3f}" if status == "ok" else "",
        "decode_per_token_ms": f"{decode_per_token_ms:.3f}" if status == "ok" else "",
        "rollout_total_ms": f"{rollout_total_ms:.3f}" if status == "ok" else "",
        "rollout_tok_s": f"{rollout_tok_s:.3f}" if status == "ok" else "",
        "notes": notes,
    }


def write_markdown(
    path: Path,
    rows: List[Dict[str, str]],
    model_dir: Path,
    prompt_len: int,
    decode_steps: int,
) -> None:
    lines: List[str] = []
    lines.append("# Framework Compare")
    lines.append("")
    lines.append(f"- Model: `{model_dir}`")
    lines.append(f"- Setting: prompt_len={prompt_len}, decode_steps={decode_steps}")
    lines.append(f"- Generated at: `{dt.datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")
    lines.append("| engine | scenario | status | prefill_ms | decode_ms/token | rollout_total_ms | rollout_tok/s | notes |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        lines.append(
            "| " + " | ".join(
                [
                    r["engine"],
                    r["scenario"],
                    r["status"],
                    r["prefill_ms"],
                    r["decode_per_token_ms"],
                    r["rollout_total_ms"],
                    r["rollout_tok_s"],
                    r["notes"],
                ]
            ) + " |"
        )

    ok_rows = [r for r in rows if r["status"] == "ok"]
    if ok_rows:
        best = max(ok_rows, key=lambda x: safe_float(x.get("rollout_tok_s", "0")))
        lines.append("")
        lines.append("## Key Point")
        lines.append(
            f"- Best measured rollout throughput: `{best['engine']} / {best['scenario']}` "
            f"at `{best['rollout_tok_s']} tok/s`."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run unified Ember vs other-framework rollout benchmark.")
    ap.add_argument("--model", type=str, required=True, help="model path or HF model id")
    ap.add_argument("--prompt-len", type=int, default=2048)
    ap.add_argument("--decode-steps", type=int, default=128)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--chunk-len", type=int, default=512)
    ap.add_argument("--build-dir", type=str, default="build")
    ap.add_argument("--single-gpu", type=str, default="0")
    ap.add_argument("--dual-gpus", type=str, default="0,1")
    ap.add_argument("--dual-split", type=str, default="9,27")
    ap.add_argument("--dual-overlap", action="store_true", default=True)
    ap.add_argument("--dual-no-overlap", dest="dual_overlap", action="store_false")
    ap.add_argument("--include-llama", action="store_true", default=True)
    ap.add_argument("--no-include-llama", dest="include_llama", action="store_false")
    ap.add_argument("--llama-bin-dir", type=str, default="/home/dong/workspace/llama.cpp/build/bin")
    ap.add_argument("--gguf", type=str, default="reports/gguf/Qwen3-4B-BF16.gguf")
    ap.add_argument("--python-bin", type=str, default="python3")
    ap.add_argument("--transformers-python-bin", type=str, default="")
    ap.add_argument("--vllm-python-bin", type=str, default="")
    ap.add_argument("--sglang-python-bin", type=str, default="")
    ap.add_argument("--vllm-tp-size", type=int, default=1)
    ap.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.8)
    ap.add_argument("--vllm-max-num-seqs", type=int, default=32)
    ap.add_argument("--vllm-enforce-eager", action="store_true", default=False)
    ap.add_argument("--sglang-tp-size", type=int, default=1)
    ap.add_argument("--include-vllm", action="store_true", default=True)
    ap.add_argument("--no-include-vllm", dest="include_vllm", action="store_false")
    ap.add_argument("--include-sglang", action="store_true", default=True)
    ap.add_argument("--no-include-sglang", dest="include_sglang", action="store_false")
    ap.add_argument("--include-transformers", action="store_true", default=True)
    ap.add_argument("--no-include-transformers", dest="include_transformers", action="store_false")
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    if args.prompt_len <= 0 or args.decode_steps <= 0:
        die("--prompt-len and --decode-steps must be > 0")

    repo = Path.cwd()
    model_dir = resolve_model_dir(args.model)
    stage_bin = (repo / args.build_dir / "ember_stage_breakdown").resolve()
    if not stage_bin.exists():
        die(f"missing binary: {stage_bin}")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (repo / "reports" / f"framework_compare_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []

    # Ember single GPU
    try:
        single_csv = out_dir / "ember_single_stage.csv"
        pms, dms, tms, tps = run_ember_case(
            repo=repo,
            logs_dir=logs_dir,
            stage_bin=stage_bin,
            model_dir=model_dir,
            prompt_len=args.prompt_len,
            decode_steps=args.decode_steps,
            iters=args.iters,
            warmup=args.warmup,
            chunk_len=args.chunk_len,
            gpus=args.single_gpu,
            split="",
            overlap=False,
            out_csv=single_csv,
        )
        rows.append(build_row("ember", f"single({args.single_gpu})", "ok", pms, dms, tms, tps, "ember_stage_breakdown"))
    except Exception as ex:
        rows.append(build_row("ember", f"single({args.single_gpu})", "error", notes=str(ex)))

    # Ember dual GPU
    try:
        dual_csv = out_dir / "ember_dual_stage.csv"
        pms, dms, tms, tps = run_ember_case(
            repo=repo,
            logs_dir=logs_dir,
            stage_bin=stage_bin,
            model_dir=model_dir,
            prompt_len=args.prompt_len,
            decode_steps=args.decode_steps,
            iters=args.iters,
            warmup=args.warmup,
            chunk_len=args.chunk_len,
            gpus=args.dual_gpus,
            split=args.dual_split,
            overlap=args.dual_overlap,
            out_csv=dual_csv,
        )
        mode_note = "overlap" if args.dual_overlap else "no_overlap"
        rows.append(build_row("ember", f"dual({args.dual_gpus})", "ok", pms, dms, tms, tps, f"split={args.dual_split} {mode_note}"))
    except Exception as ex:
        rows.append(build_row("ember", f"dual({args.dual_gpus})", "error", notes=str(ex)))

    # llama.cpp
    if args.include_llama:
        try:
            llama_bin_dir = Path(args.llama_bin_dir).expanduser().resolve()
            gguf_path = Path(args.gguf).expanduser().resolve()
            pms, dms, tms, tps = run_llama_case(
                repo=repo,
                logs_dir=logs_dir,
                llama_bin_dir=llama_bin_dir,
                gguf_path=gguf_path,
                prompt_len=args.prompt_len,
                decode_steps=args.decode_steps,
                device="CUDA0",
                split_mode="layer",
            )
            rows.append(build_row("llama.cpp", "single(CUDA0)", "ok", pms, dms, tms, tps, "llama-bench"))
        except Exception as ex:
            rows.append(build_row("llama.cpp", "single(CUDA0)", "error", notes=str(ex)))

        try:
            llama_bin_dir = Path(args.llama_bin_dir).expanduser().resolve()
            gguf_path = Path(args.gguf).expanduser().resolve()
            pms, dms, tms, tps = run_llama_case(
                repo=repo,
                logs_dir=logs_dir,
                llama_bin_dir=llama_bin_dir,
                gguf_path=gguf_path,
                prompt_len=args.prompt_len,
                decode_steps=args.decode_steps,
                device="CUDA0/CUDA1",
                split_mode="layer",
            )
            rows.append(build_row("llama.cpp", "dual(CUDA0/CUDA1)", "ok", pms, dms, tms, tps, "llama-bench"))
        except Exception as ex:
            rows.append(build_row("llama.cpp", "dual(CUDA0/CUDA1)", "error", notes=str(ex)))

    tf_python = args.transformers_python_bin.strip() or args.python_bin
    vllm_python = args.vllm_python_bin.strip() or args.python_bin
    sglang_python = args.sglang_python_bin.strip() or args.python_bin

    # vLLM availability marker
    if args.include_vllm:
        has_vllm = check_python_module(vllm_python, "vllm")
        if has_vllm:
            try:
                pms, dms, tms, tps = run_vllm_case(
                    repo=repo,
                    logs_dir=logs_dir,
                    python_bin=vllm_python,
                    model_dir=model_dir,
                    prompt_len=args.prompt_len,
                    decode_steps=args.decode_steps,
                    iters=args.iters,
                    warmup=args.warmup,
                    tp_size=args.vllm_tp_size,
                    gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                    max_num_seqs=args.vllm_max_num_seqs,
                    enforce_eager=args.vllm_enforce_eager,
                )
                rows.append(
                    build_row(
                        "vllm",
                        f"single(tp={args.vllm_tp_size})",
                        "ok",
                        pms,
                        dms,
                        tms,
                        tps,
                        notes=f"python={vllm_python}",
                    )
                )
            except Exception as ex:
                rows.append(
                    build_row(
                        "vllm",
                        f"single(tp={args.vllm_tp_size})",
                        "error",
                        notes=str(ex),
                    )
                )
        else:
            rows.append(build_row("vllm", "env-check", "skipped", notes=f"python module vllm not installed in {vllm_python}"))

    # SGLang availability marker
    if args.include_sglang:
        has_sglang = check_python_module(sglang_python, "sglang")
        if has_sglang:
            try:
                pms, dms, tms, tps = run_sglang_case(
                    repo=repo,
                    logs_dir=logs_dir,
                    python_bin=sglang_python,
                    model_dir=model_dir,
                    prompt_len=args.prompt_len,
                    decode_steps=args.decode_steps,
                    iters=args.iters,
                    warmup=args.warmup,
                    tp_size=args.sglang_tp_size,
                )
                rows.append(
                    build_row(
                        "sglang",
                        f"single(tp={args.sglang_tp_size})",
                        "ok",
                        pms,
                        dms,
                        tms,
                        tps,
                        notes=f"python={sglang_python}",
                    )
                )
            except Exception as ex:
                rows.append(
                    build_row(
                        "sglang",
                        f"single(tp={args.sglang_tp_size})",
                        "error",
                        notes=str(ex),
                    )
                )
        else:
            rows.append(build_row("sglang", "env-check", "skipped", notes=f"python module sglang not installed in {sglang_python}"))

    # HF transformers benchmark
    if args.include_transformers:
        has_torch = check_python_module(tf_python, "torch")
        has_tf = check_python_module(tf_python, "transformers")
        if has_torch and has_tf:
            try:
                pms, dms, tms, tps = run_transformers_case(
                    repo=repo,
                    logs_dir=logs_dir,
                    python_bin=tf_python,
                    model_dir=model_dir,
                    prompt_len=args.prompt_len,
                    decode_steps=args.decode_steps,
                    iters=args.iters,
                    warmup=args.warmup,
                )
                rows.append(
                    build_row(
                        "transformers",
                        "single(cuda:0)",
                        "ok",
                        pms,
                        dms,
                        tms,
                        tps,
                        notes=f"python={tf_python}",
                    )
                )
            except Exception as ex:
                rows.append(
                    build_row(
                        "transformers",
                        "single(cuda:0)",
                        "error",
                        notes=str(ex),
                    )
                )
        else:
            missing = []
            if not has_torch:
                missing.append("torch")
            if not has_tf:
                missing.append("transformers")
            rows.append(
                build_row(
                    "transformers",
                    "env-check",
                    "skipped",
                    notes=f"missing python module(s) in {tf_python}: {', '.join(missing)}",
                )
            )

    out_csv = out_dir / "framework_compare.csv"
    out_md = out_dir / "framework_compare.md"
    fieldnames = [
        "engine",
        "scenario",
        "status",
        "prefill_ms",
        "decode_per_token_ms",
        "rollout_total_ms",
        "rollout_tok_s",
        "notes",
    ]
    write_csv(out_csv, rows, fieldnames=fieldnames)
    write_markdown(out_md, rows, model_dir=model_dir, prompt_len=args.prompt_len, decode_steps=args.decode_steps)

    print("[done] framework comparison finished")
    print(f"- csv: {out_csv}")
    print(f"- md: {out_md}")


if __name__ == "__main__":
    main()

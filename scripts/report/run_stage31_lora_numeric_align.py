#!/usr/bin/env python3
import argparse
import array
import csv
import datetime as dt
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM


PAIR_RE = re.compile(
    r"layers\.([0-9]+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.lora_([AB])(?:\.default)?\.weight$"
)


def die(msg: str) -> None:
    raise SystemExit(f"error: {msg}")


def run_cmd(cmd: List[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        if p.stdout:
            f.write(p.stdout)
        if p.stderr:
            f.write("\n[stderr]\n")
            f.write(p.stderr)
    if p.returncode != 0:
        die(f"command failed rc={p.returncode}: {' '.join(cmd)} (see {log_path})")


def read_tokens(path: Path) -> List[int]:
    s = path.read_text(encoding="utf-8").strip()
    if not s:
        return []
    return [int(x) for x in s.split()]


def read_meta(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_logits(path: Path, vocab_size: int) -> torch.Tensor:
    arr = array.array("f")
    with path.open("rb") as f:
        arr.fromfile(f, vocab_size)
    if len(arr) != vocab_size:
        die(f"logits size mismatch in {path}: got {len(arr)}, expected {vocab_size}")
    return torch.tensor(arr, dtype=torch.float32)


def read_alpha_over_r(adapter_dir: Path) -> float:
    cfg = adapter_dir / "adapter_config.json"
    if not cfg.exists():
        return 1.0
    obj = json.loads(cfg.read_text(encoding="utf-8"))
    alpha = float(obj.get("lora_alpha", 0.0) or 0.0)
    r = float(obj.get("r", 0.0) or 0.0)
    if alpha <= 0.0 or r <= 0.0:
        return 1.0
    return alpha / r


def apply_lora_inplace_hf(model: AutoModelForCausalLM, adapter_dir: Path, lora_scale: float) -> int:
    safepath = adapter_dir / "adapter_model.safetensors"
    if not safepath.exists():
        die(f"missing adapter_model.safetensors: {safepath}")
    tensors = load_file(str(safepath), device="cpu")
    alpha_over_r = read_alpha_over_r(adapter_dir)
    effective_scale = lora_scale * alpha_over_r

    pairs: Dict[Tuple[int, str], Dict[str, torch.Tensor]] = {}
    for name, t in tensors.items():
        m = PAIR_RE.search(name)
        if not m:
            continue
        layer_idx = int(m.group(1))
        proj = m.group(2)
        ab = m.group(3)
        key = (layer_idx, proj)
        if key not in pairs:
            pairs[key] = {}
        pairs[key][ab] = t

    updated = 0
    for (layer_idx, proj), d in pairs.items():
        if "A" not in d or "B" not in d:
            continue
        a = d["A"]  # [r, in]
        b = d["B"]  # [out, r]
        if layer_idx < 0 or layer_idx >= len(model.model.layers):
            continue
        layer = model.model.layers[layer_idx]
        mod = getattr(layer.self_attn, proj, None)
        if mod is None or not hasattr(mod, "weight"):
            continue
        w = mod.weight
        if a.dim() != 2 or b.dim() != 2:
            continue
        if a.shape[1] != w.shape[1] or b.shape[0] != w.shape[0] or b.shape[1] != a.shape[0]:
            continue
        delta = torch.matmul(b.float(), a.float()) * float(effective_scale)
        delta = delta.to(device=w.device, dtype=w.dtype)
        with torch.no_grad():
            w.add_(delta)
        updated += 1
    if updated == 0:
        die(f"no supported LoRA A/B pairs applied from {adapter_dir}")
    return updated


def hf_last_logits(
    model_dir: Path,
    tokens: List[int],
    device: str,
    dtype: str,
) -> torch.Tensor:
    if not tokens:
        die("empty token list for HF inference")
    if dtype == "float16":
        tdtype = torch.float16
    elif dtype == "bfloat16":
        tdtype = torch.bfloat16
    else:
        tdtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=tdtype,
    )
    model.eval()
    model.to(device)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(input_ids, use_cache=False)
        logits_base = out.logits[0, -1].float().cpu()
    return model, logits_base


def compute_diff(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    d = (a - b).abs()
    return float(d.max().item()), float(d.mean().item())


def write_csv(path: Path, row: Dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(row.keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow(row)


def write_md(path: Path, row: Dict[str, str], cmd_base: str, cmd_lora: str) -> None:
    lines: List[str] = []
    lines.append("# Stage 3.1 LoRA Numeric Align")
    lines.append("")
    lines.append(f"- Generated at: `{dt.datetime.now().isoformat(timespec='seconds')}`")
    lines.append(f"- Model: `{row['model_dir']}`")
    lines.append(f"- Adapter: `{row['adapter_dir']}`")
    lines.append(f"- Prompt: `{row['prompt']}`")
    lines.append(f"- Devices: `{row['devices']}`")
    lines.append(f"- LoRA scale: `{row['lora_scale']}`")
    lines.append(f"- Adapter effective scale (alpha/r included): `{row['effective_scale']}`")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    lines.append(f"| base_max_abs_diff | {row['base_max_abs_diff']} |")
    lines.append(f"| base_mean_abs_diff | {row['base_mean_abs_diff']} |")
    lines.append(f"| lora_max_abs_diff | {row['lora_max_abs_diff']} |")
    lines.append(f"| lora_mean_abs_diff | {row['lora_mean_abs_diff']} |")
    lines.append(f"| delta_max_abs_diff | {row['delta_max_abs_diff']} |")
    lines.append(f"| delta_mean_abs_diff | {row['delta_mean_abs_diff']} |")
    lines.append(f"| delta_max_ok | {row['delta_max_ok']} |")
    lines.append("")
    lines.append("## Commands")
    lines.append(f"- base: `{cmd_base}`")
    lines.append(f"- lora: `{cmd_lora}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 3.1 LoRA numeric alignment vs HF.")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--adapter", type=str, required=True)
    ap.add_argument("--ember-bin", type=str, default="build/ember")
    ap.add_argument("--devices", type=str, default="0")
    ap.add_argument("--prompt", type=str, default="Hello, my name is")
    ap.add_argument("--lora-scale", type=float, default=1.0)
    ap.add_argument("--hf-device", type=str, default="cuda:0")
    ap.add_argument("--hf-dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--delta-max-threshold", type=float, default=1e-4)
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    repo = Path.cwd()
    model_dir = Path(args.model).expanduser().resolve()
    adapter_dir = Path(args.adapter).expanduser().resolve()
    ember_bin = Path(args.ember_bin).expanduser().resolve()
    if not ember_bin.exists():
        die(f"missing ember binary: {ember_bin}")
    if not model_dir.exists():
        die(f"model not found: {model_dir}")
    if not adapter_dir.exists():
        die(f"adapter not found: {adapter_dir}")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (repo / "reports" / f"stage31_lora_numeric_align_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    debug_base = out_dir / "debug_base"
    debug_lora = out_dir / "debug_lora"

    cmd_base = [
        str(ember_bin),
        "-m", str(model_dir),
        "--devices", args.devices,
        "--check",
        "--dump-dir", str(debug_base),
        "-p", args.prompt,
    ]
    run_cmd(cmd_base, cwd=repo, log_path=logs_dir / "ember_base.log")

    cmd_lora = [
        str(ember_bin),
        "-m", str(model_dir),
        "--devices", args.devices,
        "--adapter", str(adapter_dir),
        "--lora-scale", str(args.lora_scale),
        "--check",
        "--dump-dir", str(debug_lora),
        "-p", args.prompt,
    ]
    run_cmd(cmd_lora, cwd=repo, log_path=logs_dir / "ember_lora.log")

    meta_base = read_meta(debug_base / "meta.json")
    meta_lora = read_meta(debug_lora / "meta.json")
    vocab_size = int(meta_base["vocab_size"])
    if int(meta_lora["vocab_size"]) != vocab_size:
        die("vocab_size mismatch between base and lora debug dumps")

    tokens_base = read_tokens(debug_base / "tokens.txt")
    tokens_lora = read_tokens(debug_lora / "tokens.txt")
    if tokens_base != tokens_lora:
        die("tokenized prompt mismatch between base and lora debug dumps")
    ember_base = read_logits(debug_base / "logits.bin", vocab_size)
    ember_lora = read_logits(debug_lora / "logits.bin", vocab_size)
    ember_delta = ember_lora - ember_base

    model, hf_base = hf_last_logits(
        model_dir=model_dir,
        tokens=tokens_base,
        device=args.hf_device,
        dtype=args.hf_dtype,
    )
    updated = apply_lora_inplace_hf(model, adapter_dir=adapter_dir, lora_scale=args.lora_scale)
    with torch.no_grad():
        input_ids = torch.tensor([tokens_base], dtype=torch.long, device=args.hf_device)
        out = model(input_ids, use_cache=False)
        hf_lora = out.logits[0, -1].float().cpu()
    hf_delta = hf_lora - hf_base

    base_max, base_mean = compute_diff(hf_base, ember_base)
    lora_max, lora_mean = compute_diff(hf_lora, ember_lora)
    delta_max, delta_mean = compute_diff(hf_delta, ember_delta)

    effective_scale = args.lora_scale * read_alpha_over_r(adapter_dir)
    delta_ok = delta_max <= args.delta_max_threshold

    row = {
        "mode": "stage31_lora_numeric_align",
        "model_dir": str(model_dir),
        "adapter_dir": str(adapter_dir),
        "prompt": args.prompt,
        "devices": args.devices,
        "hf_device": args.hf_device,
        "hf_dtype": args.hf_dtype,
        "lora_scale": f"{args.lora_scale:.6f}",
        "effective_scale": f"{effective_scale:.6f}",
        "adapter_updated_matrices_hf": str(updated),
        "base_max_abs_diff": f"{base_max:.8f}",
        "base_mean_abs_diff": f"{base_mean:.8f}",
        "lora_max_abs_diff": f"{lora_max:.8f}",
        "lora_mean_abs_diff": f"{lora_mean:.8f}",
        "delta_max_abs_diff": f"{delta_max:.8f}",
        "delta_mean_abs_diff": f"{delta_mean:.8f}",
        "delta_max_threshold": f"{args.delta_max_threshold:.8f}",
        "delta_max_ok": "1" if delta_ok else "0",
    }

    write_csv(out_dir / "stage31_lora_numeric_align.csv", row)
    write_md(
        out_dir / "stage31_summary.md",
        row,
        cmd_base=" ".join(cmd_base),
        cmd_lora=" ".join(cmd_lora),
    )
    print(f"[done] out_dir={out_dir}")
    print(f"[done] csv={out_dir / 'stage31_lora_numeric_align.csv'}")
    print(f"[done] md={out_dir / 'stage31_summary.md'}")


if __name__ == "__main__":
    main()

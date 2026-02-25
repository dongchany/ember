#!/usr/bin/env python3
import argparse
import array
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM


def die(msg: str) -> None:
    raise SystemExit(f"error: {msg}")


def read_tokens(path: Path) -> List[int]:
    s = path.read_text(encoding="utf-8").strip()
    if not s:
        return []
    return [int(x) for x in s.split()]


def read_meta(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_f32(path: Path, size: int) -> torch.Tensor:
    arr = array.array("f")
    with path.open("rb") as f:
        arr.fromfile(f, size)
    if len(arr) != size:
        die(f"{path}: size mismatch got={len(arr)} expected={size}")
    return torch.tensor(arr, dtype=torch.float32)


def diff_stats(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    d = (a - b).abs()
    return float(d.max().item()), float(d.mean().item())


def load_hf_hidden(
    model_dir: Path,
    adapter_dir: Path,
    tokens: List[int],
    device: str,
    dtype: str,
) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
    if not tokens:
        die("empty tokens")
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
        out_base = model(input_ids, use_cache=False, output_hidden_states=True)
        hs_base = tuple(x[0, -1].float().cpu() for x in out_base.hidden_states)

    peft_model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    peft_model.eval()
    with torch.no_grad():
        out_lora = peft_model(input_ids, use_cache=False, output_hidden_states=True)
        hs_lora = tuple(x[0, -1].float().cpu() for x in out_lora.hidden_states)

    return hs_base, hs_lora


def write_md(path: Path, rows: List[Dict[str, str]], model_dir: Path, adapter_dir: Path, layer_worst: Dict[str, str]) -> None:
    lines: List[str] = []
    lines.append("# Stage 3.1 LoRA Delta Profile")
    lines.append("")
    lines.append(f"- Generated at: `{dt.datetime.now().isoformat(timespec='seconds')}`")
    lines.append(f"- Model: `{model_dir}`")
    lines.append(f"- Adapter: `{adapter_dir}`")
    lines.append("")
    lines.append("| name | ember_vs_hf_delta_max_abs_diff | ember_vs_hf_delta_mean_abs_diff |")
    lines.append("| --- | --- | --- |")
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['delta_max_abs_diff']} | {r['delta_mean_abs_diff']} |"
        )
    lines.append("")
    lines.append("## Key Point")
    lines.append(
        f"- Worst delta mismatch at `{layer_worst['name']}`: "
        f"max=`{layer_worst['delta_max_abs_diff']}`, mean=`{layer_worst['delta_mean_abs_diff']}`."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Profile Ember-vs-HF LoRA delta mismatch across layers.")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--adapter", type=str, required=True)
    ap.add_argument("--debug-base", type=str, required=True)
    ap.add_argument("--debug-lora", type=str, required=True)
    ap.add_argument("--hf-device", type=str, default="cuda:0")
    ap.add_argument("--hf-dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    model_dir = Path(args.model).expanduser().resolve()
    adapter_dir = Path(args.adapter).expanduser().resolve()
    debug_base = Path(args.debug_base).expanduser().resolve()
    debug_lora = Path(args.debug_lora).expanduser().resolve()
    if not model_dir.exists():
        die(f"model not found: {model_dir}")
    if not adapter_dir.exists():
        die(f"adapter not found: {adapter_dir}")
    if not debug_base.exists():
        die(f"debug_base not found: {debug_base}")
    if not debug_lora.exists():
        die(f"debug_lora not found: {debug_lora}")

    meta_b = read_meta(debug_base / "meta.json")
    meta_l = read_meta(debug_lora / "meta.json")
    hidden_size = int(meta_b["hidden_size"])
    if int(meta_l["hidden_size"]) != hidden_size:
        die("hidden_size mismatch between debug dirs")
    num_layers = int(meta_b["num_layers"])
    if int(meta_l["num_layers"]) != num_layers:
        die("num_layers mismatch between debug dirs")

    tok_b = read_tokens(debug_base / "tokens.txt")
    tok_l = read_tokens(debug_lora / "tokens.txt")
    if tok_b != tok_l:
        die("tokens mismatch between debug dirs")

    hf_base, hf_lora = load_hf_hidden(
        model_dir=model_dir,
        adapter_dir=adapter_dir,
        tokens=tok_b,
        device=args.hf_device,
        dtype=args.hf_dtype,
    )
    if len(hf_base) != len(hf_lora):
        die("HF hidden states length mismatch")
    if len(hf_base) < num_layers + 1:
        die(f"HF hidden states too short: {len(hf_base)} for num_layers={num_layers}")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (
        Path.cwd() / "reports" / f"stage31_lora_delta_profile_{ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    for i in range(num_layers):
        eb = read_f32(debug_base / f"layer_{i}_last_hidden.bin", hidden_size)
        el = read_f32(debug_lora / f"layer_{i}_last_hidden.bin", hidden_size)
        ember_delta = el - eb
        hf_delta = hf_lora[i + 1] - hf_base[i + 1]
        dmax, dmean = diff_stats(ember_delta, hf_delta)
        rows.append(
            {
                "kind": "layer",
                "index": str(i),
                "name": f"layer_{i}",
                "delta_max_abs_diff": f"{dmax:.8f}",
                "delta_mean_abs_diff": f"{dmean:.8f}",
            }
        )

    eb_final = read_f32(debug_base / "final_norm_last_hidden.bin", hidden_size)
    el_final = read_f32(debug_lora / "final_norm_last_hidden.bin", hidden_size)
    ember_final_delta = el_final - eb_final
    hf_final_delta = hf_lora[-1] - hf_base[-1]
    fdmax, fdmean = diff_stats(ember_final_delta, hf_final_delta)
    rows.append(
        {
            "kind": "final",
            "index": str(num_layers),
            "name": "final_norm",
            "delta_max_abs_diff": f"{fdmax:.8f}",
            "delta_mean_abs_diff": f"{fdmean:.8f}",
        }
    )

    rows_sorted = sorted(rows, key=lambda r: float(r["delta_max_abs_diff"]), reverse=True)
    worst = rows_sorted[0]

    csv_path = out_dir / "stage31_lora_delta_profile.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fields = ["kind", "index", "name", "delta_max_abs_diff", "delta_mean_abs_diff"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    md_path = out_dir / "stage31_summary.md"
    write_md(md_path, rows=rows, model_dir=model_dir, adapter_dir=adapter_dir, layer_worst=worst)

    print(f"[done] out_dir={out_dir}")
    print(f"[done] csv={csv_path}")
    print(f"[done] md={md_path}")
    print(f"[done] worst={worst['name']} max={worst['delta_max_abs_diff']} mean={worst['delta_mean_abs_diff']}")


if __name__ == "__main__":
    main()

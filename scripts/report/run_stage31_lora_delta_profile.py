#!/usr/bin/env python3
import argparse
import array
import csv
import datetime as dt
import json
import gc
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


def safe_ratio(num: float, den: float) -> float:
    if abs(den) < 1e-12:
        return 0.0
    return num / den


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

    del peft_model
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return hs_base, hs_lora


def parse_int_list(raw: str) -> List[int]:
    out: List[int] = []
    for x in raw.split(","):
        v = x.strip()
        if not v:
            continue
        out.append(int(v))
    return out


def parse_float_list(raw: str) -> List[float]:
    out: List[float] = []
    for x in raw.split(","):
        v = x.strip()
        if not v:
            continue
        out.append(float(v))
    return out


def write_md(
    path: Path,
    rows: List[Dict[str, str]],
    freeze_rows: List[Dict[str, str]],
    threshold_rows: List[Dict[str, str]],
    model_dir: Path,
    adapter_dir: Path,
    worst_base: Dict[str, str],
    worst_delta: Dict[str, str],
) -> None:
    lines: List[str] = []
    lines.append("# Stage 3.1 LoRA Delta Profile")
    lines.append("")
    lines.append(f"- Generated at: `{dt.datetime.now().isoformat(timespec='seconds')}`")
    lines.append(f"- Model: `{model_dir}`")
    lines.append(f"- Adapter: `{adapter_dir}`")
    lines.append("")
    lines.append("| name | base_max | lora_max | delta_max | delta/base ratio |")
    lines.append("| --- | --- | --- | --- | --- |")
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['base_max_abs_diff']} | {r['lora_max_abs_diff']} | "
            f"{r['delta_max_abs_diff']} | {r['delta_over_base_max_ratio']} |"
        )
    lines.append("")
    lines.append("## Key Point")
    lines.append(
        f"- Worst base mismatch at `{worst_base['name']}`: "
        f"max=`{worst_base['base_max_abs_diff']}`, mean=`{worst_base['base_mean_abs_diff']}`."
    )
    lines.append(
        f"- Worst LoRA-delta mismatch at `{worst_delta['name']}`: "
        f"max=`{worst_delta['delta_max_abs_diff']}`, mean=`{worst_delta['delta_mean_abs_diff']}`."
    )
    lines.append("")
    lines.append("## Threshold Crossing (delta_max_abs_diff)")
    lines.append("| threshold | first_layer_at_or_above |")
    lines.append("| --- | --- |")
    for r in threshold_rows:
        lines.append(f"| {r['threshold']} | {r['first_layer_at_or_above']} |")
    lines.append("")
    lines.append("## Freeze Prefix Risk (delta_max_abs_diff)")
    lines.append("| freeze_layers | frozen_prefix_max_delta | frozen_prefix_max_base |")
    lines.append("| --- | --- | --- |")
    for r in freeze_rows:
        lines.append(
            f"| {r['freeze_layers']} | {r['frozen_prefix_max_delta']} | {r['frozen_prefix_max_base']} |"
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
    ap.add_argument("--freeze-layers", type=str, default="8,12,18,24,30")
    ap.add_argument("--delta-thresholds", type=str, default="0.1,0.25,0.5,1.0")
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

    freeze_layers = parse_int_list(args.freeze_layers)
    delta_thresholds = parse_float_list(args.delta_thresholds)

    def _clamp_layer(v: int) -> int:
        return max(0, min(v, num_layers))

    freeze_layers = sorted(set(_clamp_layer(v) for v in freeze_layers))

    rows: List[Dict[str, str]] = []
    layer_rows: List[Dict[str, str]] = []
    for i in range(num_layers):
        eb = read_f32(debug_base / f"layer_{i}_last_hidden.bin", hidden_size)
        el = read_f32(debug_lora / f"layer_{i}_last_hidden.bin", hidden_size)
        ember_delta = el - eb
        base_max, base_mean = diff_stats(eb, hf_base[i + 1])
        lora_max, lora_mean = diff_stats(el, hf_lora[i + 1])
        hf_delta = hf_lora[i + 1] - hf_base[i + 1]
        dmax, dmean = diff_stats(ember_delta, hf_delta)
        row = {
            "kind": "layer",
            "index": str(i),
            "name": f"layer_{i}",
            "base_max_abs_diff": f"{base_max:.8f}",
            "base_mean_abs_diff": f"{base_mean:.8f}",
            "lora_max_abs_diff": f"{lora_max:.8f}",
            "lora_mean_abs_diff": f"{lora_mean:.8f}",
            "delta_max_abs_diff": f"{dmax:.8f}",
            "delta_mean_abs_diff": f"{dmean:.8f}",
            "delta_over_base_max_ratio": f"{safe_ratio(dmax, base_max):.8f}",
            "delta_over_lora_max_ratio": f"{safe_ratio(dmax, lora_max):.8f}",
        }
        rows.append(row)
        layer_rows.append(row)

    eb_final = read_f32(debug_base / "final_norm_last_hidden.bin", hidden_size)
    el_final = read_f32(debug_lora / "final_norm_last_hidden.bin", hidden_size)
    ember_final_delta = el_final - eb_final
    fbmax, fbmean = diff_stats(eb_final, hf_base[-1])
    flmax, flmean = diff_stats(el_final, hf_lora[-1])
    hf_final_delta = hf_lora[-1] - hf_base[-1]
    fdmax, fdmean = diff_stats(ember_final_delta, hf_final_delta)
    rows.append(
        {
            "kind": "final",
            "index": str(num_layers),
            "name": "final_norm",
            "base_max_abs_diff": f"{fbmax:.8f}",
            "base_mean_abs_diff": f"{fbmean:.8f}",
            "lora_max_abs_diff": f"{flmax:.8f}",
            "lora_mean_abs_diff": f"{flmean:.8f}",
            "delta_max_abs_diff": f"{fdmax:.8f}",
            "delta_mean_abs_diff": f"{fdmean:.8f}",
            "delta_over_base_max_ratio": f"{safe_ratio(fdmax, fbmax):.8f}",
            "delta_over_lora_max_ratio": f"{safe_ratio(fdmax, flmax):.8f}",
        }
    )

    rows_sorted_base = sorted(rows, key=lambda r: float(r["base_max_abs_diff"]), reverse=True)
    rows_sorted_delta = sorted(rows, key=lambda r: float(r["delta_max_abs_diff"]), reverse=True)
    worst_base = rows_sorted_base[0]
    worst_delta = rows_sorted_delta[0]

    threshold_rows: List[Dict[str, str]] = []
    for t in sorted(set(delta_thresholds)):
        first = "none"
        for r in layer_rows:
            if float(r["delta_max_abs_diff"]) >= t:
                first = r["index"]
                break
        threshold_rows.append(
            {
                "threshold": f"{t:.8f}",
                "first_layer_at_or_above": first,
            }
        )

    freeze_rows: List[Dict[str, str]] = []
    for n in freeze_layers:
        if n <= 0:
            prefix = []
        else:
            prefix = layer_rows[:n]
        if prefix:
            max_delta = max(float(r["delta_max_abs_diff"]) for r in prefix)
            max_base = max(float(r["base_max_abs_diff"]) for r in prefix)
        else:
            max_delta = 0.0
            max_base = 0.0
        freeze_rows.append(
            {
                "freeze_layers": str(n),
                "frozen_prefix_max_delta": f"{max_delta:.8f}",
                "frozen_prefix_max_base": f"{max_base:.8f}",
            }
        )

    csv_path = out_dir / "stage31_lora_delta_profile.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "kind",
            "index",
            "name",
            "base_max_abs_diff",
            "base_mean_abs_diff",
            "lora_max_abs_diff",
            "lora_mean_abs_diff",
            "delta_max_abs_diff",
            "delta_mean_abs_diff",
            "delta_over_base_max_ratio",
            "delta_over_lora_max_ratio",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    freeze_csv = out_dir / "stage31_lora_delta_freeze_summary.csv"
    with freeze_csv.open("w", encoding="utf-8", newline="") as f:
        fields = ["freeze_layers", "frozen_prefix_max_delta", "frozen_prefix_max_base"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in freeze_rows:
            w.writerow(r)

    threshold_csv = out_dir / "stage31_lora_delta_thresholds.csv"
    with threshold_csv.open("w", encoding="utf-8", newline="") as f:
        fields = ["threshold", "first_layer_at_or_above"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in threshold_rows:
            w.writerow(r)

    md_path = out_dir / "stage31_summary.md"
    write_md(
        md_path,
        rows=rows,
        freeze_rows=freeze_rows,
        threshold_rows=threshold_rows,
        model_dir=model_dir,
        adapter_dir=adapter_dir,
        worst_base=worst_base,
        worst_delta=worst_delta,
    )

    print(f"[done] out_dir={out_dir}")
    print(f"[done] csv={csv_path}")
    print(f"[done] freeze_csv={freeze_csv}")
    print(f"[done] threshold_csv={threshold_csv}")
    print(f"[done] md={md_path}")
    print(
        f"[done] worst_base={worst_base['name']} "
        f"max={worst_base['base_max_abs_diff']} mean={worst_base['base_mean_abs_diff']}"
    )
    print(
        f"[done] worst_delta={worst_delta['name']} "
        f"max={worst_delta['delta_max_abs_diff']} mean={worst_delta['delta_mean_abs_diff']}"
    )


if __name__ == "__main__":
    main()

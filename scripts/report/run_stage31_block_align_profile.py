#!/usr/bin/env python3
import argparse
import array
import csv
import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM


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


def read_intermediate_size(model_dir: Path, hidden_size: int) -> int:
    cfg = model_dir / "config.json"
    if cfg.exists():
        try:
            obj = json.loads(cfg.read_text(encoding="utf-8"))
            v = int(obj.get("intermediate_size", 0) or 0)
            if v > 0:
                return v
        except Exception:
            pass
    return hidden_size * 4


def read_f32(path: Path, size: int) -> torch.Tensor:
    arr = array.array("f")
    with path.open("rb") as f:
        arr.fromfile(f, size)
    if len(arr) != size:
        die(f"{path}: size mismatch got={len(arr)} expected={size}")
    return torch.tensor(arr, dtype=torch.float32)


def compute_diff(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float]:
    d = (a - b).abs()
    return float(d.max().item()), float(d.mean().item())


def safe_ratio(num: float, den: float) -> float:
    if abs(den) < 1e-12:
        return 0.0
    return num / den


def parse_int_list(raw: str) -> List[int]:
    out: List[int] = []
    for x in raw.split(","):
        v = x.strip()
        if not v:
            continue
        out.append(int(v))
    return out


def load_hf_model(model_dir: Path, device: str, dtype: str) -> AutoModelForCausalLM:
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
    return model


def _last_token_cpu(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        return x[0, -1].float().detach().cpu()
    if x.dim() == 2:
        return x[0].float().detach().cpu()
    return x.float().detach().cpu()


def collect_hf_blocks(
    model: AutoModelForCausalLM,
    tokens: List[int],
    layers: List[int],
    device: str,
) -> Dict[int, Dict[str, torch.Tensor]]:
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        die("unexpected HF model structure: missing model.layers")
    model_layers = model.model.layers

    captures: Dict[int, Dict[str, torch.Tensor]] = {i: {} for i in layers}
    handles = []

    def on_layer_input(layer_idx: int) -> Callable:
        def _hook(_module, inp):
            if not inp:
                return
            captures[layer_idx]["layer_input"] = _last_token_cpu(inp[0])

        return _hook

    def on_self_attn(layer_idx: int) -> Callable:
        def _hook(_module, _inp, out):
            y = out[0] if isinstance(out, (tuple, list)) else out
            captures[layer_idx]["attn_out"] = _last_token_cpu(y)

        return _hook

    def on_post_norm_pre(layer_idx: int) -> Callable:
        def _hook(_module, inp):
            if not inp:
                return
            captures[layer_idx]["attn_residual"] = _last_token_cpu(inp[0])

        return _hook

    def on_post_norm(layer_idx: int) -> Callable:
        def _hook(_module, _inp, out):
            captures[layer_idx]["post_attn_norm"] = _last_token_cpu(out)

        return _hook

    def on_gate_pre(layer_idx: int) -> Callable:
        def _hook(_module, _inp, out):
            captures[layer_idx]["mlp_gate_pre"] = _last_token_cpu(out)

        return _hook

    def on_up(layer_idx: int) -> Callable:
        def _hook(_module, _inp, out):
            captures[layer_idx]["mlp_up"] = _last_token_cpu(out)

        return _hook

    def on_mlp(layer_idx: int) -> Callable:
        def _hook(_module, _inp, out):
            captures[layer_idx]["mlp_out"] = _last_token_cpu(out)

        return _hook

    def on_layer(layer_idx: int) -> Callable:
        def _hook(_module, _inp, out):
            y = out[0] if isinstance(out, (tuple, list)) else out
            captures[layer_idx]["last_hidden"] = _last_token_cpu(y)

        return _hook

    for layer_idx in layers:
        if layer_idx < 0 or layer_idx >= len(model_layers):
            die(f"layer out of range: {layer_idx} (num_layers={len(model_layers)})")
        lyr = model_layers[layer_idx]
        handles.append(lyr.register_forward_pre_hook(on_layer_input(layer_idx)))
        handles.append(lyr.self_attn.register_forward_hook(on_self_attn(layer_idx)))
        handles.append(lyr.post_attention_layernorm.register_forward_pre_hook(on_post_norm_pre(layer_idx)))
        handles.append(lyr.post_attention_layernorm.register_forward_hook(on_post_norm(layer_idx)))
        handles.append(lyr.mlp.gate_proj.register_forward_hook(on_gate_pre(layer_idx)))
        handles.append(lyr.mlp.up_proj.register_forward_hook(on_up(layer_idx)))
        handles.append(lyr.mlp.register_forward_hook(on_mlp(layer_idx)))
        handles.append(lyr.register_forward_hook(on_layer(layer_idx)))

    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    with torch.no_grad():
        _ = model(input_ids, use_cache=False)

    for layer_idx in layers:
        row = captures[layer_idx]
        if "mlp_gate_pre" in row and "mlp_up" in row:
            gate = row["mlp_gate_pre"]
            up = row["mlp_up"]
            gate_act = torch.nn.functional.silu(gate)
            row["mlp_gate_act"] = gate_act
            row["mlp_mul"] = gate_act * up

    for h in handles:
        h.remove()

    return captures


def resolve_core_model(m) -> AutoModelForCausalLM:
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m
    if hasattr(m, "base_model") and hasattr(m.base_model, "model") and hasattr(m.base_model.model, "layers"):
        return m.base_model
    if hasattr(m, "get_base_model"):
        base = m.get_base_model()
        if hasattr(base, "model") and hasattr(base.model, "layers"):
            return base
    die("failed to resolve core model with model.layers")


def collect_hf_blocks_base_and_lora(
    model_dir: Path,
    adapter_dir: Path,
    tokens: List[int],
    device: str,
    dtype: str,
    layers: List[int],
) -> Tuple[Dict[int, Dict[str, torch.Tensor]], Dict[int, Dict[str, torch.Tensor]]]:
    base_model = load_hf_model(model_dir=model_dir, device=device, dtype=dtype)
    base_core = resolve_core_model(base_model)
    base_cap = collect_hf_blocks(model=base_core, tokens=tokens, layers=layers, device=device)

    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=False)
    peft_model.eval()
    lora_core = resolve_core_model(peft_model)
    lora_cap = collect_hf_blocks(model=lora_core, tokens=tokens, layers=layers, device=device)

    del peft_model
    del base_model
    torch.cuda.empty_cache()
    return base_cap, lora_cap


def run_ember_check(
    repo: Path,
    ember_bin: Path,
    model_dir: Path,
    prompt: str,
    devices: str,
    dump_layer: int,
    dump_dir: Path,
    log_path: Path,
    adapter_dir: Optional[Path] = None,
    lora_scale: float = 1.0,
) -> None:
    cmd = [
        str(ember_bin),
        "-m", str(model_dir),
        "--devices", devices,
        "--check",
        "--dump-layer", str(dump_layer),
        "--dump-dir", str(dump_dir),
        "-p", prompt,
    ]
    if adapter_dir is not None:
        cmd.extend(["--adapter", str(adapter_dir), "--lora-scale", str(lora_scale)])
    run_cmd(cmd, cwd=repo, log_path=log_path)


def load_ember_block(
    debug_dir: Path,
    layer: int,
    block: str,
    hidden_size: int,
    intermediate_size: int,
) -> Optional[torch.Tensor]:
    if block == "final_norm":
        p = debug_dir / "final_norm_last_hidden.bin"
        if not p.exists():
            return None
        return read_f32(p, hidden_size)
    if block in {"mlp_gate_pre", "mlp_up", "mlp_gate_act", "mlp_mul"}:
        size = intermediate_size
    else:
        size = hidden_size
    p = debug_dir / f"layer_{layer}_{block}.bin"
    if not p.exists():
        return None
    return read_f32(p, size)


def load_ember_layer_input(
    debug_all_dir: Path,
    layer: int,
    hidden_size: int,
) -> Optional[torch.Tensor]:
    if layer <= 0:
        return None
    p = debug_all_dir / f"layer_{layer - 1}_last_hidden.bin"
    if not p.exists():
        return None
    return read_f32(p, hidden_size)


def attn_residual_decomp(
    e_in: torch.Tensor,
    e_attn: torch.Tensor,
    e_res: torch.Tensor,
    h_in: torch.Tensor,
    h_attn: torch.Tensor,
    h_res: torch.Tensor,
) -> Dict[str, float]:
    in_diff = e_in - h_in
    attn_diff = e_attn - h_attn
    res_diff = e_res - h_res
    sum_diff = in_diff + attn_diff
    gap = res_diff - sum_diff
    return {
        "input_max": float(in_diff.abs().max().item()),
        "input_mean": float(in_diff.abs().mean().item()),
        "attn_max": float(attn_diff.abs().max().item()),
        "attn_mean": float(attn_diff.abs().mean().item()),
        "residual_max": float(res_diff.abs().max().item()),
        "residual_mean": float(res_diff.abs().mean().item()),
        "sum_max": float(sum_diff.abs().max().item()),
        "sum_mean": float(sum_diff.abs().mean().item()),
        "gap_max": float(gap.abs().max().item()),
        "gap_mean": float(gap.abs().mean().item()),
    }


def write_md(
    path: Path,
    rows: List[Dict[str, str]],
    model_dir: Path,
    adapter_dir: Path,
    layers: List[int],
    worst: Dict[str, str],
) -> None:
    lines: List[str] = []
    lines.append("# Stage 3.1 Block Align Profile")
    lines.append("")
    lines.append(f"- Generated at: `{dt.datetime.now().isoformat(timespec='seconds')}`")
    lines.append(f"- Model: `{model_dir}`")
    lines.append(f"- Adapter: `{adapter_dir}`")
    lines.append(f"- Layers: `{','.join(str(x) for x in layers)}`")
    lines.append("")
    lines.append("| layer | block | base_max | lora_max | delta_max | delta/base |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in rows:
        lines.append(
            f"| {r['layer']} | {r['block']} | {r['base_max_abs_diff']} | "
            f"{r['lora_max_abs_diff']} | {r['delta_max_abs_diff']} | {r['delta_over_base_max_ratio']} |"
        )
    lines.append("")
    lines.append("## Key Point")
    lines.append(
        f"- Worst delta mismatch: layer `{worst['layer']}`, block `{worst['block']}`, "
        f"delta_max=`{worst['delta_max_abs_diff']}`, base_max=`{worst['base_max_abs_diff']}`."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage31 block-level base/lora/delta align profile.")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--adapter", type=str, required=True)
    ap.add_argument("--ember-bin", type=str, default="build/ember")
    ap.add_argument("--devices", type=str, default="0")
    ap.add_argument("--prompt", type=str, default="Extract fields from JSON:")
    ap.add_argument("--lora-scale", type=float, default=1.0)
    ap.add_argument("--layers", type=str, default="31,32,33,34,35")
    ap.add_argument("--hf-device", type=str, default="cuda:0")
    ap.add_argument("--hf-dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    repo = Path.cwd()
    model_dir = Path(args.model).expanduser().resolve()
    adapter_dir = Path(args.adapter).expanduser().resolve()
    ember_bin = Path(args.ember_bin).expanduser().resolve()
    if not model_dir.exists():
        die(f"model not found: {model_dir}")
    if not adapter_dir.exists():
        die(f"adapter not found: {adapter_dir}")
    if not ember_bin.exists():
        die(f"ember binary not found: {ember_bin}")

    layers = sorted(set(parse_int_list(args.layers)))
    if not layers:
        die("empty layers list")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (
        repo / "reports" / f"stage31_block_align_profile_{ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    debug_base_root = out_dir / "debug_base"
    debug_lora_root = out_dir / "debug_lora"
    debug_base_all = out_dir / "debug_base_all"
    debug_lora_all = out_dir / "debug_lora_all"
    debug_base_root.mkdir(parents=True, exist_ok=True)
    debug_lora_root.mkdir(parents=True, exist_ok=True)

    run_ember_check(
        repo=repo,
        ember_bin=ember_bin,
        model_dir=model_dir,
        prompt=args.prompt,
        devices=args.devices,
        dump_layer=-1,
        dump_dir=debug_base_all,
        log_path=logs_dir / "ember_base_all.log",
        adapter_dir=None,
    )
    run_ember_check(
        repo=repo,
        ember_bin=ember_bin,
        model_dir=model_dir,
        prompt=args.prompt,
        devices=args.devices,
        dump_layer=-1,
        dump_dir=debug_lora_all,
        log_path=logs_dir / "ember_lora_all.log",
        adapter_dir=adapter_dir,
        lora_scale=args.lora_scale,
    )

    for layer in layers:
        run_ember_check(
            repo=repo,
            ember_bin=ember_bin,
            model_dir=model_dir,
            prompt=args.prompt,
            devices=args.devices,
            dump_layer=layer,
            dump_dir=debug_base_root / f"layer_{layer}",
            log_path=logs_dir / f"ember_base_layer_{layer}.log",
            adapter_dir=None,
        )
        run_ember_check(
            repo=repo,
            ember_bin=ember_bin,
            model_dir=model_dir,
            prompt=args.prompt,
            devices=args.devices,
            dump_layer=layer,
            dump_dir=debug_lora_root / f"layer_{layer}",
            log_path=logs_dir / f"ember_lora_layer_{layer}.log",
            adapter_dir=adapter_dir,
            lora_scale=args.lora_scale,
        )

    meta = read_meta(debug_base_all / "meta.json")
    hidden_size = int(meta["hidden_size"])
    intermediate_size = (
        int(meta["intermediate_size"])
        if "intermediate_size" in meta
        else read_intermediate_size(model_dir=model_dir, hidden_size=hidden_size)
    )
    tokens = read_tokens(debug_base_all / "tokens.txt")
    if not tokens:
        die("empty tokenized prompt")

    hf_base, hf_lora = collect_hf_blocks_base_and_lora(
        model_dir=model_dir,
        adapter_dir=adapter_dir,
        tokens=tokens,
        device=args.hf_device,
        dtype=args.hf_dtype,
        layers=layers,
    )

    blocks = [
        "layer_input",
        "attn_out",
        "attn_residual",
        "post_attn_norm",
        "mlp_gate_pre",
        "mlp_up",
        "mlp_gate_act",
        "mlp_mul",
        "mlp_out",
        "last_hidden",
    ]

    rows: List[Dict[str, str]] = []
    decomp_rows: List[Dict[str, str]] = []
    for layer in layers:
        dbg_b = debug_base_root / f"layer_{layer}"
        dbg_l = debug_lora_root / f"layer_{layer}"
        hf_b = hf_base.get(layer, {})
        hf_l = hf_lora.get(layer, {})
        for block in blocks:
            if block == "layer_input":
                eb = load_ember_layer_input(debug_base_all, layer, hidden_size)
                el = load_ember_layer_input(debug_lora_all, layer, hidden_size)
            else:
                eb = load_ember_block(dbg_b, layer, block, hidden_size, intermediate_size)
                el = load_ember_block(dbg_l, layer, block, hidden_size, intermediate_size)
            hb = hf_b.get(block)
            hl = hf_l.get(block)
            if eb is None or el is None or hb is None or hl is None:
                continue
            if eb.numel() != hb.numel() or el.numel() != hl.numel():
                continue
            bmax, bmean = compute_diff(eb, hb)
            lmax, lmean = compute_diff(el, hl)
            dmax, dmean = compute_diff(el - eb, hl - hb)
            rows.append(
                {
                    "layer": str(layer),
                    "block": block,
                    "base_max_abs_diff": f"{bmax:.8f}",
                    "base_mean_abs_diff": f"{bmean:.8f}",
                    "lora_max_abs_diff": f"{lmax:.8f}",
                    "lora_mean_abs_diff": f"{lmean:.8f}",
                    "delta_max_abs_diff": f"{dmax:.8f}",
                    "delta_mean_abs_diff": f"{dmean:.8f}",
                    "delta_over_base_max_ratio": f"{safe_ratio(dmax, bmax):.8f}",
                }
            )

        # Decompose residual mismatch into input + attn terms.
        eb_in = load_ember_layer_input(debug_base_all, layer, hidden_size)
        el_in = load_ember_layer_input(debug_lora_all, layer, hidden_size)
        eb_attn = load_ember_block(dbg_b, layer, "attn_out", hidden_size, intermediate_size)
        el_attn = load_ember_block(dbg_l, layer, "attn_out", hidden_size, intermediate_size)
        eb_res = load_ember_block(dbg_b, layer, "attn_residual", hidden_size, intermediate_size)
        el_res = load_ember_block(dbg_l, layer, "attn_residual", hidden_size, intermediate_size)
        hb_in = hf_b.get("layer_input")
        hl_in = hf_l.get("layer_input")
        hb_attn = hf_b.get("attn_out")
        hl_attn = hf_l.get("attn_out")
        hb_res = hf_b.get("attn_residual")
        hl_res = hf_l.get("attn_residual")
        if all(x is not None for x in [eb_in, el_in, eb_attn, el_attn, eb_res, el_res, hb_in, hl_in, hb_attn, hl_attn, hb_res, hl_res]):
            base_m = attn_residual_decomp(eb_in, eb_attn, eb_res, hb_in, hb_attn, hb_res)
            lora_m = attn_residual_decomp(el_in, el_attn, el_res, hl_in, hl_attn, hl_res)
            delta_m = attn_residual_decomp(
                el_in - eb_in,
                el_attn - eb_attn,
                el_res - eb_res,
                hl_in - hb_in,
                hl_attn - hb_attn,
                hl_res - hb_res,
            )
            decomp_rows.append(
                {
                    "layer": str(layer),
                    "base_input_max": f"{base_m['input_max']:.8f}",
                    "base_attn_max": f"{base_m['attn_max']:.8f}",
                    "base_residual_max": f"{base_m['residual_max']:.8f}",
                    "base_sum_max": f"{base_m['sum_max']:.8f}",
                    "base_gap_max": f"{base_m['gap_max']:.8f}",
                    "lora_input_max": f"{lora_m['input_max']:.8f}",
                    "lora_attn_max": f"{lora_m['attn_max']:.8f}",
                    "lora_residual_max": f"{lora_m['residual_max']:.8f}",
                    "lora_sum_max": f"{lora_m['sum_max']:.8f}",
                    "lora_gap_max": f"{lora_m['gap_max']:.8f}",
                    "delta_input_max": f"{delta_m['input_max']:.8f}",
                    "delta_attn_max": f"{delta_m['attn_max']:.8f}",
                    "delta_residual_max": f"{delta_m['residual_max']:.8f}",
                    "delta_sum_max": f"{delta_m['sum_max']:.8f}",
                    "delta_gap_max": f"{delta_m['gap_max']:.8f}",
                }
            )

    if not rows:
        die("no comparable block rows found")

    rows_sorted = sorted(rows, key=lambda r: float(r["delta_max_abs_diff"]), reverse=True)
    worst = rows_sorted[0]

    csv_path = out_dir / "stage31_block_align_profile.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "layer",
            "block",
            "base_max_abs_diff",
            "base_mean_abs_diff",
            "lora_max_abs_diff",
            "lora_mean_abs_diff",
            "delta_max_abs_diff",
            "delta_mean_abs_diff",
            "delta_over_base_max_ratio",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    md_path = out_dir / "stage31_summary.md"
    write_md(
        path=md_path,
        rows=rows,
        model_dir=model_dir,
        adapter_dir=adapter_dir,
        layers=layers,
        worst=worst,
    )

    decomp_csv = out_dir / "stage31_attn_residual_decomp.csv"
    if decomp_rows:
        with decomp_csv.open("w", encoding="utf-8", newline="") as f:
            fields = [
                "layer",
                "base_input_max",
                "base_attn_max",
                "base_residual_max",
                "base_sum_max",
                "base_gap_max",
                "lora_input_max",
                "lora_attn_max",
                "lora_residual_max",
                "lora_sum_max",
                "lora_gap_max",
                "delta_input_max",
                "delta_attn_max",
                "delta_residual_max",
                "delta_sum_max",
                "delta_gap_max",
            ]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in decomp_rows:
                w.writerow(r)

    print(f"[done] out_dir={out_dir}")
    print(f"[done] csv={csv_path}")
    if decomp_rows:
        print(f"[done] decomp_csv={decomp_csv}")
    print(f"[done] md={md_path}")
    print(
        f"[done] worst layer={worst['layer']} block={worst['block']} "
        f"delta_max={worst['delta_max_abs_diff']} base_max={worst['base_max_abs_diff']}"
    )


if __name__ == "__main__":
    main()

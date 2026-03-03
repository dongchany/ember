#!/usr/bin/env python3
import argparse
import array
import glob
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_f32(path, size):
    arr = array.array("f")
    with open(path, "rb") as f:
        arr.fromfile(f, size)
    if len(arr) != size:
        raise ValueError(f"{path}: size mismatch {len(arr)} vs {size}")
    return torch.tensor(arr, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description="Compare Ember hidden states with HF reference.")
    parser.add_argument("--model", required=True, help="Model directory (local HF cache).")
    parser.add_argument("--debug-dir", required=True, help="Ember debug dir.")
    parser.add_argument("--layer", type=int, default=-1, help="Layer index for attn/mlp compare.")
    parser.add_argument("--max-abs-threshold", type=float, default=None,
                        help="Fail if any layer/final max_abs_diff exceeds this value.")
    parser.add_argument("--mean-abs-threshold", type=float, default=None,
                        help="Fail if any layer/final mean_abs_diff exceeds this value.")
    args = parser.parse_args()

    meta_path = os.path.join(args.debug_dir, "meta.json")
    tokens_path = os.path.join(args.debug_dir, "tokens.txt")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    hidden_size = int(meta["hidden_size"])

    with open(tokens_path, "r", encoding="utf-8") as f:
        tokens = [int(x) for x in f.read().strip().split()]
    if not tokens:
        print("No tokens found.", file=sys.stderr)
        return 1

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()

    input_ids = torch.tensor([tokens], dtype=torch.long)
    attn_out = None
    mlp_out = None
    layer_mod = None
    if args.layer >= 0:
        layer_mod = model.model.layers[args.layer]

        def _attn_hook(_module, _inp, out):
            nonlocal attn_out
            if isinstance(out, (tuple, list)):
                attn_out = out[0]
            else:
                attn_out = out

        def _mlp_hook(_module, _inp, out):
            nonlocal mlp_out
            mlp_out = out

        layer_mod.self_attn.register_forward_hook(_attn_hook)
        layer_mod.mlp.register_forward_hook(_mlp_hook)

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states

    layer_files = glob.glob(os.path.join(args.debug_dir, "layer_*_last_hidden.bin"))
    layers = []
    for path in layer_files:
        base = os.path.basename(path)
        parts = base.split("_")
        if len(parts) < 3:
            continue
        try:
            layer_idx = int(parts[1])
        except ValueError:
            continue
        layers.append((layer_idx, path))
    layers.sort(key=lambda x: x[0])
    if not layers:
        print("No layer dumps found.", file=sys.stderr)
        return 1

    worst_max = 0.0
    worst_mean = 0.0
    worst_max_name = None
    worst_mean_name = None

    print("Layer\tmax_abs_diff\tmean_abs_diff")
    for layer_idx, path in layers:
        ember_vec = read_f32(path, hidden_size)
        # hidden_states[0] is embedding, so layer_idx -> hidden_states[layer_idx + 1]
        if layer_idx + 1 >= len(hidden_states):
            print(f"Layer {layer_idx}: missing HF hidden state")
            continue
        hf_vec = hidden_states[layer_idx + 1][0, -1].float().cpu()

        diff = (hf_vec - ember_vec).abs()
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())
        print(f"{layer_idx}\t{max_diff:.6f}\t{mean_diff:.6f}")
        if max_diff > worst_max:
            worst_max = max_diff
            worst_max_name = f"layer_{layer_idx}"
        if mean_diff > worst_mean:
            worst_mean = mean_diff
            worst_mean_name = f"layer_{layer_idx}"

    final_path = os.path.join(args.debug_dir, "final_norm_last_hidden.bin")
    if os.path.exists(final_path):
        ember_final = read_f32(final_path, hidden_size)
        hf_final = hidden_states[-1][0, -1].float().cpu()
        diff = (hf_final - ember_final).abs()
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())
        print("\nFinal norm vs HF last hidden")
        print(f"max_abs_diff: {max_diff:.6f}")
        print(f"mean_abs_diff: {mean_diff:.6f}")
        if max_diff > worst_max:
            worst_max = max_diff
            worst_max_name = "final_norm"
        if mean_diff > worst_mean:
            worst_mean = mean_diff
            worst_mean_name = "final_norm"

    # Show top token for sanity
    if tokenizer is not None:
        top_id = int(torch.argmax(outputs.logits[0, -1]).item())
        tok = tokenizer.convert_ids_to_tokens(top_id)
        print(f"\nHF top1 token: {top_id} {tok}")

    if args.layer >= 0:
        prefix = f"layer_{args.layer}_"
        attn_path = os.path.join(args.debug_dir, prefix + "attn_out.bin")
        attn_res_path = os.path.join(args.debug_dir, prefix + "attn_residual.bin")
        norm_path = os.path.join(args.debug_dir, prefix + "post_attn_norm.bin")
        mlp_path = os.path.join(args.debug_dir, prefix + "mlp_out.bin")
        gate_pre_path = os.path.join(args.debug_dir, prefix + "mlp_gate_pre.bin")
        up_path = os.path.join(args.debug_dir, prefix + "mlp_up.bin")
        gate_act_path = os.path.join(args.debug_dir, prefix + "mlp_gate_act.bin")
        mul_path = os.path.join(args.debug_dir, prefix + "mlp_mul.bin")

        hf_attn_res = None
        hf_norm = None
        if attn_out is not None and os.path.exists(attn_path):
            ember_attn = read_f32(attn_path, hidden_size)
            hf_attn = attn_out[0, -1].float().cpu()
            diff = (hf_attn - ember_attn).abs()
            print("\nAttn out vs HF self_attn output")
            print(f"max_abs_diff: {float(diff.max()):.6f}")
            print(f"mean_abs_diff: {float(diff.mean()):.6f}")

        if attn_out is not None and os.path.exists(attn_res_path):
            ember_attn_res = read_f32(attn_res_path, hidden_size)
            residual = hidden_states[args.layer][0, -1].float().cpu()
            hf_attn_res = residual + attn_out[0, -1].float().cpu()
            diff = (hf_attn_res - ember_attn_res).abs()
            print("\nAttn residual vs HF (residual + attn_out)")
            print(f"max_abs_diff: {float(diff.max()):.6f}")
            print(f"mean_abs_diff: {float(diff.mean()):.6f}")

        if os.path.exists(norm_path):
            ember_norm = read_f32(norm_path, hidden_size)
            if layer_mod is not None and hasattr(layer_mod, "post_attention_layernorm"):
                if hf_attn_res is None:
                    hf_attn_res = hidden_states[args.layer][0, -1].float().cpu()
                hf_norm = layer_mod.post_attention_layernorm(hf_attn_res.unsqueeze(0)).squeeze(0).detach().cpu()
                diff = (hf_norm - ember_norm).abs()
                print("\nPost-attn norm vs HF layernorm")
                print(f"max_abs_diff: {float(diff.max()):.6f}")
                print(f"mean_abs_diff: {float(diff.mean()):.6f}")

        if mlp_out is not None and os.path.exists(mlp_path):
            ember_mlp = read_f32(mlp_path, hidden_size)
            hf_mlp = mlp_out[0, -1].float().detach().cpu()
            diff = (hf_mlp - ember_mlp).abs()
            print("\nMLP out vs HF mlp output")
            print(f"max_abs_diff: {float(diff.max()):.6f}")
            print(f"mean_abs_diff: {float(diff.mean()):.6f}")

        if hf_norm is not None:
            gate = layer_mod.mlp.gate_proj(hf_norm).detach().cpu()
            up = layer_mod.mlp.up_proj(hf_norm).detach().cpu()
            gate_act = torch.nn.functional.silu(gate).detach().cpu()
            mul = gate_act * up

            if os.path.exists(gate_pre_path):
                ember_gate = read_f32(gate_pre_path, gate.numel())
                diff = (gate.squeeze(0) - ember_gate).abs()
                print("\nGate proj vs HF gate_proj")
                print(f"max_abs_diff: {float(diff.max()):.6f}")
                print(f"mean_abs_diff: {float(diff.mean()):.6f}")

            if os.path.exists(up_path):
                ember_up = read_f32(up_path, up.numel())
                diff = (up.squeeze(0) - ember_up).abs()
                print("\nUp proj vs HF up_proj")
                print(f"max_abs_diff: {float(diff.max()):.6f}")
                print(f"mean_abs_diff: {float(diff.mean()):.6f}")

            if os.path.exists(gate_act_path):
                ember_gate_act = read_f32(gate_act_path, gate_act.numel())
                diff = (gate_act.squeeze(0) - ember_gate_act).abs()
                print("\nGate activation vs HF silu(gate)")
                print(f"max_abs_diff: {float(diff.max()):.6f}")
                print(f"mean_abs_diff: {float(diff.mean()):.6f}")

            if os.path.exists(mul_path):
                ember_mul = read_f32(mul_path, mul.numel())
                diff = (mul.squeeze(0) - ember_mul).abs()
                print("\nGate*Up vs HF")
                print(f"max_abs_diff: {float(diff.max()):.6f}")
                print(f"mean_abs_diff: {float(diff.mean()):.6f}")

            # Compare using Ember norm as input to isolate GEMM/layout issues
            try:
                ember_norm_in = torch.tensor(ember_norm, dtype=torch.float32).unsqueeze(0)
                gate_e = layer_mod.mlp.gate_proj(ember_norm_in).detach().cpu()
                up_e = layer_mod.mlp.up_proj(ember_norm_in).detach().cpu()
                gate_act_e = torch.nn.functional.silu(gate_e).detach().cpu()
                mul_e = gate_act_e * up_e

                if os.path.exists(gate_pre_path):
                    ember_gate = read_f32(gate_pre_path, gate_e.numel())
                    diff = (gate_e.squeeze(0) - ember_gate).abs()
                    print("\nGate proj vs HF gate_proj (Ember norm input)")
                    print(f"max_abs_diff: {float(diff.max()):.6f}")
                    print(f"mean_abs_diff: {float(diff.mean()):.6f}")

                if os.path.exists(up_path):
                    ember_up = read_f32(up_path, up_e.numel())
                    diff = (up_e.squeeze(0) - ember_up).abs()
                    print("\nUp proj vs HF up_proj (Ember norm input)")
                    print(f"max_abs_diff: {float(diff.max()):.6f}")
                    print(f"mean_abs_diff: {float(diff.mean()):.6f}")

                if os.path.exists(gate_act_path):
                    ember_gate_act = read_f32(gate_act_path, gate_act_e.numel())
                    diff = (gate_act_e.squeeze(0) - ember_gate_act).abs()
                    print("\nGate activation vs HF silu(gate) (Ember norm input)")
                    print(f"max_abs_diff: {float(diff.max()):.6f}")
                    print(f"mean_abs_diff: {float(diff.mean()):.6f}")

                if os.path.exists(mul_path):
                    ember_mul = read_f32(mul_path, mul_e.numel())
                    diff = (mul_e.squeeze(0) - ember_mul).abs()
                    print("\nGate*Up vs HF (Ember norm input)")
                    print(f"max_abs_diff: {float(diff.max()):.6f}")
                    print(f"mean_abs_diff: {float(diff.mean()):.6f}")
            except Exception as exc:
                print(f"\n[Warning] Ember-norm isolation failed: {exc}")

    if args.max_abs_threshold is not None and worst_max > args.max_abs_threshold:
        name = worst_max_name or "unknown"
        print(f"[Fail] max_abs_diff {worst_max:.6f} > {args.max_abs_threshold:.6f} at {name}",
              file=sys.stderr)
        return 2
    if args.mean_abs_threshold is not None and worst_mean > args.mean_abs_threshold:
        name = worst_mean_name or "unknown"
        print(f"[Fail] mean_abs_diff {worst_mean:.6f} > {args.mean_abs_threshold:.6f} at {name}",
              file=sys.stderr)
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

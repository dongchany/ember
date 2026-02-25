#!/usr/bin/env python3
import argparse
import datetime as dt
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common_report import die, split_ints, write_csv


def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0.0:
        return min(values)
    if q >= 1.0:
        return max(values)
    xs = sorted(values)
    idx = int(round((len(xs) - 1) * q))
    return xs[idx]


def resolve_layers(model: torch.nn.Module) -> Sequence[torch.nn.Module]:
    candidates = [
        getattr(model, "model", None),
        getattr(getattr(model, "model", None), "model", None),
        getattr(model, "base_model", None),
        getattr(getattr(model, "base_model", None), "model", None),
        getattr(getattr(getattr(model, "base_model", None), "model", None), "model", None),
    ]
    for c in candidates:
        if c is None or not hasattr(c, "layers"):
            continue
        layers = getattr(c, "layers")
        if len(layers) > 0 and hasattr(layers[0], "self_attn") and hasattr(layers[0].self_attn, "k_proj"):
            return layers
    for _, mod in model.named_modules():
        if not hasattr(mod, "layers"):
            continue
        layers = getattr(mod, "layers")
        try:
            if len(layers) > 0 and hasattr(layers[0], "self_attn") and hasattr(layers[0].self_attn, "k_proj"):
                return layers
        except Exception:
            continue
    die("failed to resolve transformer layers with self_attn.k_proj")
    raise AssertionError("unreachable")


def capture_k_proj_outputs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    layer_step: int,
) -> Dict[int, torch.Tensor]:
    layers = resolve_layers(model)
    captures: Dict[int, torch.Tensor] = {}
    handles: List[torch.utils.hooks.RemovableHandle] = []

    for i, layer in enumerate(layers):
        if layer_step > 1 and (i % layer_step != 0):
            continue
        if not hasattr(layer, "self_attn") or not hasattr(layer.self_attn, "k_proj"):
            continue
        k_proj = layer.self_attn.k_proj

        def _hook(_mod: torch.nn.Module, _inp: Tuple[torch.Tensor, ...], out: torch.Tensor, idx: int = i) -> None:
            # out shape: [batch, seq_len, hidden]
            captures[idx] = out.detach().float().cpu()[0]

        handles.append(k_proj.register_forward_hook(_hook))

    with torch.no_grad():
        _ = model(input_ids=input_ids, use_cache=False)

    for h in handles:
        h.remove()
    if not captures:
        die("no k_proj outputs were captured")
    return captures


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    die(f"unsupported dtype: {name}")
    raise AssertionError("unreachable")


def build_input_ids(tokenizer: AutoTokenizer, text: str, seq_len: int, device: str) -> torch.Tensor:
    seed_ids = tokenizer.encode(text, add_special_tokens=False)
    if not seed_ids:
        seed_ids = [tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1]
    ids: List[int] = []
    while len(ids) < seq_len:
        ids.extend(seed_ids)
    ids = ids[:seq_len]
    return torch.tensor([ids], dtype=torch.long, device=device)


def rank_error(delta_k: torch.Tensor, rank: int, niter: int) -> float:
    # delta_k: [seq, hidden] on CPU float32
    m, n = delta_k.shape
    r = min(rank, m, n)
    if r <= 0:
        return 1.0
    denom = float(torch.linalg.norm(delta_k).item())
    if denom <= 1e-12:
        return 0.0
    q = min(max(r + 8, r), min(m, n))
    U, S, V = torch.pca_lowrank(delta_k, q=q, center=False, niter=niter)
    Ur = U[:, :r]
    Sr = S[:r]
    Vr = V[:, :r]
    recon = (Ur * Sr) @ Vr.T
    err = float(torch.linalg.norm(delta_k - recon).item()) / denom
    return err


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 4.4 UpdatableKV rank-k sweep (proxy).")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--adapter", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--prompt", type=str, default="Extract structured fields from document:")
    ap.add_argument("--ranks", type=str, default="8,16,32,64")
    ap.add_argument("--refresh-ks", type=str, default="1,5,10,20,50")
    ap.add_argument("--layer-step", type=int, default=1)
    ap.add_argument("--pca-niter", type=int, default=2)
    ap.add_argument("--quality-threshold", type=float, default=0.300000)
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    if args.seq_len <= 0:
        die("--seq-len must be > 0")
    if args.layer_step <= 0:
        die("--layer-step must be > 0")
    if args.pca_niter < 0:
        die("--pca-niter must be >= 0")

    model_dir = Path(args.model).expanduser().resolve()
    adapter_dir = Path(args.adapter).expanduser().resolve()
    if not model_dir.exists():
        die(f"model not found: {model_dir}")
    if not adapter_dir.exists():
        die(f"adapter not found: {adapter_dir}")

    ranks = sorted(set(split_ints(args.ranks)))
    ks = sorted(set(split_ints(args.refresh_ks)))

    tdtype = dtype_from_name(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
    )
    input_ids = build_input_ids(tokenizer, args.prompt, args.seq_len, args.device)

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=tdtype,
    )
    model.eval().to(args.device)
    base_k = capture_k_proj_outputs(model, input_ids, layer_step=args.layer_step)

    peft_model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    peft_model.eval()
    lora_k = capture_k_proj_outputs(peft_model, input_ids, layer_step=args.layer_step)

    layer_ids = sorted(set(base_k.keys()) & set(lora_k.keys()))
    if not layer_ids:
        die("no overlapping captured layers between base and lora")

    layer_rank_rows: List[Dict[str, str]] = []
    by_rank_errors: Dict[int, List[float]] = {r: [] for r in ranks}
    for lid in layer_ids:
        delta_k = (lora_k[lid] - base_k[lid]).contiguous()
        delta_norm = float(torch.linalg.norm(delta_k).item())
        for r in ranks:
            rel_err = rank_error(delta_k=delta_k, rank=r, niter=args.pca_niter)
            by_rank_errors[r].append(rel_err)
            layer_rank_rows.append(
                {
                    "layer": str(lid),
                    "rank": str(r),
                    "delta_k_fro_norm": f"{delta_norm:.8f}",
                    "rel_fro_error": f"{rel_err:.8f}",
                }
            )

    rank_rows: List[Dict[str, str]] = []
    for r in ranks:
        errs = by_rank_errors[r]
        if not errs:
            continue
        rank_rows.append(
            {
                "rank": str(r),
                "num_layers": str(len(errs)),
                "mean_rel_error": f"{(sum(errs) / len(errs)):.8f}",
                "p95_rel_error": f"{quantile(errs, 0.95):.8f}",
                "max_rel_error": f"{max(errs):.8f}",
            }
        )

    rank_k_rows: List[Dict[str, str]] = []
    for rr in rank_rows:
        r = int(rr["rank"])
        mean_e = float(rr["mean_rel_error"])
        p95_e = float(rr["p95_rel_error"])
        max_e = float(rr["max_rel_error"])
        for k in ks:
            # Proxy for no-refresh drift within a k-step window:
            # average age of cache state in cycle ~= (k-1)/2.
            drift = (k - 1) / 2.0
            mean_proxy = mean_e * (1.0 + drift)
            p95_proxy = p95_e * (1.0 + drift)
            max_proxy = max_e * (1.0 + drift)
            gate_ok = "1" if p95_proxy <= args.quality_threshold else "0"
            rank_k_rows.append(
                {
                    "rank": str(r),
                    "refresh_k": str(k),
                    "proxy_mean_rel_error": f"{mean_proxy:.8f}",
                    "proxy_p95_rel_error": f"{p95_proxy:.8f}",
                    "proxy_max_rel_error": f"{max_proxy:.8f}",
                    "gate_ok": gate_ok,
                }
            )

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (Path.cwd() / "reports" / f"stage44_updatablekv_sweep_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_layer_rank_csv = out_dir / "stage44_layer_rank_errors.csv"
    out_rank_csv = out_dir / "stage44_rank_summary.csv"
    out_rank_k_csv = out_dir / "stage44_rank_k_sweep.csv"
    out_md = out_dir / "stage44_summary.md"

    write_csv(out_layer_rank_csv, layer_rank_rows)
    write_csv(out_rank_csv, rank_rows)
    write_csv(out_rank_k_csv, rank_k_rows)

    best_rows = [r for r in rank_k_rows if r.get("gate_ok", "0") == "1"]
    if best_rows:
        best = min(best_rows, key=lambda r: float(r["proxy_mean_rel_error"]))
        best_line = (
            f"- Best feasible (proxy): rank={best['rank']}, k={best['refresh_k']}, "
            f"proxy_p95={best['proxy_p95_rel_error']}"
        )
    else:
        best_line = "- No feasible rank-k pair under current threshold."

    lines = [
        "# Stage 4.4 UpdatableKV Sweep (Proxy)",
        "",
        f"- Generated at: `{dt.datetime.now().isoformat(timespec='seconds')}`",
        f"- Model: `{model_dir}`",
        f"- Adapter: `{adapter_dir}`",
        f"- Device/dtype: `{args.device}` / `{args.dtype}`",
        f"- seq_len={args.seq_len}, layer_step={args.layer_step}, pca_niter={args.pca_niter}",
        f"- Quality threshold (proxy p95): `{args.quality_threshold:.6f}`",
        "",
        "## Rank Summary",
        "| rank | num_layers | mean_rel_error | p95_rel_error | max_rel_error |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in rank_rows:
        lines.append(
            f"| {r['rank']} | {r['num_layers']} | {r['mean_rel_error']} | {r['p95_rel_error']} | {r['max_rel_error']} |"
        )
    lines.extend(
        [
            "",
            "## Rank x Refresh-k (Proxy)",
            "| rank | refresh_k | proxy_mean_rel_error | proxy_p95_rel_error | proxy_max_rel_error | gate_ok |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for r in rank_k_rows:
        lines.append(
            f"| {r['rank']} | {r['refresh_k']} | {r['proxy_mean_rel_error']} | "
            f"{r['proxy_p95_rel_error']} | {r['proxy_max_rel_error']} | {r['gate_ok']} |"
        )
    lines.extend(["", "## Gate", best_line])
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("[done] stage4.4 updatablekv sweep")
    print(f"- layer-rank csv: {out_layer_rank_csv}")
    print(f"- rank summary csv: {out_rank_csv}")
    print(f"- rank-k csv: {out_rank_k_csv}")
    print(f"- summary md: {out_md}")


if __name__ == "__main__":
    main()

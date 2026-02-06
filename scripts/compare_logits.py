#!/usr/bin/env python3
import argparse
import array
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_tokens(path):
    with open(path, "r", encoding="utf-8") as f:
        data = f.read().strip()
    if not data:
        return []
    return [int(x) for x in data.split()]


def read_logits(path, vocab_size):
    arr = array.array("f")
    with open(path, "rb") as f:
        arr.fromfile(f, vocab_size)
    if len(arr) != vocab_size:
        raise ValueError(f"logits size mismatch: got {len(arr)}, expected {vocab_size}")
    return torch.tensor(arr, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description="Compare Ember logits with HF reference.")
    parser.add_argument("--model", required=True, help="Model directory (local HF cache).")
    parser.add_argument("--debug-dir", required=True, help="Ember debug dir (contains meta.json).")
    parser.add_argument("--topk", type=int, default=10, help="Top-K to display.")
    parser.add_argument("--max-abs-threshold", type=float, default=None,
                        help="Fail if max_abs_diff exceeds this value.")
    parser.add_argument("--mean-abs-threshold", type=float, default=None,
                        help="Fail if mean_abs_diff exceeds this value.")
    args = parser.parse_args()

    meta_path = os.path.join(args.debug_dir, "meta.json")
    tokens_path = os.path.join(args.debug_dir, "tokens.txt")
    logits_path = os.path.join(args.debug_dir, "logits.bin")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    vocab_size = int(meta["vocab_size"])

    tokens = read_tokens(tokens_path)
    if not tokens:
        print("No tokens found in tokens.txt", file=sys.stderr)
        return 1

    ember_logits = read_logits(logits_path, vocab_size)

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
    with torch.no_grad():
        outputs = model(input_ids, use_cache=False)
        hf_logits = outputs.logits[0, -1].float().cpu()

    diff = (hf_logits - ember_logits).abs()
    max_diff = float(diff.max().item())
    mean_diff = float(diff.mean().item())

    print(f"max_abs_diff: {max_diff:.6f}")
    print(f"mean_abs_diff: {mean_diff:.6f}")

    if args.max_abs_threshold is not None and max_diff > args.max_abs_threshold:
        print(f"[Fail] max_abs_diff {max_diff:.6f} > {args.max_abs_threshold:.6f}",
              file=sys.stderr)
        return 2
    if args.mean_abs_threshold is not None and mean_diff > args.mean_abs_threshold:
        print(f"[Fail] mean_abs_diff {mean_diff:.6f} > {args.mean_abs_threshold:.6f}",
              file=sys.stderr)
        return 3

    topk = max(1, args.topk)
    hf_vals, hf_idx = torch.topk(hf_logits, topk)
    em_vals, em_idx = torch.topk(ember_logits, topk)

    print("\nHF top-k:")
    for score, idx in zip(hf_vals.tolist(), hf_idx.tolist()):
        tok = tokenizer.convert_ids_to_tokens(int(idx))
        print(f"{idx}\t{score:.6f}\t{tok}")

    print("\nEmber top-k:")
    for score, idx in zip(em_vals.tolist(), em_idx.tolist()):
        tok = tokenizer.convert_ids_to_tokens(int(idx))
        print(f"{idx}\t{score:.6f}\t{tok}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

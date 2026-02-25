#!/usr/bin/env python3
import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


def sync_cuda(torch_mod) -> None:
    if torch_mod.cuda.is_available():
        torch_mod.cuda.synchronize()


def build_random_prompt(vocab_size: int, prompt_len: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    hi = max(1, vocab_size - 1)
    return [rng.randint(0, hi) for _ in range(prompt_len)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Transformers rollout benchmark (prefill/decode split).")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--prompt-len", type=int, default=2048)
    ap.add_argument("--decode-steps", type=int, default=128)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--out-json", type=str, default="")
    args = ap.parse_args()

    if args.prompt_len <= 0 or args.decode_steps <= 0:
        die("--prompt-len and --decode-steps must be > 0")
    if args.iters <= 0 or args.warmup < 0:
        die("--iters must be > 0 and --warmup must be >= 0")

    import torch  # type: ignore
    from transformers import AutoModelForCausalLM  # type: ignore

    if not torch.cuda.is_available():
        die("CUDA is not available in this Python environment")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    model_dir = str(Path(args.model).expanduser().resolve())
    cfg_path = Path(model_dir) / "config.json"
    if not cfg_path.exists():
        die(f"missing config.json under model dir: {model_dir}")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    vocab_size = int(cfg.get("vocab_size", 0))
    if vocab_size <= 0:
        die(f"invalid vocab_size in {cfg_path}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            dtype=torch_dtype,
            trust_remote_code=True,
            local_files_only=True,
        ).to(args.device)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            local_files_only=True,
        ).to(args.device)
    model.eval()

    prompt_ids = build_random_prompt(vocab_size, args.prompt_len, args.seed)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)

    prefill_ms_sum = 0.0
    decode_ms_sum = 0.0

    with torch.no_grad():
        for i in range(args.warmup + args.iters):
            sync_cuda(torch)
            t0 = time.perf_counter()
            out = model(input_ids=input_ids, use_cache=True, return_dict=True)
            sync_cuda(torch)
            t1 = time.perf_counter()

            past_key_values = out.past_key_values
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

            t2 = time.perf_counter()
            for _ in range(1, args.decode_steps):
                out = model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = out.past_key_values
                next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            sync_cuda(torch)
            t3 = time.perf_counter()

            if i >= args.warmup:
                prefill_ms_sum += (t1 - t0) * 1000.0
                decode_ms_sum += (t3 - t2) * 1000.0

    prefill_ms = prefill_ms_sum / float(args.iters)
    decode_ms = decode_ms_sum / float(args.iters)
    decode_per_token_ms = decode_ms / float(args.decode_steps)
    rollout_total_ms = prefill_ms + decode_ms
    rollout_tok_s = (args.decode_steps * 1000.0 / rollout_total_ms) if rollout_total_ms > 0.0 else 0.0

    result: Dict[str, object] = {
        "engine": "transformers",
        "model_dir": model_dir,
        "device": args.device,
        "dtype": args.dtype,
        "prompt_len": args.prompt_len,
        "decode_steps": args.decode_steps,
        "prefill_ms": prefill_ms,
        "decode_per_token_ms": decode_per_token_ms,
        "rollout_total_ms": rollout_total_ms,
        "rollout_tok_s": rollout_tok_s,
        "iters": args.iters,
        "warmup": args.warmup,
    }

    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out_json.strip():
        out_path = Path(args.out_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

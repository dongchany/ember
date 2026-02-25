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


def build_random_tokens(vocab_size: int, n: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    hi = max(1, vocab_size - 1)
    return [rng.randint(0, hi) for _ in range(n)]


def load_vocab_size(model_dir: Path) -> int:
    cfg = model_dir / "config.json"
    if not cfg.exists():
        die(f"missing config.json: {cfg}")
    obj = json.loads(cfg.read_text(encoding="utf-8"))
    v = int(obj.get("vocab_size", 0))
    if v <= 0:
        die(f"invalid vocab_size in {cfg}")
    return v


def timed_generate(llm, prompts, sampling_params, use_tqdm: bool = False) -> float:
    t0 = time.perf_counter()
    _ = llm.generate(prompts=prompts, sampling_params=sampling_params, use_tqdm=use_tqdm)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


def main() -> None:
    ap = argparse.ArgumentParser(description="vLLM rollout benchmark with prefill/decode proxy split.")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--prompt-len", type=int, default=2048)
    ap.add_argument("--decode-steps", type=int, default=128)
    ap.add_argument("--dtype", type=str, default="float16")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    ap.add_argument("--max-model-len", type=int, default=0, help="0 means prompt_len + decode_steps + 64")
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--enforce-eager", action="store_true", default=False)
    ap.add_argument("--out-json", type=str, default="")
    args = ap.parse_args()

    if args.prompt_len <= 0 or args.decode_steps <= 0:
        die("--prompt-len and --decode-steps must be > 0")
    if args.tensor_parallel_size <= 0:
        die("--tensor-parallel-size must be > 0")
    if args.max_num_seqs <= 0:
        die("--max-num-seqs must be > 0")
    if args.iters <= 0 or args.warmup < 0:
        die("--iters must be > 0 and --warmup must be >= 0")

    from vllm import LLM, SamplingParams  # type: ignore

    model_dir = Path(args.model).expanduser().resolve()
    vocab_size = load_vocab_size(model_dir)
    max_model_len = args.max_model_len if args.max_model_len > 0 else (args.prompt_len + args.decode_steps + 64)
    total_runs = args.warmup + args.iters
    long_prompts = [
        {"prompt_token_ids": build_random_tokens(vocab_size, args.prompt_len, args.seed + 1000 + i)}
        for i in range(total_runs)
    ]
    short_prompts_single = [
        {"prompt_token_ids": build_random_tokens(vocab_size, 1, args.seed + 2000 + i)}
        for i in range(total_runs)
    ]
    short_prompts_full = [
        {"prompt_token_ids": build_random_tokens(vocab_size, 1, args.seed + 3000 + i)}
        for i in range(total_runs)
    ]

    llm = LLM(
        model=str(model_dir),
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=args.max_num_seqs,
        seed=args.seed,
        enforce_eager=args.enforce_eager,
    )

    prefill_plus_one = SamplingParams(max_tokens=1, temperature=0.0)
    decode_single = SamplingParams(max_tokens=1, temperature=0.0)
    decode_full = SamplingParams(max_tokens=args.decode_steps, temperature=0.0)

    prefill_plus_one_ms_sum = 0.0
    decode_single_ms_sum = 0.0
    decode_full_ms_sum = 0.0

    for i in range(total_runs):
        a = timed_generate(llm, [long_prompts[i]], prefill_plus_one, use_tqdm=False)
        b = timed_generate(llm, [short_prompts_single[i]], decode_single, use_tqdm=False)
        c = timed_generate(llm, [short_prompts_full[i]], decode_full, use_tqdm=False)
        if i >= args.warmup:
            prefill_plus_one_ms_sum += a
            decode_single_ms_sum += b
            decode_full_ms_sum += c

    prefill_plus_one_ms = prefill_plus_one_ms_sum / float(args.iters)
    decode_single_ms = decode_single_ms_sum / float(args.iters)
    decode_full_ms = decode_full_ms_sum / float(args.iters)

    prefill_ms = max(0.0, prefill_plus_one_ms - decode_single_ms)
    decode_per_token_ms = decode_full_ms / float(args.decode_steps)
    rollout_total_ms = prefill_ms + decode_per_token_ms * float(args.decode_steps)
    rollout_tok_s = (args.decode_steps * 1000.0 / rollout_total_ms) if rollout_total_ms > 0.0 else 0.0

    out: Dict[str, object] = {
        "engine": "vllm",
        "model_dir": str(model_dir),
        "prompt_len": args.prompt_len,
        "decode_steps": args.decode_steps,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "prefill_proxy_plus_one_ms": prefill_plus_one_ms,
        "decode_proxy_single_ms": decode_single_ms,
        "decode_proxy_full_ms": decode_full_ms,
        "prefill_ms": prefill_ms,
        "decode_per_token_ms": decode_per_token_ms,
        "rollout_total_ms": rollout_total_ms,
        "rollout_tok_s": rollout_tok_s,
        "iters": args.iters,
        "warmup": args.warmup,
    }
    text = json.dumps(out, ensure_ascii=False, indent=2)
    if args.out_json.strip():
        p = Path(args.out_json).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

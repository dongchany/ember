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


def timed_generate(engine, input_ids, sampling_params: Dict[str, object]) -> float:
    t0 = time.perf_counter()
    _ = engine.generate(input_ids=input_ids, sampling_params=sampling_params)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


def main() -> None:
    ap = argparse.ArgumentParser(description="SGLang rollout benchmark with prefill/decode proxy split.")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--prompt-len", type=int, default=2048)
    ap.add_argument("--decode-steps", type=int, default=128)
    ap.add_argument("--dtype", type=str, default="float16")
    ap.add_argument("--tp-size", type=int, default=1)
    ap.add_argument("--mem-fraction-static", type=float, default=0.8)
    ap.add_argument("--context-length", type=int, default=0, help="0 means prompt_len + decode_steps + 64")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--out-json", type=str, default="")
    args = ap.parse_args()

    if args.prompt_len <= 0 or args.decode_steps <= 0:
        die("--prompt-len and --decode-steps must be > 0")
    if args.tp_size <= 0:
        die("--tp-size must be > 0")
    if args.iters <= 0 or args.warmup < 0:
        die("--iters must be > 0 and --warmup must be >= 0")

    from sglang.srt.entrypoints.engine import Engine  # type: ignore

    model_dir = Path(args.model).expanduser().resolve()
    vocab_size = load_vocab_size(model_dir)
    context_length = args.context_length if args.context_length > 0 else (args.prompt_len + args.decode_steps + 64)
    total_runs = args.warmup + args.iters
    long_prompts = [
        build_random_tokens(vocab_size, args.prompt_len, args.seed + 1000 + i)
        for i in range(total_runs)
    ]
    short_prompts_single = [
        build_random_tokens(vocab_size, 1, args.seed + 2000 + i)
        for i in range(total_runs)
    ]
    short_prompts_full = [
        build_random_tokens(vocab_size, 1, args.seed + 3000 + i)
        for i in range(total_runs)
    ]

    engine = Engine(
        model_path=str(model_dir),
        trust_remote_code=True,
        skip_tokenizer_init=True,
        tp_size=args.tp_size,
        context_length=context_length,
        dtype=args.dtype,
        mem_fraction_static=args.mem_fraction_static,
        random_seed=args.seed,
        log_level="error",
    )

    prefill_plus_one_params = {"max_new_tokens": 1, "temperature": 0.0}
    decode_single_params = {"max_new_tokens": 1, "temperature": 0.0}
    decode_full_params = {"max_new_tokens": args.decode_steps, "temperature": 0.0}

    prefill_plus_one_ms_sum = 0.0
    decode_single_ms_sum = 0.0
    decode_full_ms_sum = 0.0

    try:
        for i in range(total_runs):
            a = timed_generate(engine, long_prompts[i], prefill_plus_one_params)
            b = timed_generate(engine, short_prompts_single[i], decode_single_params)
            c = timed_generate(engine, short_prompts_full[i], decode_full_params)
            if i >= args.warmup:
                prefill_plus_one_ms_sum += a
                decode_single_ms_sum += b
                decode_full_ms_sum += c
    finally:
        engine.shutdown()

    prefill_plus_one_ms = prefill_plus_one_ms_sum / float(args.iters)
    decode_single_ms = decode_single_ms_sum / float(args.iters)
    decode_full_ms = decode_full_ms_sum / float(args.iters)

    prefill_ms = max(0.0, prefill_plus_one_ms - decode_single_ms)
    decode_per_token_ms = decode_full_ms / float(args.decode_steps)
    rollout_total_ms = prefill_ms + decode_per_token_ms * float(args.decode_steps)
    rollout_tok_s = (args.decode_steps * 1000.0 / rollout_total_ms) if rollout_total_ms > 0.0 else 0.0

    out: Dict[str, object] = {
        "engine": "sglang",
        "model_dir": str(model_dir),
        "prompt_len": args.prompt_len,
        "decode_steps": args.decode_steps,
        "dtype": args.dtype,
        "tp_size": args.tp_size,
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

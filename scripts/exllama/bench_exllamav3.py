#!/usr/bin/env python3
import argparse
import csv
import gc
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch


@dataclass
class BenchResult:
    mode: str  # "single" | "tp"
    gpus: str  # "CUDA0" | "CUDA0/CUDA1"
    prompt_len: int
    gen_len: int
    batch_size: int
    iters: int
    warmup: int
    attn_mode: str
    max_seq_len: int
    ttft_ms: float
    prefill_ms: float
    decode_ms: float
    decode_tok_s: float


def _round_up(n: int, m: int) -> int:
    return ((n + m - 1) // m) * m


def sync_all() -> None:
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)


def _parse_devices(s: str) -> List[int]:
    out: List[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok == "":
            continue
        out.append(int(tok))
    return out


def _bench_once(model_dir: Path,
                mode: str,
                devices: List[int],
                prompt_len: int,
                gen_len: int,
                batch_size: int,
                iters: int,
                warmup: int,
                attn_mode: str,
                max_seq_len: int,
                tp_backend: str,
                reserve_gb: float) -> BenchResult:
    if batch_size != 1:
        raise ValueError("this benchmark currently supports batch_size=1 only")

    from exllamav3 import Cache, Config, GreedySampler, Model, Tokenizer

    cfg = Config.from_directory(str(model_dir))
    vocab = int(getattr(cfg, "vocab_size"))
    if vocab <= 0:
        raise RuntimeError("invalid vocab_size from config")

    # Each scenario constructs its own model to keep memory accounting simple.
    model = Model.from_config(cfg)
    cache = Cache(model, max_num_tokens=max_seq_len)
    tok = Tokenizer.from_config(cfg)
    sampler = GreedySampler()

    sync_all()
    torch.cuda.empty_cache()

    if mode == "single":
        model.load(device=devices[0], progressbar=False)
        gpus_str = f"CUDA{devices[0]}"
    elif mode == "tp":
        # Use reserve_per_device to select a subset of GPUs.
        dev_count = torch.cuda.device_count()
        if dev_count <= 0:
            raise RuntimeError("CUDA device_count=0")
        rpd: List[float] = [-1.0] * dev_count
        for d in devices:
            if d < 0 or d >= dev_count:
                raise ValueError(f"invalid device id: {d}")
            rpd[d] = float(reserve_gb)
        # Gather logits on the first device to match a "GPU0 baseline" comparison.
        model.load(tensor_p=True,
                   reserve_per_device=rpd,
                   tp_output_device=devices[0],
                   tp_backend=tp_backend,
                   progressbar=False)
        gpus_str = "/".join(f"CUDA{d}" for d in devices)
    else:
        raise ValueError(f"unknown mode: {mode}")

    # Warmup.
    gen = torch.Generator().manual_seed(1234)
    for _ in range(warmup):
        ids = torch.randint(0, vocab, (1, prompt_len), dtype=torch.long, generator=gen)
        params = {"attn_mode": attn_mode, "cache": cache, "past_len": 0, "batch_shape": (batch_size, max_seq_len)}
        model.prefill(ids[:, :-1], params=params)
        params = {"attn_mode": attn_mode, "cache": cache, "past_len": prompt_len - 1, "batch_shape": (batch_size, max_seq_len)}
        _ = model.forward(ids[:, -1:], params=params)
        sync_all()

    ttft_s_sum = 0.0
    prefill_s_sum = 0.0
    decode_s_sum = 0.0

    for _ in range(iters):
        ids = torch.randint(0, vocab, (1, prompt_len), dtype=torch.long, generator=gen)

        sync_all()
        t0 = time.perf_counter()

        params = {"attn_mode": attn_mode, "cache": cache, "past_len": 0, "batch_shape": (batch_size, max_seq_len)}
        model.prefill(ids[:, :-1], params=params)
        sync_all()
        t_prefill_end = time.perf_counter()

        # Decode: generate gen_len tokens (including the first token).
        t_decode_start = t_prefill_end

        context_ids = ids
        past_len = prompt_len - 1
        recurrent_states = None

        # First token (used for TTFT).
        params = {
            "attn_mode": attn_mode,
            "cache": cache,
            "past_len": past_len,
            "batch_shape": (batch_size, max_seq_len),
            "recurrent_states": recurrent_states,
        }
        logits = model.forward(context_ids[:, -1:], params=params)
        sample = sampler.forward(logits, tokenizer=tok)
        recurrent_states = params.get("recurrent_states")
        context_ids = torch.cat((context_ids, sample.cpu()), dim=-1)
        past_len += 1

        sync_all()
        t_first_end = time.perf_counter()

        # Remaining tokens.
        for _ in range(gen_len - 1):
            params = {
                "attn_mode": attn_mode,
                "cache": cache,
                "past_len": past_len,
                "batch_shape": (batch_size, max_seq_len),
                "recurrent_states": recurrent_states,
            }
            logits = model.forward(context_ids[:, -1:], params=params)
            sample = sampler.forward(logits, tokenizer=tok)
            recurrent_states = params.get("recurrent_states")
            context_ids = torch.cat((context_ids, sample.cpu()), dim=-1)
            past_len += 1

        sync_all()
        t_decode_end = time.perf_counter()

        ttft_s_sum += (t_first_end - t0)
        prefill_s_sum += (t_prefill_end - t0)
        decode_s_sum += (t_decode_end - t_decode_start)

    # Avoid holding onto VRAM across scenarios.
    model.unload()
    del cache
    del model
    gc.collect()
    torch.cuda.empty_cache()
    sync_all()

    ttft_ms = (ttft_s_sum / iters) * 1000.0
    prefill_ms = (prefill_s_sum / iters) * 1000.0
    decode_ms = (decode_s_sum / iters) * 1000.0
    decode_tok_s = (gen_len * iters) / decode_s_sum if decode_s_sum > 0 else 0.0

    return BenchResult(
        mode=mode,
        gpus=gpus_str,
        prompt_len=prompt_len,
        gen_len=gen_len,
        batch_size=batch_size,
        iters=iters,
        warmup=warmup,
        attn_mode=attn_mode,
        max_seq_len=max_seq_len,
        ttft_ms=ttft_ms,
        prefill_ms=prefill_ms,
        decode_ms=decode_ms,
        decode_tok_s=decode_tok_s,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="ExLlamaV3 EXL3 benchmark (TTFT + decode tok/s).")
    ap.add_argument("--model", required=True, type=str, help="EXL3 model directory")
    ap.add_argument("--prompt-len", type=int, default=2048)
    ap.add_argument("--gen-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--attn-mode", type=str, default="flash_attn", choices=["flash_attn", "flash_attn_nc", "sdpa_nc"])
    ap.add_argument("--devices", type=str, default="0,1", help="GPU ids for TP mode, e.g. 0,1")
    ap.add_argument("--single-device", type=int, default=0, help="GPU id for single-GPU baseline")
    ap.add_argument("--tp-backend", type=str, default="native", choices=["native", "nccl"])
    ap.add_argument("--reserve-gb", type=float, default=0.5, help="reserve (GB) on each selected GPU for TP mode")
    ap.add_argument("--max-seq-len", type=int, default=0, help="override cache max seq len (rounded up to 256)")
    ap.add_argument("--csv", type=str, default="", help="write results CSV to this path (default: stdout)")
    args = ap.parse_args()

    model_dir = Path(args.model).expanduser().resolve()
    if not (model_dir / "config.json").exists():
        raise SystemExit(f"missing config.json under: {model_dir}")

    if args.prompt_len <= 0 or args.gen_len <= 0:
        raise SystemExit("--prompt-len/--gen-len must be > 0")
    if args.iters <= 0 or args.warmup < 0:
        raise SystemExit("--iters must be > 0 and --warmup must be >= 0")
    if args.batch_size != 1:
        raise SystemExit("this script currently supports --batch-size 1 only")

    total_seq = args.prompt_len + args.gen_len
    max_seq_len = args.max_seq_len if args.max_seq_len > 0 else total_seq
    max_seq_len = _round_up(max_seq_len, 256)

    # Benchmark both scenarios.
    single = _bench_once(
        model_dir=model_dir,
        mode="single",
        devices=[args.single_device],
        prompt_len=args.prompt_len,
        gen_len=args.gen_len,
        batch_size=args.batch_size,
        iters=args.iters,
        warmup=args.warmup,
        attn_mode=args.attn_mode,
        max_seq_len=max_seq_len,
        tp_backend=args.tp_backend,
        reserve_gb=args.reserve_gb,
    )
    tp_devs = _parse_devices(args.devices)
    if len(tp_devs) < 2:
        raise SystemExit("--devices must include at least 2 GPU ids for TP run")
    tp = _bench_once(
        model_dir=model_dir,
        mode="tp",
        devices=tp_devs,
        prompt_len=args.prompt_len,
        gen_len=args.gen_len,
        batch_size=args.batch_size,
        iters=args.iters,
        warmup=args.warmup,
        attn_mode=args.attn_mode,
        max_seq_len=max_seq_len,
        tp_backend=args.tp_backend,
        reserve_gb=args.reserve_gb,
    )

    speedup = (tp.decode_tok_s / single.decode_tok_s) if single.decode_tok_s > 0 else 0.0

    out = None
    f = None
    if args.csv:
        p = Path(args.csv).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        f = p.open("w", encoding="utf-8", newline="")
        out = f
    else:
        out = None

    headers = [
        "mode",
        "gpus",
        "prompt_len",
        "gen_len",
        "batch_size",
        "iters",
        "warmup",
        "attn_mode",
        "max_seq_len",
        "ttft_ms",
        "prefill_ms",
        "decode_ms",
        "decode_tok_s",
        "speedup_vs_single",
    ]
    rows = []
    for r in (single, tp):
        rows.append([
            r.mode,
            r.gpus,
            str(r.prompt_len),
            str(r.gen_len),
            str(r.batch_size),
            str(r.iters),
            str(r.warmup),
            r.attn_mode,
            str(r.max_seq_len),
            f"{r.ttft_ms:.3f}",
            f"{r.prefill_ms:.3f}",
            f"{r.decode_ms:.3f}",
            f"{r.decode_tok_s:.3f}",
            f"{(1.0 if r.mode == 'single' else speedup):.4f}",
        ])

    if out is None:
        w = csv.writer(sys.stdout)
        w.writerow(headers)
        w.writerows(rows)
    else:
        w = csv.writer(out)
        w.writerow(headers)
        w.writerows(rows)
        f.close()

    # Also print a short summary to stderr for interactive runs.
    print(
        f"[exllama] single decode_tok_s={single.decode_tok_s:.3f}, tp decode_tok_s={tp.decode_tok_s:.3f}, speedup={speedup:.3f}",
        file=sys.stderr,
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Stage:
    name: str
    compute_us: float


def simulate_2stage(
    stage0_us: float,
    stage1_us: float,
    transfer_us: float,
    micro_batches: int,
) -> Tuple[float, float, float]:
    """
    Very small discrete-event sim of a 2-stage pipeline:
      S0 compute -> transfer -> S1 compute

    Returns:
      total_us, util0, util1
    """
    t0 = 0.0
    t1 = 0.0
    busy0 = 0.0
    busy1 = 0.0

    for _ in range(micro_batches):
        start0 = t0
        end0 = start0 + stage0_us
        busy0 += stage0_us
        t0 = end0

        # transfer can only start after S0 finishes this micro-batch, and
        # must complete before S1 can start it.
        arrive1 = end0 + transfer_us

        start1 = max(t1, arrive1)
        end1 = start1 + stage1_us
        busy1 += stage1_us
        t1 = end1

    total = max(t0, t1)
    util0 = busy0 / total if total > 0 else 0.0
    util1 = busy1 / total if total > 0 else 0.0
    return total, util0, util1


def main():
    ap = argparse.ArgumentParser(description="Ember naive pipeline bubble model (2 GPUs).")
    ap.add_argument("--phase", choices=["prefill", "decode"], required=True)
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--split", type=str, default="20,20", help="e.g. 20,20")
    ap.add_argument("--layer-us", type=float, required=True, help="per-layer compute time (us) for this phase")
    ap.add_argument("--transfer-us", type=float, required=True, help="activation transfer time per boundary (us)")
    ap.add_argument("--micro-batches", type=int, default=1, help="prefill micro-batches or decode steps grouped")
    args = ap.parse_args()

    a, b = [int(x) for x in args.split.split(",")]
    if a + b != args.layers:
        raise SystemExit(f"split {a},{b} does not sum to --layers {args.layers}")

    stage0 = a * args.layer_us
    stage1 = b * args.layer_us

    total_us, util0, util1 = simulate_2stage(stage0, stage1, args.transfer_us, args.micro_batches)
    avg_util = 0.5 * (util0 + util1)
    bubble = 1.0 - avg_util

    print("phase,split,micro_batches,total_ms,bubble_ratio,gpu0_util,gpu1_util")
    print(
        f"{args.phase},{a}/{b},{args.micro_batches},{total_us/1000.0:.3f},"
        f"{bubble*100.0:.2f}%,{util0*100.0:.2f}%,{util1*100.0:.2f}%"
    )


if __name__ == "__main__":
    main()


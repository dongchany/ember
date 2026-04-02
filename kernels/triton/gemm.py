from __future__ import annotations

from typing import Callable

import torch
import triton
import triton.language as tl


_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=4, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_M": 8}, num_warps=8, num_stages=4),
]


@triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    IS_BF16: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_in_group = pid % num_pid_in_group

    # Grouped launch order improves locality compared with plain row-major tiles.
    pid_m = first_pid_m + (pid_in_group % group_size_m)
    pid_n = pid_in_group // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        k_offsets = k_start * BLOCK_K + offs_k
        a_mask = (offs_m[:, None] < M) & (k_offsets[None, :] < K)
        b_mask = (k_offsets[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = accumulator.to(tl.bfloat16) if IS_BF16 else accumulator.to(tl.float16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def _validate_inputs(a: torch.Tensor, b: torch.Tensor) -> None:
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("matmul expects 2D tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"incompatible shapes: {tuple(a.shape)} x {tuple(b.shape)}")
    if a.dtype != b.dtype:
        raise ValueError(f"dtype mismatch: {a.dtype} != {b.dtype}")
    if a.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"matmul supports only float16 and bfloat16, got {a.dtype}")
    if a.device != b.device:
        raise ValueError(f"device mismatch: {a.device} != {b.device}")
    if a.device.type != "cuda":
        raise ValueError("matmul requires CUDA tensors")


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    _validate_inputs(a, b)

    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    _gemm_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        IS_BF16=a.dtype == torch.bfloat16,
    )
    return c


def _benchmark_ms(fn: Callable[[], torch.Tensor], warmup: int = 10, iters: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def benchmark_gemm(
    M: int = 4096,
    N: int = 4096,
    K: int = 4096,
    dtype: torch.dtype = torch.float16,
) -> float:
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_gemm requires a CUDA device")
    if dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"benchmark_gemm supports only float16 and bfloat16, got {dtype}")

    device = torch.device("cuda")
    a = torch.randn((M, K), device=device, dtype=dtype)
    b = torch.randn((K, N), device=device, dtype=dtype)

    triton_ms = _benchmark_ms(lambda: matmul(a, b))
    torch_ms = _benchmark_ms(lambda: torch.matmul(a, b))
    total_flops = 2.0 * M * N * K
    triton_tflops = total_flops / (triton_ms * 1e9)
    torch_tflops = total_flops / (torch_ms * 1e9)
    speedup = torch_ms / triton_ms

    print(
        f"GEMM {M}x{K} @ {K}x{N} [{dtype}]\n"
        f"  Triton:       {triton_ms:.3f} ms  {triton_tflops:.2f} TFLOPS\n"
        f"  torch.matmul: {torch_ms:.3f} ms  {torch_tflops:.2f} TFLOPS\n"
        f"  Speedup:      {speedup:.2f}x"
    )
    return speedup


if __name__ == "__main__":
    benchmark_gemm()

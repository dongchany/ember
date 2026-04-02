from __future__ import annotations

from typing import Callable

import torch
import triton
import triton.language as tl


_SUPPORTED_X_DTYPES = (torch.float16, torch.bfloat16)
_SUPPORTED_WEIGHT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_MAX_HIDDEN_SIZE = 8192


@triton.jit
def _fused_rmsnorm_residual_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    output_ptr,
    updated_residual_ptr,
    n_cols,
    eps,
    stride_x_row,
    stride_residual_row,
    stride_output_row,
    stride_updated_residual_row,
    IS_BF16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    x = tl.load(x_ptr + row_idx * stride_x_row + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + row_idx * stride_residual_row + offsets, mask=mask, other=0.0).to(tl.float32)
    updated_residual = x + residual
    mean_square = tl.sum(updated_residual * updated_residual, axis=0) / n_cols
    inv_rms = tl.rsqrt(mean_square + eps)

    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    output = updated_residual * inv_rms * weight

    output_ptrs = output_ptr + row_idx * stride_output_row + offsets
    updated_residual_ptrs = updated_residual_ptr + row_idx * stride_updated_residual_row + offsets
    if IS_BF16:
        tl.store(output_ptrs, output.to(tl.bfloat16), mask=mask)
        tl.store(updated_residual_ptrs, updated_residual.to(tl.bfloat16), mask=mask)
    else:
        tl.store(output_ptrs, output.to(tl.float16), mask=mask)
        tl.store(updated_residual_ptrs, updated_residual.to(tl.float16), mask=mask)


@triton.jit
def _rmsnorm_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    n_cols,
    eps,
    stride_x_row,
    stride_output_row,
    IS_BF16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    x = tl.load(x_ptr + row_idx * stride_x_row + offsets, mask=mask, other=0.0).to(tl.float32)
    mean_square = tl.sum(x * x, axis=0) / n_cols
    inv_rms = tl.rsqrt(mean_square + eps)

    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    output = x * inv_rms * weight

    output_ptrs = output_ptr + row_idx * stride_output_row + offsets
    if IS_BF16:
        tl.store(output_ptrs, output.to(tl.bfloat16), mask=mask)
    else:
        tl.store(output_ptrs, output.to(tl.float16), mask=mask)


def _validate_x(x: torch.Tensor) -> None:
    if x.ndim < 1:
        raise ValueError("expected at least 1 dimension")
    if x.dtype not in _SUPPORTED_X_DTYPES:
        raise TypeError(f"supported input dtypes are {_SUPPORTED_X_DTYPES}, got {x.dtype}")
    if x.device.type != "cuda":
        raise ValueError("expected CUDA tensors")


def _validate_weight(weight: torch.Tensor, hidden_size: int, device: torch.device) -> None:
    if weight.ndim != 1:
        raise ValueError("weight must be 1D")
    if weight.shape[0] != hidden_size:
        raise ValueError(f"expected weight shape ({hidden_size},), got {tuple(weight.shape)}")
    if weight.dtype not in _SUPPORTED_WEIGHT_DTYPES:
        raise TypeError(f"supported weight dtypes are {_SUPPORTED_WEIGHT_DTYPES}, got {weight.dtype}")
    if weight.device != device:
        raise ValueError(f"device mismatch: {weight.device} != {device}")


def _hidden_size(x: torch.Tensor) -> int:
    hidden_size = x.shape[-1]
    if hidden_size > _MAX_HIDDEN_SIZE:
        raise ValueError(f"hidden size {hidden_size} exceeds max supported size {_MAX_HIDDEN_SIZE}")
    return hidden_size


def _launch_config(hidden_size: int) -> tuple[int, int]:
    block_size = triton.next_power_of_2(hidden_size)
    if block_size <= 1024:
        num_warps = 4
    elif block_size <= 4096:
        num_warps = 8
    else:
        num_warps = 16
    return block_size, num_warps


def _flatten_rows(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().reshape(-1, x.shape[-1])


def _torch_fused_rmsnorm_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    updated_residual = (x.float() + residual.float()).to(x.dtype)
    mean_square = updated_residual.float().pow(2).mean(dim=-1, keepdim=True)
    normed = (updated_residual.float() * torch.rsqrt(mean_square + eps) * weight.float()).to(x.dtype)
    return normed, updated_residual


def _torch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    mean_square = x.float().pow(2).mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(mean_square + eps) * weight.float()).to(x.dtype)


def fused_rmsnorm_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_x(x)
    if residual.shape != x.shape:
        raise ValueError(f"shape mismatch: {tuple(x.shape)} != {tuple(residual.shape)}")
    if residual.dtype != x.dtype:
        raise TypeError(f"dtype mismatch: {x.dtype} != {residual.dtype}")
    if residual.device != x.device:
        raise ValueError(f"device mismatch: {x.device} != {residual.device}")

    hidden_size = _hidden_size(x)
    _validate_weight(weight, hidden_size, x.device)

    x_rows = _flatten_rows(x)
    residual_rows = _flatten_rows(residual)
    weight_contiguous = weight.contiguous()
    output_rows = torch.empty_like(x_rows)
    updated_residual_rows = torch.empty_like(residual_rows)

    block_size, num_warps = _launch_config(hidden_size)
    grid = (x_rows.shape[0],)
    _fused_rmsnorm_residual_kernel[grid](
        x_rows,
        residual_rows,
        weight_contiguous,
        output_rows,
        updated_residual_rows,
        hidden_size,
        eps,
        x_rows.stride(0),
        residual_rows.stride(0),
        output_rows.stride(0),
        updated_residual_rows.stride(0),
        IS_BF16=x.dtype == torch.bfloat16,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return output_rows.reshape_as(x), updated_residual_rows.reshape_as(residual)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    _validate_x(x)
    hidden_size = _hidden_size(x)
    _validate_weight(weight, hidden_size, x.device)

    x_rows = _flatten_rows(x)
    weight_contiguous = weight.contiguous()
    output_rows = torch.empty_like(x_rows)

    block_size, num_warps = _launch_config(hidden_size)
    grid = (x_rows.shape[0],)
    _rmsnorm_kernel[grid](
        x_rows,
        weight_contiguous,
        output_rows,
        hidden_size,
        eps,
        x_rows.stride(0),
        output_rows.stride(0),
        IS_BF16=x.dtype == torch.bfloat16,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return output_rows.reshape_as(x)


def _benchmark_ms(fn: Callable[[], torch.Tensor | tuple[torch.Tensor, torch.Tensor]], warmup: int = 10, iters: int = 100) -> float:
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


def benchmark_fused_rmsnorm(
    batch_size: int = 4096,
    hidden_size: int = 4096,
    dtype: torch.dtype = torch.float16,
    eps: float = 1e-6,
) -> float:
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_fused_rmsnorm requires a CUDA device")
    if dtype not in _SUPPORTED_X_DTYPES:
        raise TypeError(f"supported benchmark dtypes are {_SUPPORTED_X_DTYPES}, got {dtype}")

    device = torch.device("cuda")
    x = torch.randn((batch_size, hidden_size), device=device, dtype=dtype)
    residual = torch.randn_like(x)
    weight = torch.randn((hidden_size,), device=device, dtype=dtype)

    triton_ms = _benchmark_ms(lambda: fused_rmsnorm_residual(x, residual, weight, eps))
    torch_ms = _benchmark_ms(lambda: _torch_fused_rmsnorm_residual(x, residual, weight, eps))
    speedup = torch_ms / triton_ms

    print(
        f"Fused RMSNorm+Residual {batch_size}x{hidden_size} [{dtype}]\n"
        f"  Triton:     {triton_ms:.3f} ms\n"
        f"  PyTorch:    {torch_ms:.3f} ms\n"
        f"  Speedup:    {speedup:.2f}x"
    )
    return speedup


if __name__ == "__main__":
    benchmark_fused_rmsnorm()

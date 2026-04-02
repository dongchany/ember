from __future__ import annotations

from typing import Callable

import torch
import triton
import triton.language as tl


_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)
_MAX_HEAD_DIM = 1024


def precompute_freqs(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim <= 0:
        raise ValueError("head_dim must be positive")
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even")
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    if theta <= 0.0:
        raise ValueError("theta must be positive")

    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freq_indices = torch.arange(0, head_dim, 2, dtype=torch.float32)
    inv_freq = theta ** (-freq_indices / head_dim)
    freqs = torch.outer(positions, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


@triton.jit
def rope_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    position_ptr,
    output_ptr,
    num_rows,
    num_heads,
    half_dim,
    stride_x_row,
    stride_cos_row,
    stride_sin_row,
    stride_output_row,
    stride_position,
    IS_BF16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(axis=0)
    pair_offsets = tl.arange(0, BLOCK_SIZE)
    mask = (row_idx < num_rows) & (pair_offsets < half_dim)

    token_idx = row_idx // num_heads
    position = tl.load(position_ptr + token_idx * stride_position)

    x_even = tl.load(x_ptr + row_idx * stride_x_row + pair_offsets * 2, mask=mask, other=0.0).to(tl.float32)
    x_odd = tl.load(x_ptr + row_idx * stride_x_row + pair_offsets * 2 + 1, mask=mask, other=0.0).to(tl.float32)
    cos = tl.load(cos_ptr + position * stride_cos_row + pair_offsets, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr + position * stride_sin_row + pair_offsets, mask=mask, other=0.0).to(tl.float32)

    out_even = x_even * cos - x_odd * sin
    out_odd = x_odd * cos + x_even * sin

    even_ptrs = output_ptr + row_idx * stride_output_row + pair_offsets * 2
    odd_ptrs = even_ptrs + 1
    if IS_BF16:
        tl.store(even_ptrs, out_even.to(tl.bfloat16), mask=mask)
        tl.store(odd_ptrs, out_odd.to(tl.bfloat16), mask=mask)
    else:
        tl.store(even_ptrs, out_even.to(tl.float16), mask=mask)
        tl.store(odd_ptrs, out_odd.to(tl.float16), mask=mask)


def _validate_x(x: torch.Tensor) -> None:
    if x.ndim != 4:
        raise ValueError(f"expected x with shape [batch, seq, heads, head_dim], got {tuple(x.shape)}")
    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"supported input dtypes are {_SUPPORTED_DTYPES}, got {x.dtype}")
    if x.device.type != "cuda":
        raise ValueError("apply_rope requires CUDA tensors")
    if x.shape[-1] % 2 != 0:
        raise ValueError("head_dim must be even")
    if x.shape[-1] > _MAX_HEAD_DIM:
        raise ValueError(f"head_dim {x.shape[-1]} exceeds max supported size {_MAX_HEAD_DIM}")


def _normalize_positions(
    positions: torch.Tensor | None,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    max_seq_len: int,
) -> torch.Tensor:
    if positions is None:
        return torch.arange(seq_len, device=device, dtype=torch.int32).expand(batch_size, seq_len).contiguous()

    if positions.device != device:
        raise ValueError(f"device mismatch: {positions.device} != {device}")
    if positions.ndim == 1:
        if positions.shape[0] != seq_len:
            raise ValueError(f"expected positions shape ({seq_len},), got {tuple(positions.shape)}")
        normalized = positions.reshape(1, seq_len).expand(batch_size, seq_len)
    elif positions.ndim == 2:
        if tuple(positions.shape) != (batch_size, seq_len):
            raise ValueError(f"expected positions shape ({batch_size}, {seq_len}), got {tuple(positions.shape)}")
        normalized = positions
    else:
        raise ValueError(f"positions must be 1D or 2D, got {positions.ndim}D")

    if normalized.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"positions must use int32 or int64, got {normalized.dtype}")

    normalized_contiguous = normalized.contiguous()
    min_position = int(normalized_contiguous.min().item())
    max_position = int(normalized_contiguous.max().item())
    if min_position < 0:
        raise ValueError("positions must be non-negative")
    if max_position >= max_seq_len:
        raise ValueError(f"positions must be < {max_seq_len}, got {max_position}")
    return normalized_contiguous.to(dtype=torch.int32)


def _validate_freqs(
    cos: torch.Tensor,
    sin: torch.Tensor,
    head_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cos.shape != sin.shape:
        raise ValueError(f"cos/sin shape mismatch: {tuple(cos.shape)} != {tuple(sin.shape)}")
    if cos.ndim != 2:
        raise ValueError(f"expected cos/sin with shape [max_seq_len, head_dim // 2], got {tuple(cos.shape)}")
    if cos.shape[1] != head_dim // 2:
        raise ValueError(f"expected cos/sin second dimension {head_dim // 2}, got {cos.shape[1]}")
    if cos.device != device or sin.device != device:
        raise ValueError(f"cos/sin must be on {device}, got {cos.device} and {sin.device}")
    if not cos.dtype.is_floating_point or not sin.dtype.is_floating_point:
        raise TypeError("cos and sin must be floating-point tensors")
    return cos.contiguous(), sin.contiguous()


def _launch_config(head_dim: int) -> tuple[int, int]:
    block_size = triton.next_power_of_2(head_dim // 2)
    if block_size <= 32:
        num_warps = 2
    elif block_size <= 64:
        num_warps = 4
    else:
        num_warps = 8
    return block_size, num_warps


def _torch_apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    batch_size, seq_len, _, head_dim = x.shape
    normalized_positions = _normalize_positions(positions, batch_size, seq_len, x.device, cos.shape[0])
    cos_selected = cos[normalized_positions].unsqueeze(2).to(torch.float32)
    sin_selected = sin[normalized_positions].unsqueeze(2).to(torch.float32)

    x_even = x[..., 0::2].float()
    x_odd = x[..., 1::2].float()
    out_even = x_even * cos_selected - x_odd * sin_selected
    out_odd = x_odd * cos_selected + x_even * sin_selected

    output = torch.empty_like(x)
    output[..., 0::2] = out_even.to(x.dtype)
    output[..., 1::2] = out_odd.to(x.dtype)
    return output


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    _validate_x(x)
    _, seq_len, num_heads, head_dim = x.shape
    cos_contiguous, sin_contiguous = _validate_freqs(cos, sin, head_dim, x.device)
    normalized_positions = _normalize_positions(positions, x.shape[0], seq_len, x.device, cos_contiguous.shape[0])

    x_rows = x.contiguous().reshape(-1, head_dim)
    output_rows = torch.empty_like(x_rows)
    position_rows = normalized_positions.reshape(-1)
    block_size, num_warps = _launch_config(head_dim)

    grid = (x_rows.shape[0],)
    rope_kernel[grid](
        x_rows,
        cos_contiguous,
        sin_contiguous,
        position_rows,
        output_rows,
        x_rows.shape[0],
        num_heads,
        head_dim // 2,
        x_rows.stride(0),
        cos_contiguous.stride(0),
        sin_contiguous.stride(0),
        output_rows.stride(0),
        position_rows.stride(0),
        IS_BF16=x.dtype == torch.bfloat16,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return output_rows.reshape_as(x)


def _benchmark_ms(fn: Callable[[], torch.Tensor], warmup: int = 10, iters: int = 100) -> float:
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


def benchmark_rope(
    batch_size: int = 8,
    seq_len: int = 2048,
    num_heads: int = 32,
    head_dim: int = 128,
    dtype: torch.dtype = torch.float16,
    theta: float = 10000.0,
) -> float:
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark_rope requires a CUDA device")
    if dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"supported benchmark dtypes are {_SUPPORTED_DTYPES}, got {dtype}")

    device = torch.device("cuda")
    x = torch.randn((batch_size, seq_len, num_heads, head_dim), device=device, dtype=dtype)
    cos, sin = precompute_freqs(head_dim, seq_len, theta)
    cos = cos.to(device=device)
    sin = sin.to(device=device)

    triton_ms = _benchmark_ms(lambda: apply_rope(x, cos, sin))
    torch_ms = _benchmark_ms(lambda: _torch_apply_rope(x, cos, sin))
    speedup = torch_ms / triton_ms

    print(
        f"RoPE {batch_size}x{seq_len}x{num_heads}x{head_dim} [{dtype}]\n"
        f"  Triton:     {triton_ms:.3f} ms\n"
        f"  PyTorch:    {torch_ms:.3f} ms\n"
        f"  Speedup:    {speedup:.2f}x"
    )
    return speedup


if __name__ == "__main__":
    benchmark_rope()

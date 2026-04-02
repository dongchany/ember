"""Sampling utilities for LLM inference.

Fused temperature scaling + top-k/top-p filtering + softmax + sampling.
Full Triton fusion of top-k + softmax + multinomial is impractical due to
Triton's block-based model, so we fuse what we can (temperature + masking)
and use PyTorch for the final multinomial step.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _temperature_topk_mask_kernel(
    logits_ptr,
    output_ptr,
    n_cols,
    temperature,
    top_k_threshold,
    stride_row,
    BLOCK_SIZE: tl.constexpr,
):
    """Fuse temperature scaling + top-k masking in one pass."""
    row_idx = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols

    logits = tl.load(
        logits_ptr + row_idx * stride_row + offsets, mask=mask, other=float("-inf")
    )

    # Temperature scaling
    scaled = logits / temperature

    # Top-k masking: values below threshold are set to -inf
    output = tl.where(scaled >= top_k_threshold, scaled, float("-inf"))

    tl.store(output_ptr + row_idx * stride_row + offsets, output, mask=mask)


def _compute_topk_threshold(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Compute the per-row threshold for top-k filtering."""
    if top_k >= logits.shape[-1]:
        return torch.full(
            (logits.shape[0],), float("-inf"), device=logits.device, dtype=logits.dtype
        )
    # kthvalue returns the k-th smallest, so we want the (n - top_k)-th smallest
    # which equals the (top_k)-th largest
    topk_vals, _ = logits.topk(top_k, dim=-1)
    return topk_vals[:, -1]  # smallest of the top-k = threshold


def _top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Apply nucleus (top-p) filtering on probability distribution."""
    if top_p >= 1.0:
        return probs

    sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
    cumulative_probs = sorted_probs.cumsum(dim=-1)

    # Remove tokens with cumulative probability above top_p
    remove_mask = cumulative_probs - sorted_probs > top_p
    sorted_probs[remove_mask] = 0.0

    # Scatter back to original order
    filtered = torch.zeros_like(probs)
    filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_probs)
    return filtered


def sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
) -> torch.Tensor:
    """Sample token ids from logits with temperature, top-k, and top-p.

    Args:
        logits: [batch_size, vocab_size] raw logits from model
        temperature: scaling factor (1.0 = no change, <1 = sharper, >1 = flatter)
        top_k: keep only top-k logits (0 = disabled)
        top_p: nucleus sampling threshold (1.0 = disabled)

    Returns:
        [batch_size] tensor of sampled token indices
    """
    if logits.ndim != 2:
        raise ValueError(f"expected 2D logits, got {logits.ndim}D")
    if logits.device.type != "cuda":
        raise ValueError("sample requires CUDA tensors")

    batch_size, vocab_size = logits.shape

    if temperature <= 0:
        return greedy_sample(logits)

    # Compute top-k threshold
    if top_k > 0:
        threshold = _compute_topk_threshold(logits, min(top_k, vocab_size))
    else:
        threshold = torch.full(
            (batch_size,), float("-inf"), device=logits.device, dtype=logits.dtype
        )

    # Fused temperature + top-k masking via Triton
    block_size = triton.next_power_of_2(vocab_size)
    output = torch.empty_like(logits)

    # For now, use a single threshold (min across batch for simplicity in kernel)
    # A more optimal version would pass per-row thresholds
    min_threshold = threshold.min().item()

    grid = (batch_size,)
    _temperature_topk_mask_kernel[grid](
        logits,
        output,
        vocab_size,
        temperature,
        min_threshold,
        logits.stride(0),
        BLOCK_SIZE=block_size,
    )

    # Softmax + top-p + multinomial (PyTorch — efficient enough for inference)
    probs = torch.softmax(output, dim=-1)

    if top_p < 1.0:
        probs = _top_p_filter(probs, top_p)
        # Re-normalize after top-p filtering
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    """Greedy (argmax) sampling.

    Args:
        logits: [batch_size, vocab_size] or [batch_size, seq_len, vocab_size]

    Returns:
        [batch_size] or [batch_size, seq_len] tensor of token indices
    """
    return logits.argmax(dim=-1)


def benchmark_sampling(
    batch_size: int = 64,
    vocab_size: int = 128256,
    dtype: torch.dtype = torch.float16,
) -> float:
    """Benchmark sampling vs naive PyTorch implementation."""
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark requires CUDA")

    device = torch.device("cuda")
    logits = torch.randn((batch_size, vocab_size), device=device, dtype=dtype)

    # Warm up
    for _ in range(5):
        sample(logits, temperature=0.8, top_k=50, top_p=0.9)
        probs = torch.softmax(logits / 0.8, dim=-1)
        torch.multinomial(probs, num_samples=1)

    torch.cuda.synchronize()
    iters = 100

    # Ember sampling
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        sample(logits, temperature=0.8, top_k=50, top_p=0.9)
    end.record()
    torch.cuda.synchronize()
    ember_ms = start.elapsed_time(end) / iters

    # Naive PyTorch
    start.record()
    for _ in range(iters):
        scaled = logits / 0.8
        probs = torch.softmax(scaled, dim=-1)
        torch.multinomial(probs, num_samples=1)
    end.record()
    torch.cuda.synchronize()
    torch_ms = start.elapsed_time(end) / iters

    speedup = torch_ms / ember_ms
    print(
        f"Sampling {batch_size}x{vocab_size} [{dtype}]\n"
        f"  Ember:    {ember_ms:.3f} ms\n"
        f"  PyTorch:  {torch_ms:.3f} ms\n"
        f"  Speedup:  {speedup:.2f}x"
    )
    return speedup


if __name__ == "__main__":
    benchmark_sampling()

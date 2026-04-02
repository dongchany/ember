import pytest
import torch

from kernels.triton.fused_rope import apply_rope, precompute_freqs


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


_HEAD_DIMS = [64, 128, 256]


def _torch_apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    batch_size, seq_len, _, _ = x.shape
    if positions is None:
        normalized_positions = torch.arange(seq_len, device=x.device, dtype=torch.int64).expand(batch_size, seq_len)
    elif positions.ndim == 1:
        normalized_positions = positions.reshape(1, seq_len).expand(batch_size, seq_len)
    else:
        normalized_positions = positions

    cos_selected = cos[normalized_positions].unsqueeze(2).to(torch.float32)
    sin_selected = sin[normalized_positions].unsqueeze(2).to(torch.float32)
    x_even = x[..., 0::2].float()
    x_odd = x[..., 1::2].float()

    output = torch.empty_like(x)
    output[..., 0::2] = (x_even * cos_selected - x_odd * sin_selected).to(x.dtype)
    output[..., 1::2] = (x_odd * cos_selected + x_even * sin_selected).to(x.dtype)
    return output


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float16:
        return 1e-3, 1e-3
    return 1e-2, 1e-2


@pytest.mark.parametrize("head_dim", _HEAD_DIMS)
def test_apply_rope_fp16(head_dim: int) -> None:
    batch_size = 3
    seq_len = 17
    num_heads = 5
    x = torch.randn((batch_size, seq_len, num_heads, head_dim), device="cuda", dtype=torch.float16)
    cos, sin = precompute_freqs(head_dim, seq_len)
    cos = cos.to(device="cuda")
    sin = sin.to(device="cuda")

    actual = apply_rope(x, cos, sin)
    expected = _torch_apply_rope(x, cos, sin)

    atol, rtol = _tolerances(torch.float16)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("head_dim", _HEAD_DIMS)
def test_apply_rope_with_noncontiguous_positions(head_dim: int) -> None:
    batch_size = 2
    seq_len = 19
    num_heads = 4
    max_seq_len = seq_len * 2
    x = torch.randn((batch_size, seq_len, num_heads, head_dim), device="cuda", dtype=torch.float16)
    cos, sin = precompute_freqs(head_dim, max_seq_len)
    cos = cos.to(device="cuda")
    sin = sin.to(device="cuda")

    positions = torch.arange(max_seq_len, device="cuda", dtype=torch.int64).repeat(batch_size, 2)[:, ::2][:, :seq_len]
    assert not positions.is_contiguous()

    actual = apply_rope(x, cos, sin, positions=positions)
    expected = _torch_apply_rope(x, cos, sin, positions=positions)

    atol, rtol = _tolerances(torch.float16)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("head_dim", _HEAD_DIMS)
def test_apply_rope_bf16(head_dim: int) -> None:
    if not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 is not supported on this GPU")

    batch_size = 2
    seq_len = 13
    num_heads = 6
    x = torch.randn((batch_size, seq_len, num_heads, head_dim), device="cuda", dtype=torch.bfloat16)
    cos, sin = precompute_freqs(head_dim, seq_len)
    cos = cos.to(device="cuda")
    sin = sin.to(device="cuda")

    actual = apply_rope(x, cos, sin)
    expected = _torch_apply_rope(x, cos, sin)

    atol, rtol = _tolerances(torch.bfloat16)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

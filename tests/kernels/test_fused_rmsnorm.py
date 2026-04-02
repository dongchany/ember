import pytest
import torch

from kernels.triton.fused_rmsnorm import fused_rmsnorm_residual, rmsnorm


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


_HIDDEN_SIZES = [768, 2048, 4096, 8192]


def _torch_fused_rmsnorm_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    updated_residual = (x.float() + residual.float()).to(x.dtype)
    mean_square = updated_residual.float().pow(2).mean(dim=-1, keepdim=True)
    output = (updated_residual.float() * torch.rsqrt(mean_square + eps) * weight.float()).to(x.dtype)
    return output, updated_residual


def _torch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    mean_square = x.float().pow(2).mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(mean_square + eps) * weight.float()).to(x.dtype)


def _tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float16:
        return 1e-3, 1e-3
    return 1e-2, 1e-2


@pytest.mark.parametrize("hidden_size", _HIDDEN_SIZES)
def test_fused_rmsnorm_residual_fp16(hidden_size: int) -> None:
    eps = 1e-6
    batch_size = 7
    x = torch.randn((batch_size, hidden_size), device="cuda", dtype=torch.float16)
    residual = torch.randn_like(x)
    weight = torch.randn((hidden_size,), device="cuda", dtype=torch.float16)

    actual_output, actual_residual = fused_rmsnorm_residual(x, residual, weight, eps)
    expected_output, expected_residual = _torch_fused_rmsnorm_residual(x, residual, weight, eps)

    atol, rtol = _tolerances(torch.float16)
    torch.testing.assert_close(actual_output, expected_output, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual_residual, expected_residual, atol=atol, rtol=rtol)


@pytest.mark.parametrize("hidden_size", _HIDDEN_SIZES)
def test_rmsnorm_fp16(hidden_size: int) -> None:
    eps = 1e-6
    batch_size = 7
    x = torch.randn((batch_size, hidden_size), device="cuda", dtype=torch.float16)
    weight = torch.randn((hidden_size,), device="cuda", dtype=torch.float16)

    actual = rmsnorm(x, weight, eps)
    expected = _torch_rmsnorm(x, weight, eps)

    atol, rtol = _tolerances(torch.float16)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("hidden_size", _HIDDEN_SIZES)
def test_fused_rmsnorm_residual_bf16(hidden_size: int) -> None:
    if not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 is not supported on this GPU")

    eps = 1e-6
    batch_size = 7
    x = torch.randn((batch_size, hidden_size), device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn((hidden_size,), device="cuda", dtype=torch.bfloat16)

    actual_output, actual_residual = fused_rmsnorm_residual(x, residual, weight, eps)
    expected_output, expected_residual = _torch_fused_rmsnorm_residual(x, residual, weight, eps)

    atol, rtol = _tolerances(torch.bfloat16)
    torch.testing.assert_close(actual_output, expected_output, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual_residual, expected_residual, atol=atol, rtol=rtol)


@pytest.mark.parametrize("hidden_size", _HIDDEN_SIZES)
def test_rmsnorm_bf16(hidden_size: int) -> None:
    if not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 is not supported on this GPU")

    eps = 1e-6
    batch_size = 7
    x = torch.randn((batch_size, hidden_size), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((hidden_size,), device="cuda", dtype=torch.bfloat16)

    actual = rmsnorm(x, weight, eps)
    expected = _torch_rmsnorm(x, weight, eps)

    atol, rtol = _tolerances(torch.bfloat16)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

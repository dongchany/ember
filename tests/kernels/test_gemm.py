import pytest
import torch

from kernels.triton.gemm import matmul


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _assert_gemm_close(a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float) -> None:
    actual = matmul(a, b)
    expected = torch.matmul(a, b)
    assert actual.dtype == a.dtype
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def test_gemm_square() -> None:
    a = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
    b = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
    _assert_gemm_close(a, b, atol=1e-2, rtol=1e-2)


def test_gemm_rectangular() -> None:
    a = torch.randn((768, 1536), device="cuda", dtype=torch.float16)
    b = torch.randn((1536, 640), device="cuda", dtype=torch.float16)
    _assert_gemm_close(a, b, atol=1e-2, rtol=1e-2)


def test_gemm_bf16() -> None:
    if not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 is not supported on this GPU")

    a = torch.randn((768, 1024), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((1024, 512), device="cuda", dtype=torch.bfloat16)
    _assert_gemm_close(a, b, atol=1e-2, rtol=1e-2)


def test_gemm_small() -> None:
    a = torch.randn((128, 128), device="cuda", dtype=torch.float16)
    b = torch.randn((128, 128), device="cuda", dtype=torch.float16)
    _assert_gemm_close(a, b, atol=1e-2, rtol=1e-2)

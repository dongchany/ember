import pytest
import torch

from kernels.triton.sampling import greedy_sample, sample

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_greedy_sample() -> None:
    logits = torch.tensor([[1.0, 5.0, 2.0], [4.0, 1.0, 3.0]], device="cuda")
    result = greedy_sample(logits)
    expected = torch.tensor([1, 0], device="cuda")
    assert torch.equal(result, expected)


def test_sample_with_temperature() -> None:
    torch.manual_seed(42)
    # Very low temperature should behave like greedy
    logits = torch.tensor([[1.0, 100.0, 2.0]], device="cuda", dtype=torch.float16)
    results = [sample(logits, temperature=0.01).item() for _ in range(10)]
    assert all(r == 1 for r in results), f"Low temp should pick argmax, got {results}"


def test_sample_zero_temperature_is_greedy() -> None:
    logits = torch.tensor(
        [[1.0, 5.0, 2.0], [4.0, 1.0, 3.0]], device="cuda", dtype=torch.float16
    )
    result = sample(logits, temperature=0.0)
    expected = greedy_sample(logits)
    assert torch.equal(result, expected)


def test_sample_top_k() -> None:
    torch.manual_seed(0)
    # With top_k=1, should always pick the max
    logits = torch.randn((8, 1000), device="cuda", dtype=torch.float16)
    result = sample(logits, temperature=1.0, top_k=1, top_p=1.0)
    expected = greedy_sample(logits)
    assert torch.equal(result, expected)


def test_sample_top_p() -> None:
    torch.manual_seed(0)
    logits = torch.randn((4, 500), device="cuda", dtype=torch.float16)
    # Should not crash and should return valid indices
    result = sample(logits, temperature=1.0, top_k=0, top_p=0.5)
    assert result.shape == (4,)
    assert (result >= 0).all() and (result < 500).all()


def test_sample_batch() -> None:
    torch.manual_seed(0)
    batch_size = 32
    vocab_size = 32000
    logits = torch.randn((batch_size, vocab_size), device="cuda", dtype=torch.float16)
    result = sample(logits, temperature=0.8, top_k=50, top_p=0.9)
    assert result.shape == (batch_size,)
    assert result.dtype in (torch.int64, torch.int32)
    assert (result >= 0).all() and (result < vocab_size).all()

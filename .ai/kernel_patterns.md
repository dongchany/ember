# Ember Kernel Writing Patterns

## Language: Triton (primary), CUDA (extreme cases only)

## Kernel Registration

```python
@ember.kernel(
    kind="elementwise",          # elementwise | reduction | matmul | attention | custom
    fusable_epilog=True,         # Can the graph compiler fuse epilog ops into this kernel?
    memory_pattern="pointwise",  # pointwise | broadcast | gather | scatter
)
@triton.jit
def kernel_name(...):
    ...
```

## Autotune Pattern

```python
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 256, "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=8, num_stages=3),
        # ... more configs
    ],
    key=["M", "N", "K"],  # Re-tune when these change
)
@triton.jit
def gemm_kernel(...):
    ...
```

## File Naming
- `kernels/triton/<op_name>.py` - One file per operation
- Each file exports: kernel function + Python wrapper + benchmark function

## Testing
- Every kernel must have a reference implementation (usually PyTorch)
- Numerical tolerance: fp16 < 1e-3, bf16 < 1e-2, fp32 < 1e-5
- Performance test: must not regress vs baseline

## Key Optimization Techniques (from Modular's approach)
1. Shape specialization: use `tl.constexpr` for known dimensions
2. Software pipelining: overlap loads with compute via `num_stages`
3. Memory coalescing: ensure warp-contiguous memory access
4. Shared memory: use `tl.load` with eviction policies
5. Tensor Core: use `tl.dot` for matmul (maps to HMMA/WMMA)

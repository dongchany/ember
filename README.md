# Ember

An open-source, full-stack AI inference platform.

Ember covers the entire inference stack from GPU kernels to serving, built on
[Triton](https://github.com/triton-lang/triton),
[IREE](https://github.com/iree-org/iree), and
[SGLang](https://github.com/sgl-project/sglang).

## Architecture

```
  AI Evolution Layer   Auto-Evolve (AI-driven continuous optimization)
  ─────────────────────────────────────────────────────────────────
  Layer 8  Serve       SGLang-based, 3-process, continuous batching
  Layer 7  Pipeline    Text gen / speculative / constrained decoding
  Layer 6  KV Cache    Paged + multi-tier (GPU, CPU, SSD)
  Layer 5  NN Module   Transformer, Attention, RoPE, weight loading
  Layer 4  Graph       Static graph API + IREE compiler (custom passes)
  Layer 3  Runtime     IREE Runtime + LLM scheduling extensions
  Layer 2  Kernels     Triton kernels + FlashAttention + FlagGems
  Layer 1  Compiler    MLIR/LLVM + Triton compiler + CUTLASS
  ─────────────────────────────────────────────────────────────────
  Hardware             NVIDIA (PTX) | AMD (ROCm) | Apple (Metal)
```

## Status

**Early development.** Not yet functional.

## Build

```bash
bazel build //...
```

## License

Apache 2.0. See [LICENSE](LICENSE).

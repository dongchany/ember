# Ember Development Guide

## What is Ember?
Open-source full-stack AI inference platform. Covers GPU kernels through serving.
Built on Triton + IREE + SGLang. Targets NVIDIA and AMD GPUs.

## Build
```bash
bazel build //...
bazel test //...
```

## Project Structure
```
kernels/triton/      Triton GPU kernels (Python)
kernels/cuda/        Hand-written CUDA kernels (extreme cases)
compiler/passes/     Custom MLIR optimization passes (C++)
runtime/             IREE Runtime extensions (C++)
python/ember/        Python package
  graph/             Static graph API
  nn/                Neural network modules (Transformer, Attention, etc.)
  cache/             KV cache management
  pipeline/          Inference pipeline orchestration
  serve/             Inference server (SGLang-based)
  auto_evolve/       AI-driven optimization system
  _bindings/         nanobind C++/Python bridge
models/              Model adapters
tests/               Tests (mirrors python/ember/ structure)
benchmarks/          Performance benchmarks
third_party/         IREE and SGLang forks (git submodules)
.ai/                 Context files for AI agents
docs/tasks/          Task specs and roadmap
```

## Key Patterns
- Kernels: Triton with `@triton.autotune`. See `.ai/kernel_patterns.md`
- Graph API outputs StableHLO → IREE compiles → VMFB artifact
- NN modules build static graphs (not eager execution)
- Tests: numerical accuracy vs reference + performance benchmark

## Architecture Decisions
See `.ai/architecture_decisions.md` for rationale behind all major choices.

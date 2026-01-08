# Architecture Overview

This document explains the Ember inference engine at a high level and highlights
Qwen3-specific details that matter for correctness and performance.

## Dataflow (single step)

Input IDs
  -> Embedding lookup
  -> N decoder layers:
     - Input RMSNorm
     - Q/K/V projection
     - Q/K per-head RMSNorm (Qwen3-specific)
     - RoPE (rotary position embedding)
     - KV cache update
     - Attention (GQA)
     - O projection + residual
     - Post-attn RMSNorm
     - MLP (SwiGLU) + residual
  -> Final RMSNorm
  -> LM head -> logits
  -> Sampler -> next token

## Key modules

- `main.cpp`:
  CLI entry point, argument parsing, session lifecycle, sampling loop.
- `core/`:
  - `config.h`: model/runtime config, sampling params.
  - `session.h`: KV cache and token history.
  - `sampler.h`: temperature/top-k/top-p + penalties.
  - `tokenizer.h`: HuggingFace tokenizer loader (BPE).
- `runtime/iruntime.h`:
  Runtime abstraction and device mapping.
- `formats/safetensors.*`:
  Model weight loader (HuggingFace safetensors).
- `backends/cuda/`:
  CUDA runtime and kernels (RMSNorm, RoPE, attention, MLP).

## Qwen3-specific details

- Head dimension is explicit in config (not always hidden_size / num_heads).
- Attention uses GQA (num_kv_heads < num_heads).
- Q and K are normalized per head dimension (q_norm/k_norm).
- RoPE uses model-configured rope_theta.
- MLP uses SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x)).

## Where to look when debugging

- Output quality or logits drift:
  - `backends/cuda/cuda_runtime.cpp`: layer forward, q/k norm, MLP.
  - `backends/cuda/kernels/`: RMSNorm, RoPE, attention, softmax.
- Tokenization mismatch:
  - `core/tokenizer.h`, `formats/tokenizer.json` handling.
- Sampling/looping text:
  - `core/sampler.h` and CLI flags in `main.cpp`.

## Multi-GPU

Pipeline parallelism splits layers across devices. Hidden states are moved via
`cudaMemcpyPeer` between devices. See `runtime/iruntime.h` and
`backends/cuda/cuda_runtime.cpp`.

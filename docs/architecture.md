# Architecture Overview

This document describes Ember's current runtime architecture and the planned
architecture changes for Qwen3.5.

Status date: **March 8, 2026**.

## 1. Current architecture (Qwen3)

Ember currently runs Qwen3 dense decoder models with a Transformer-style layer
loop.

Single-step flow:

```text
Input IDs
  -> Embedding lookup
  -> N decoder layers:
     - Input RMSNorm
     - Q/K/V projection
     - Q/K per-head RMSNorm
     - RoPE
     - KV cache update
     - Attention (GQA)
     - O projection + residual
     - Post-attn RMSNorm
     - MLP (SwiGLU) + residual
  -> Final RMSNorm
  -> LM head -> logits
  -> Sampler -> next token
```

## 2. Key modules

- `apps/ember_cli/main.cpp`
  CLI entry point, argument parsing, session lifecycle, and token loop.
- `core/config.h`, `core/config_loader.cpp`
  Model/runtime config and load path.
- `core/session.h`
  Runtime state (token history + KV cache ownership/layout).
- `core/sampler.h`
  Sampling params and token selection logic.
- `runtime/iruntime.h`
  Backend abstraction and device mapping.
- `formats/safetensors.*`
  HuggingFace safetensors loading.
- `backends/cuda/cuda_runtime.cpp`
  CUDA forward orchestration and layer execution.
- `backends/cuda/kernels/`
  CUDA kernels for RMSNorm, RoPE, attention, softmax, and related ops.

## 3. Qwen3.5 architecture gap

Qwen3.5 introduces a hybrid stack and (for larger variants) MoE. This does not
fit a pure "all layers are Transformer attention + FFN" assumption.

Main gaps:

1. Layer type model:
   current code assumes one dominant layer type; Qwen3.5 needs per-layer type
   dispatch (`DeltaNet` or `GatedAttention`).
2. State model:
   current runtime centers on KV cache; DeltaNet layers need recurrent state in
   addition to (or instead of) KV cache.
3. Weight schema:
   current Qwen3 naming/layout is insufficient for Qwen3.5 hybrid blocks.
4. MoE path:
   routing/dispatch/combine is not yet in the hot path.

## 4. Planned Qwen3.5 runtime shape

Target per-step flow for hybrid dense models:

```text
Input IDs
  -> Embedding
  -> For each layer i:
     if layer[i] == DeltaNet:
       norm -> deltanet update (recurrent state) -> residual -> norm -> ffn -> residual
     else (GatedAttention):
       norm -> gated attention (KV path) -> residual -> norm -> ffn -> residual
  -> Final norm
  -> LM head -> logits -> sampler
```

Required structural changes:

- Extend session state with recurrent-state buffers for DeltaNet layers.
- Split layer forward into two explicit paths in CUDA runtime.
- Extend model config with hybrid layout metadata.
- Extend loader/key mapping for Qwen3.5 weight names.

## 5. Incremental implementation order

1. Hybrid dense first (0.8B/2B/4B/9B/27B):
   DeltaNet + GatedAttention + recurrent state + hybrid dispatch.
2. MoE next (35B-A3B and above):
   router, dispatch/combine, and expert scheduling.
3. Memory/offload optimization:
   FP8 load path, CPU offload for cold experts, prefetch scheduling.
4. Training/rollout loops:
   best-of-N and LoRA-oriented train/update paths.

Detailed milestone plan is maintained in
`docs/qwen35_upgrade_plan.md`.

## 6. Multi-GPU note

Current pipeline parallelism remains valid as the base strategy. Under Qwen3.5,
it must handle mixed layer types and mixed state (KV + recurrent) while keeping
cross-device transfer asynchronous.

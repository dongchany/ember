# Qwen3.5 Upgrade Plan

Status date: **March 8, 2026**
Owner: Ember core runtime

## 1. Objective

Become a high-performance **pure C++/CUDA** engine for Qwen3.5 on consumer
hardware, then extend from inference to an integrated training/improvement loop.

Primary target platform:
- 2x RTX 3080 Ti (12 GB each, PCIe 4.0 x8/x8, no NVLink)
- i9-11900K (AVX-512)
- 64 GB DDR4

## 2. Current baseline

Already in-tree:
- Qwen3 FP16/BF16 inference path
- CUDA runtime + kernels
- pipeline-parallel runtime skeleton
- tokenizer/safetensors loader
- CLI and server path
- regression scripts and correctness tooling

Not yet in-tree:
- Qwen3.5 Hybrid DeltaNet path
- Qwen3.5 MoE runtime path
- FP8 W8A16 load/dequant production path
- expert CPU offload scheduler
- native training loop

## 3. Why Qwen3.5

Strategic reasons:
1. One family spans small to very large scales (dense + MoE variants).
2. Same family can support both local and larger-system deployment paths.
3. Clear differentiation opportunity for a native C++/CUDA runtime.

## 4. Hardware constraints that drive design

1. No FP8 Tensor Core on RTX 3080 Ti:
   FP8 is storage/bandwidth optimization only; compute path remains W8A16.
2. No NVLink:
   inter-GPU transfer is over PCIe and must be overlapped aggressively.
3. Total VRAM is 24 GB:
   larger MoE variants require expert caching and CPU offload.
4. CPU bandwidth is limited relative to GPU:
   CPU expert compute is fallback, not the main execution path.

## 5. Architecture delta (Qwen3 -> Qwen3.5)

Required additions:

1. Hybrid layer dispatch:
   `DeltaNet` and `GatedAttention` need different forward paths.
2. Hybrid state model:
   KV cache + recurrent state must coexist in `Session`.
3. Loader/config extension:
   add hybrid layout metadata and Qwen3.5 weight key mapping.
4. MoE runtime path:
   router, token dispatch/combine, shared expert, expert scheduling.

## 6. Phase roadmap

### Phase 0: Dense hybrid inference first

Scope:
- Qwen3.5 dense variants first (0.8B/2B/4B/9B/27B)
- no MoE in this phase

Deliverables:
1. Gated DeltaNet forward CUDA path.
2. Gated Attention path update (including output gate behavior).
3. Recurrent state management in `Session`.
4. Hybrid layer layout parsing + per-layer dispatch.
5. End-to-end greedy alignment against HF reference.

Exit criteria:
1. `Qwen3.5-0.8B` greedy decode matches HF reference for fixed prompt suite.
2. `Qwen3.5-4B` runs stably on single 3080 Ti in BF16.
3. `Qwen3.5-9B` runs stably in dual-GPU pipeline BF16.

### Phase 1: MoE runtime support

Scope:
- MoE execution path for 35B-A3B class models and above.

Deliverables:
1. Top-K router kernel.
2. token dispatch/combine kernels.
3. shared expert path.
4. expert scheduling interface with GPU cache + host backing store.

Exit criteria:
1. 35B-A3B inference path functional end-to-end with offload enabled.
2. Deterministic decode stability under repeated runs.

### Phase 2: FP8 W8A16 + heterogeneous offload

Scope:
- operational FP8 loading path and practical large-model deployment on
  2x3080Ti + host memory.

Deliverables:
1. FP8 safetensors load path (E4M3/E5M2 handling as required by model files).
2. W8A16 dequant+GEMM path.
3. dual-GPU layer partition + asynchronous transfer scheduling.
4. expert cache policy and asynchronous host prefetch.

Exit criteria:
1. 9B FP8 and BF16 both pass correctness gate.
2. 27B/35B class path runs with documented offload policy.
3. profile output shows bounded offload stall ratio under target load.

### Phase 3: performance optimization

Scope:
- close or exceed mainstream serving framework baselines on same hardware.

Deliverables:
1. kernel fusion where profitable.
2. decode launch overhead reduction (including graph capture where safe).
3. pipeline bubble reduction and scheduler tuning.
4. optional speculative decode path after base stability.

Exit criteria:
1. measurable TTFT/TPOT improvements over Phase 2 baseline.
2. no regression in quality metrics on locked eval set.

### Phase 4: native training/improvement loop

Scope:
- start with high-ROI features before full generic training.

Priority order:
1. Best-of-N generation + verifier loop.
2. LoRA SFT update loop.
3. DPO/GRPO-style advanced loop after stability.

Exit criteria:
1. reproducible quality gain on selected task suite.
2. full inference->evaluate->update loop in one runtime stack.

## 7. Benchmark and validation policy (mandatory)

Each major phase must publish:

1. Regression gate:
   build + CPU tests + CUDA smoke + HF alignment checks.
2. Quality gate:
   lm-eval tasks (minimum: `mmlu`, `gsm8k`, `hellaswag`, `arc_challenge`,
   `truthfulqa`).
3. Serving gate:
   TTFT/TPOT/ITL + throughput under fixed trace configuration.
4. Standardized report:
   include mean/median/P99 with exact commit SHA and runtime flags.

## 8. Milestone summary

- M1: Qwen3.5 dense hybrid path correct and stable.
- M2: MoE runtime path with expert dispatch/caching available.
- M3: FP8+offload path stable on target 2x3080Ti platform.
- M4: performance tuning reaches target TTFT/TPOT envelope.
- M5: native best-of-N and LoRA fine-tuning loop operational.

## 9. Non-goals (for near-term focus)

Not in immediate scope until core path is stable:
- broad multi-model support outside Qwen family
- full multimodal stack
- full pretraining-scale optimizer stack

## 10. Tracking and change control

When a scope change is made, update this file with:
1. exact date
2. reason for change
3. impacted milestone(s)
4. revised acceptance criteria

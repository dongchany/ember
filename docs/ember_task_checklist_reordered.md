# Ember Task Checklist (Reordered by Dependency and Impact)

This checklist reorders the original `Ember_Full_TaskChecklist_v2(1).md` into a strict execution order.
No time assumptions are encoded.

## 0. Hard Gates (must be true before large runs)

- [x] `build/ember_stage_breakdown` and `build/ember_benchmark` pass smoke run on `0` and `0,1`.
- [x] Main 4B model snapshot is fixed and recorded (Qwen3-4B-Instruct-2507).
- [x] GGUF path for cross-framework baseline is fixed and recorded.
- [x] Output directory convention is fixed (`reports/<experiment_name>_<date>`).

## 1. P1 Core Evidence First (Motivation + Main Claim)

### 1.1 Rollout Time Decomposition (P1 Fig 2)
- [x] Add/verify prefill/decode/sampling/transfer breakdown export.
- [x] Run context x gen-length matrix on 4B.
- [x] Export `prefill_share` curves and raw tables.
- [x] Decision rule: if prefill share is low, shift narrative to sync/memory + temporal cache semantics.

### 1.2 Prefix Reuse Gain (P1 Sec 4.4)
- [ ] Build shared-prefix workload (long common instruction/schema prefix).
- [ ] Measure with and without prefix cache.
- [ ] Export savings vs prefix length.

### 1.3 Multi-Round Cumulative Cost (P1 Fig 3)
- [ ] Simulate 10-50 rounds policy updates.
- [ ] Compare cumulative GPU time for `Naive`, `Prefix-only`, `UpdateLocality`.
- [ ] Export per-round and cumulative curves.

## 2. P1 Enabling Engine Features (minimum viable cache-policy system)

### 2.1 Multi-candidate Rollout + Logprobs
- [ ] `generate(prompts, num_candidates, sampling_params)` supports N=4/8/16.
- [ ] Export token-level logprobs.
- [ ] Numerical consistency checks against reference implementation.

### 2.2 Prefix Cache + LoRA Hot Update
- [ ] Prefix KV cache lifecycle (create/hit/invalidate).
- [ ] LoRA adapter load/inject/hot-swap without base reload.
- [ ] Export hot-update latency and correctness checks.

### 2.3 Cache Policy Interface (first usable set)
- [ ] `CachePolicy` abstraction.
- [ ] Implement `Naive`.
- [ ] Implement `UpdateLocality(N)`.
- [ ] Implement `PeriodicRefresh(k)`.
- [ ] Policy stats export (hit/miss/recompute/error proxy).

## 3. P1 Key Ablations (decide whether P5 stays independent)

### 3.1 Strategy Family Main Table
- [ ] Compare strategy variants on extraction workload.
- [ ] Report GPU-hours, quality, memory, prefill savings.

### 3.2 Update Locality Sweep (critical)
- [ ] Sweep `N` (e.g. 25/50/75/full refresh boundary).
- [ ] Report speedup vs quality tradeoff.
- [ ] Produce recommended `N` range and failure boundary.

### 3.3 UpdatableKV Sweep (promote or demote)
- [ ] Sweep LoRA rank `r` and refresh interval `k`.
- [ ] Measure correction error and end quality.
- [ ] Gate: keep as P5 only if multi-layer correction stays stable.

## 4. Training Loop Integration (prove end-to-end utility)

### 4.1 Baseline ladders
- [ ] SFT baseline.
- [ ] Best-of-N baseline.
- [ ] DPO loop (main).
- [ ] GRPO loop (secondary comparison).

### 4.2 Unified backend advantage
- [ ] Measure dual-stack vs unified memory footprint.
- [ ] Measure weight-sync overhead vs in-process hot update.
- [ ] Report rollout+update end-to-end throughput.

## 5. Cross-Framework Comparison (required for external credibility)

### 5.1 Mandatory comparisons
- [x] Ember vs llama.cpp (single + dual GPU, same prompt/gen setup).
- [ ] Ember vs PyTorch generate baseline (if available).
- [x] Report harness caveats and non-apples-to-apples parts explicitly.

### 5.2 Optional when environment allows
- [ ] Ember vs vLLM rollout baseline.
- [ ] Ember vs SGLang rollout baseline.
- [x] Mark skipped items with environment reason if unavailable.

## 6. Secondary Papers / Deferred Work (only after P1 data lock)

- [ ] P2 engine paper polishing from stage profiling + loading comparisons.
- [ ] P3 training-kernel replacement and dual-stack elimination deep dive.
- [ ] P4 extraction recipe full reward ablations.
- [ ] P5 full theorem/tightness only if 3.3 passes gate.

## 7. Done Criteria for "P1-ready project state"

- [ ] P1 figures/tables all mapped to concrete report files.
- [ ] At least one complete run of strategy comparison with reproducible commands.
- [x] Cross-framework table includes real measurements plus explicit skipped reasons.
- [ ] All `XX%` placeholders removed from talk/paper source notes.

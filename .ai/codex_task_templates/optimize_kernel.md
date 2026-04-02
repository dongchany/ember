# Task: Optimize Kernel
# Project: Ember (AI inference platform)
# Working Directory: ~/workspace/ember

## Goal
[One sentence: what to optimize and why]

## Current Kernel
- File: kernels/triton/<name>.py
- Profile data: [attach or describe bottleneck]

## Optimization Target
- Metric: [TFLOPS / bandwidth utilization / latency]
- Baseline: [current number]
- Goal: [target number or percentage improvement]

## Constraints
- Must maintain numerical accuracy (vs reference, tolerance < 1e-3 for fp16)
- Must pass all existing tests
- Must work on both NVIDIA and AMD (Triton multi-backend)

## Deliverables
- [ ] Optimized kernel variant
- [ ] Benchmark comparison (before/after)
- [ ] Updated autotune configs if applicable
- [ ] Test verifying numerical accuracy

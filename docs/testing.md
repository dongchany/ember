# Testing and Regression

Status date: **March 8, 2026**.

This guide defines the default validation ladder for current Qwen3 support and
adds the benchmark methodology required for the Qwen3.5 upgrade cycle.

## 1. Fast regression ladder (current default)

Quick script cookbook:
`scripts/ci/README.md`

Recommended local entrypoint:
`scripts/ci/run_local.sh` (profiles: `quick/full/full-lite/perf`)

## 1.1 Build-only check (always run)

```bash
cmake --build build --parallel
```

CI helper:

```bash
scripts/ci/build.sh
```

## 1.2 CPU tests (fast, no GPU)

```bash
./build/ember_tests
```

## 1.3 CTest (optional)

```bash
ctest --test-dir build
```

GPU-only labels:

```bash
ctest --test-dir build -L gpu
```

## 1.4 CUDA smoke tests

Kernel smoke (no model required):

```bash
./build/ember_cuda_kernels_smoke
```

Runtime smoke (requires `MODEL_PATH`):

```bash
MODEL_PATH="/path/to/models--Qwen--Qwen3-0.6B" \
./build/ember_cuda_runtime_smoke
```

## 1.5 Correctness alignment checks

Dump outputs:

```bash
./build/ember -m /path/to/model --check --dump-layer 2 -p "Hello, my name is"
```

Compare logits:

```bash
python3 scripts/compare_logits.py \
  --model /path/to/model \
  --debug-dir debug/check_models--Qwen--Qwen3-0_6B
```

Compare hidden states:

```bash
python3 scripts/compare_hidden.py \
  --model /path/to/model \
  --debug-dir debug/check_models--Qwen--Qwen3-0_6B \
  --layer 2
```

## 1.6 Unified CI script options

`scripts/ci/dev_check.sh`, `gpu_check.sh`, `layer_check.sh`, and
`greedy_regression.sh` share these options:

```text
--build-dir <dir>
--model-path <dir>
--model-paths <dir1,dir2>   # gpu_check/dev_check
--model-id <id>             # e.g. Qwen3-8B or Qwen/Qwen3-8B
--hub-root <dir>            # e.g. ~/xilinx/huggingface/hub
--gpus <ids>                # e.g. 1 or 0,1
--python-runner <auto|uv|direct>
--python-bin <path>
--require-hf-compare / --no-require-hf-compare
```

Example:

```bash
scripts/ci/dev_check.sh --full \
  --hub-root ~/xilinx/huggingface/hub \
  --model-id Qwen3-8B \
  --gpus 0,1 \
  --python-runner uv \
  --python-bin /home/dong/workspace/ember/torch-env/bin/python
```

## 2. Standard benchmark stack (for Qwen3.5 milestones)

For upgrade milestones, report both quality and serving performance with
reproducible methods.

## 2.1 Layer A: model quality (lm-eval)

Goal: verify architectural/quantization changes do not break quality.

Recommended tasks:
- `mmlu`
- `gsm8k`
- `hellaswag`
- `arc_challenge`
- `truthfulqa`

Run (via OpenAI-compatible endpoint):

```bash
lm_eval --model local-completions \
  --model_args model=qwen3.5-9b,base_url=http://localhost:8080/v1 \
  --tasks mmlu,gsm8k,hellaswag,arc_challenge,truthfulqa \
  --batch_size 8 \
  --output_path results/lm_eval_qwen35_9b/
```

## 2.2 Layer B: serving performance (trace replay)

Goal: evaluate user-facing latency and throughput with realistic request
patterns.

Core metrics:
- TTFT (time to first token)
- TPOT (time per output token)
- ITL (inter-token latency)
- output throughput (tok/s)

Use one benchmark runner consistently across engines
(Ember / llama.cpp / SGLang / KTransformers) with the same:
- hardware
- model
- prompt/output length constraints
- request arrival pattern

## 2.3 Layer C: MLPerf-compatible reporting

Goal: publish a standardized view of performance across scenarios.

Minimum recommended internal scenarios:
- Offline throughput
- Single-request latency (single-stream style)
- Interactive percentile report (TTFT/TPOT)

Report at least:
- mean
- median
- P99

for TTFT/TPOT and total throughput.

## 3. Required report bundle for each major milestone

For each milestone branch/tag, publish one report directory containing:

1. commit SHA and build flags
2. hardware/software environment
3. regression ladder pass/fail outputs
4. lm-eval task scores
5. serving latency/throughput metrics (mean/median/P99)
6. comparison table vs previous milestone baseline

## 4. CI notes

- `.github/workflows/ci.yml` builds in a CUDA container (no GPU required).
- `.github/workflows/gpu-check.yml` expects a self-hosted GPU runner.
- `scripts/ci/gpu_check.sh` threshold envs:
  - `LOGITS_MAX_ABS_THRESHOLD`
  - `LOGITS_MEAN_ABS_THRESHOLD`
  - `HIDDEN_MAX_ABS_THRESHOLD`
  - `HIDDEN_MEAN_ABS_THRESHOLD`

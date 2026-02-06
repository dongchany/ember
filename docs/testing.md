# Testing and Regression

This project uses a layered testing approach.

## 1. Build-only check (always run)

```
cmake --build build -j
```

CI helper:
```
scripts/ci/build.sh
```

## 2. CPU unit tests (fast, no GPU)

```
./build/ember_tests
```

## 3. CTest (optional)

Run all registered tests (useful once the build directory exists):
```
ctest --test-dir build
```

Run only GPU-labeled tests:
```
ctest --test-dir build -L gpu
```

Pass a model path for tests that need it:
```
MODEL_PATH="/path/to/models--Qwen--Qwen3-0.6B" \
ctest --test-dir build -L gpu
```

## 4. CUDA kernel smoke tests (GPU, no model required)

```
./build/ember_cuda_kernels_smoke
```

## 4. CUDA runtime smoke test (GPU, requires MODEL_PATH)

```
MODEL_PATH="/path/to/models--Qwen--Qwen3-0.6B" \
./build/ember_cuda_runtime_smoke
```

## 4. Correctness check (GPU)

Dump outputs:
```
./build/ember -m /path/to/model --check --dump-layer 2 -p "Hello, my name is"
```

Compare logits:
```
python3 scripts/compare_logits.py \
  --model /path/to/model \
  --debug-dir debug/check_models--Qwen--Qwen3-0_6B
```

Compare hidden states:
```
python3 scripts/compare_hidden.py \
  --model /path/to/model \
  --debug-dir debug/check_models--Qwen--Qwen3-0_6B \
  --layer 2
```

## Recommended workflow (local GPU)

Run these in order. Stop on the first failure and inspect the printed diffs.

Quick dev loop (build + CPU tests + kernel smoke):
```
scripts/ci/dev_check.sh
```

Add a model path to include CUDA runtime smoke:
```
MODEL_PATH="/path/to/models--Qwen--Qwen3-0.6B" \
scripts/ci/dev_check.sh
```

Enable deeper checks when needed:
```
MODEL_PATH="/path/to/models--Qwen--Qwen3-0.6B" \
RUN_LAYER_CHECK=1 \
RUN_GPU_CHECK=1 \
scripts/ci/dev_check.sh
```

1) Build (fast, always run):
```
scripts/ci/build.sh
```

2) CPU unit tests (fast, no GPU):
```
./build/ember_tests
```

3) CTest (optional):
```
ctest --test-dir build
```

4) CUDA kernel smoke tests (GPU, no model required):
```
./build/ember_cuda_kernels_smoke
```

5) CUDA runtime smoke test (GPU, requires MODEL_PATH):
```
MODEL_PATH="/path/to/models--Qwen--Qwen3-0.6B" \
./build/ember_cuda_runtime_smoke
```

6) Quick GPU correctness with the smaller model (baseline):
```
MODEL_PATH="/path/to/models--Qwen--Qwen3-0.6B" \
LOGITS_MAX_ABS_THRESHOLD=4 \
LOGITS_MEAN_ABS_THRESHOLD=1 \
HIDDEN_MAX_ABS_THRESHOLD=1 \
HIDDEN_MEAN_ABS_THRESHOLD=0.2 \
HIDDEN_LAYER=2 \
scripts/ci/gpu_check.sh
```

7) Optional: run the larger model as a deeper regression:
```
MODEL_PATH="/path/to/models--Qwen--Qwen3-4B-Instruct-2507" \
LOGITS_MAX_ABS_THRESHOLD=4 \
LOGITS_MEAN_ABS_THRESHOLD=1 \
HIDDEN_MAX_ABS_THRESHOLD=1 \
HIDDEN_MEAN_ABS_THRESHOLD=0.2 \
HIDDEN_LAYER=2 \
scripts/ci/gpu_check.sh
```

8) Optional: run both models in one command:
```
MODEL_PATHS="/path/to/models--Qwen--Qwen3-0.6B,/path/to/models--Qwen--Qwen3-4B-Instruct-2507" \
LOGITS_MAX_ABS_THRESHOLD=4 \
LOGITS_MEAN_ABS_THRESHOLD=1 \
HIDDEN_MAX_ABS_THRESHOLD=1 \
HIDDEN_MEAN_ABS_THRESHOLD=0.2 \
HIDDEN_LAYER=2 \
scripts/ci/gpu_check.sh
```

If the logits check fails, fix that first. If logits passes but hidden states fail,
use `compare_hidden.py` output to pick a layer and then run `scripts/ci/layer_check.sh`
to get per-layer intermediate diffs.

## 7. Single-layer integration check (debugging)

Use this when a GPU correctness check fails and you want to localize the issue to
a specific layer and its sub-ops (attn/MLP intermediates).

Notes:
- "dump" means saving intermediate tensors (typically the last token vector) to files.
- "dump layer" selects which layer to dump. Use `-1` to dump all layers (slower, more files).

```
MODEL_PATH="/path/to/models--Qwen--Qwen3-0.6B" \
LAYER=2 \
HIDDEN_MAX_ABS_THRESHOLD=1 \
HIDDEN_MEAN_ABS_THRESHOLD=0.2 \
scripts/ci/layer_check.sh
```

## 8. Sampling sanity (GPU)

Greedy (deterministic):
```
./build/ember -m /path/to/model -p "Hello, my name is" --temp 0 --top-k 1 --top-p 1
```

Typical sampling:
```
./build/ember -m /path/to/model -p "Hello, my name is" \
  --temp 0.7 --top-p 0.9 --top-k 40 \
  --repeat-penalty 1.1 --presence-penalty 0.2 --frequency-penalty 0.2 \
  --no-repeat-ngram 3
```

## Expected thresholds

These are guidance values for Qwen3-0.6B:
- `compare_logits.py`: max_abs_diff < 4.0, mean_abs_diff < 1.0
- `compare_hidden.py` (layer 2): max_abs_diff ~1.0, mean_abs_diff ~0.2

Use these as guardrails, not strict guarantees.

## CI notes

- `.github/workflows/ci.yml` builds in a CUDA container (no GPU required).
- `.github/workflows/gpu-check.yml` is optional and expects a self-hosted GPU runner.
- `scripts/ci/gpu_check.sh` can enforce thresholds via:
  - `LOGITS_MAX_ABS_THRESHOLD`
  - `LOGITS_MEAN_ABS_THRESHOLD`
  - `HIDDEN_MAX_ABS_THRESHOLD`
  - `HIDDEN_MEAN_ABS_THRESHOLD`
- `gpu_check.sh` expects `MODEL_PATH` to be set on the runner (a local HF model directory).
- You can also set `MODEL_PATHS` (comma-separated) to run multiple models.
- `RUN_HIDDEN_COMPARE=0` disables `compare_hidden.py`.
- `HIDDEN_LAYER` controls which layer is dumped/compared (default: 2).
- GPU check uses a model-specific dump dir: `debug/check_<model_basename>`.

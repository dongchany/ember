# CI Scripts Quickstart

This folder contains local-check scripts for fast iteration.

## Which script to run

- `run_local.sh`: Recommended local entrypoint with profiles (`quick/full/full-lite/perf`).
- `dev_check.sh`: One entrypoint for daily checks (build/test/smoke/correctness/regression).
- `gpu_check.sh`: Check-mode dump + optional HF logits/hidden compare.
- `layer_check.sh`: Narrow down drift to a specific layer.
- `greedy_regression.sh`: Token-level greedy regression with baseline files.

## 30-second start

```bash
scripts/ci/run_local.sh
```

Runs:
- Build
- `ember_tests`
- `ember_cuda_kernels_smoke`

Success criteria:
- exit code `0`
- no `[FAIL]` lines

## Full check (HF cache model id style)

```bash
scripts/ci/run_local.sh full \
  --hub-root ~/xilinx/huggingface/hub \
  --gpus 0,1 \
  --model-b Qwen3-8B
```

## Full check on limited VRAM (practical preset)

```bash
scripts/ci/run_local.sh full-lite \
  --hub-root ~/xilinx/huggingface/hub \
  --gpus 0,1 \
  --model-b Qwen3-8B \
  --no-require-hf-compare
```

## If you only want generated text

`gpu_check/layer_check` are check-mode by design (dump tensors, not text generation).

Use CLI directly:

```bash
./build/ember -m /path/to/model --devices 0,1 -p "Hello, my name is" -n 128
```

## Tips

- `--model-b` and `--model` are accepted aliases in CI scripts.
- `--gpus 1` uses single GPU 1. `--gpus 0,1` uses two GPUs.
- `scripts/ci/run_local.sh --help` shows profile semantics and examples.
- `--help` on each script includes examples and expected outcomes.

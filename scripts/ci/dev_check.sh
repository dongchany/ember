#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

BUILD_DIR="${BUILD_DIR:-build}"
MODEL_PATH="${MODEL_PATH:-}"
MODEL_PATHS="${MODEL_PATHS:-}"

MODE="quick"
for arg in "$@"; do
  case "${arg}" in
    --quick) MODE="quick" ;;
    --full) MODE="full" ;;
    --perf) MODE="perf" ;;
    -h|--help)
      cat <<'EOF'
Usage: scripts/ci/dev_check.sh [--quick|--full|--perf]

Modes:
  --quick  Build + CPU tests + minimal GPU smoke (default)
  --full   Run additional GPU correctness checks (requires MODEL_PATH)
  --perf   Run lightweight performance-oriented benchmarks (requires GPU; MODEL_PATH optional)

Environment variables:
  BUILD_DIR=build            build directory
  MODEL_PATH=/path/to/model  model directory for runtime checks
  MODEL_PATHS=a,b,c          comma-separated list for gpu_check.sh

Toggles:
  RUN_KERNEL_SMOKE=1
  RUN_RUNTIME_SMOKE=1
  RUN_VARPOS_SMOKE=0
  RUN_DUAL_GPU_SMOKE=0
  RUN_LAYER_CHECK=0
  RUN_GPU_CHECK=0
EOF
      exit 0
      ;;
    *)
      echo "[dev-check] unknown argument: ${arg}" >&2
      exit 1
      ;;
  esac
done

RUN_KERNEL_SMOKE="${RUN_KERNEL_SMOKE:-1}"
RUN_RUNTIME_SMOKE="${RUN_RUNTIME_SMOKE:-1}"

if [[ "${MODE}" == "full" ]]; then
  RUN_VARPOS_SMOKE="${RUN_VARPOS_SMOKE:-1}"
  RUN_DUAL_GPU_SMOKE="${RUN_DUAL_GPU_SMOKE:-1}"
  RUN_LAYER_CHECK="${RUN_LAYER_CHECK:-1}"
  RUN_GPU_CHECK="${RUN_GPU_CHECK:-1}"
else
  RUN_VARPOS_SMOKE="${RUN_VARPOS_SMOKE:-0}"
  RUN_DUAL_GPU_SMOKE="${RUN_DUAL_GPU_SMOKE:-0}"
  RUN_LAYER_CHECK="${RUN_LAYER_CHECK:-0}"
  RUN_GPU_CHECK="${RUN_GPU_CHECK:-0}"
fi

echo "[dev-check] Build"
scripts/ci/build.sh

echo "[dev-check] CPU tests"
"${BUILD_DIR}/ember_tests"

if [[ "${RUN_KERNEL_SMOKE}" != "0" ]]; then
  echo "[dev-check] CUDA kernel smoke"
  "${BUILD_DIR}/ember_cuda_kernels_smoke"
fi

if [[ -n "${MODEL_PATH}" && "${RUN_RUNTIME_SMOKE}" != "0" ]]; then
  echo "[dev-check] CUDA runtime smoke"
  MODEL_PATH="${MODEL_PATH}" "${BUILD_DIR}/ember_cuda_runtime_smoke"
fi

if [[ -n "${MODEL_PATH}" && "${RUN_VARPOS_SMOKE}" != "0" ]]; then
  echo "[dev-check] Varpos batch smoke"
  MODEL_PATH="${MODEL_PATH}" "${BUILD_DIR}/ember_varpos_batch_smoke"
fi

if [[ -n "${MODEL_PATH}" && "${RUN_DUAL_GPU_SMOKE}" != "0" ]]; then
  echo "[dev-check] Dual GPU smoke"
  MODEL_PATH="${MODEL_PATH}" "${BUILD_DIR}/ember_dual_gpu_smoke"
fi

if [[ "${RUN_LAYER_CHECK}" != "0" ]]; then
  echo "[dev-check] Single-layer check"
  MODEL_PATH="${MODEL_PATH}" scripts/ci/layer_check.sh
fi

if [[ "${RUN_GPU_CHECK}" != "0" ]]; then
  echo "[dev-check] GPU correctness check"
  MODEL_PATH="${MODEL_PATH}" MODEL_PATHS="${MODEL_PATHS}" scripts/ci/gpu_check.sh
fi

if [[ "${MODE}" == "perf" ]]; then
  if [[ -x "${BUILD_DIR}/ember_kernel_bench" ]]; then
    echo "[dev-check] Kernel microbench"
    "${BUILD_DIR}/ember_kernel_bench" --dtype f16
  fi
  if [[ -n "${MODEL_PATH}" && -x "${BUILD_DIR}/ember_stage_breakdown" ]]; then
    echo "[dev-check] Stage breakdown (quick)"
    "${BUILD_DIR}/ember_stage_breakdown" --model "${MODEL_PATH}" --gpus 0 --prompt-len 128 --decode-steps 64 --iters 1 --warmup 0
  fi
fi

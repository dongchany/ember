#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

BUILD_DIR="${BUILD_DIR:-build}"
MODEL_PATH="${MODEL_PATH:-}"
MODEL_PATHS="${MODEL_PATHS:-}"

RUN_KERNEL_SMOKE="${RUN_KERNEL_SMOKE:-1}"
RUN_RUNTIME_SMOKE="${RUN_RUNTIME_SMOKE:-1}"
RUN_LAYER_CHECK="${RUN_LAYER_CHECK:-0}"
RUN_GPU_CHECK="${RUN_GPU_CHECK:-0}"

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

if [[ "${RUN_LAYER_CHECK}" != "0" ]]; then
  echo "[dev-check] Single-layer check"
  MODEL_PATH="${MODEL_PATH}" scripts/ci/layer_check.sh
fi

if [[ "${RUN_GPU_CHECK}" != "0" ]]; then
  echo "[dev-check] GPU correctness check"
  MODEL_PATH="${MODEL_PATH}" MODEL_PATHS="${MODEL_PATHS}" scripts/ci/gpu_check.sh
fi

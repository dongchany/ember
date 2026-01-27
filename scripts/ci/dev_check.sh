#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

BUILD_DIR="${BUILD_DIR:-build}"

RUN_KERNEL_SMOKE="${RUN_KERNEL_SMOKE:-1}"

echo "[dev-check] Build"
scripts/ci/build.sh

echo "[dev-check] CPU tests"
"${BUILD_DIR}/ember_tests"

if [[ "${RUN_KERNEL_SMOKE}" != "0" ]]; then
    echo "[dev-check] CUDA kernel smoke"
    "${BUILD_DIR}/ember_cuda_kernels_smoke"
fi

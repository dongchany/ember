#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

BUILD_DIR="${BUILD_DIR:-build}"
CUDA_ARCH="${CUDA_ARCH:-86}"
CMAKE_FORCE_CONFIGURE="${CMAKE_FORCE_CONFIGURE:-0}"
CMAKE_CACHE="${BUILD_DIR}/CMakeCache.txt"

if [[ -t 1 ]]; then
  COLOR_YELLOW=$'\033[1;33m'
  COLOR_GREEN=$'\033[0;32m'
  COLOR_RESET=$'\033[0m'
else
  COLOR_YELLOW=""
  COLOR_GREEN=""
  COLOR_RESET=""
fi

log_notice() {
  printf '%b\n' "${COLOR_YELLOW}==> $*${COLOR_RESET}"
}

log_ok() {
  printf '%b\n' "${COLOR_GREEN}==> $*${COLOR_RESET}"
}

needs_configure=0
if [[ "${CMAKE_FORCE_CONFIGURE}" == "1" ]]; then
  needs_configure=1
elif [[ ! -f "${CMAKE_CACHE}" ]]; then
  needs_configure=1
else
  if find . -path "./${BUILD_DIR}" -prune -o \( -name 'CMakeLists.txt' -o -name '*.cmake' \) -newer "${CMAKE_CACHE}" -print -quit | grep -q .; then
    needs_configure=1
  fi
fi

if [[ "${needs_configure}" == "1" ]]; then
  log_ok "Running CMake configure..."
  cmake -S . -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"
else
  log_notice "CMake configure skipped (cache is up-to-date)."
fi
cmake --build "${BUILD_DIR}" -j

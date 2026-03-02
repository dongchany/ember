#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/common.sh"
cd "${REPO_ROOT}"

PROFILE="quick"
HAS_MODEL_ARG=0

print_usage() {
  cat <<'EOF'
Usage:
  scripts/ci/run_local.sh [quick|full|full-lite|perf] [dev_check options...]

Profiles:
  quick      Build + CPU tests + CUDA kernel smoke
  full       dev_check --full
  full-lite  dev_check --full with practical local defaults:
             RUN_RUNTIME_SMOKE=0
             RUN_VARPOS_SMOKE=0
             RUN_DUAL_GPU_SMOKE=0
             RUN_GREEDY_REGRESSION=0
             --no-require-hf-compare (can be overridden by --require-hf-compare)
  perf       dev_check --perf

Examples:
  scripts/ci/run_local.sh
  scripts/ci/run_local.sh full --hub-root ~/xilinx/huggingface/hub --gpus 0,1 --model-b Qwen3-8B
  scripts/ci/run_local.sh full-lite --hub-root ~/xilinx/huggingface/hub --gpus 0,1 --model-b Qwen3-8B
  scripts/ci/run_local.sh full-lite --hub-root ~/xilinx/huggingface/hub --gpus 1 --model-b Qwen3-8B --python-bin ./torch-env/bin/python --python-runner direct
EOF
}

forward_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    quick|full|full-lite|perf)
      PROFILE="$1"
      shift
      ;;
    --quick)
      PROFILE="quick"
      shift
      ;;
    --full)
      PROFILE="full"
      shift
      ;;
    --full-lite|--lite)
      PROFILE="full-lite"
      shift
      ;;
    --perf)
      PROFILE="perf"
      shift
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    --)
      shift
      forward_args+=("$@")
      break
      ;;
    --model|--model-b|--model-id|--model-path|--model-paths)
      HAS_MODEL_ARG=1
      forward_args+=("$1")
      shift
      if [[ $# -gt 0 ]]; then
        forward_args+=("$1")
        shift
      fi
      ;;
    --model=*|--model-b=*|--model-id=*|--model-path=*|--model-paths=*)
      HAS_MODEL_ARG=1
      forward_args+=("$1")
      shift
      ;;
    *)
      forward_args+=("$1")
      shift
      ;;
  esac
done

mode_arg="--quick"
extra_args=()

case "${PROFILE}" in
  quick)
    mode_arg="--quick"
    ;;
  full)
    mode_arg="--full"
    ;;
  full-lite)
    mode_arg="--full"
    : "${RUN_RUNTIME_SMOKE:=0}"
    : "${RUN_VARPOS_SMOKE:=0}"
    : "${RUN_DUAL_GPU_SMOKE:=0}"
    : "${RUN_GREEDY_REGRESSION:=0}"
    export RUN_RUNTIME_SMOKE RUN_VARPOS_SMOKE RUN_DUAL_GPU_SMOKE RUN_GREEDY_REGRESSION
    extra_args+=(--no-require-hf-compare)
    ;;
  perf)
    mode_arg="--perf"
    ;;
  *)
    ci_die "run-local" "unknown profile: ${PROFILE}"
    ;;
esac

ci_log "run-local" "profile=${PROFILE}"
if [[ "${PROFILE}" == "full-lite" ]]; then
  ci_log "run-local" "full-lite toggles: RUN_RUNTIME_SMOKE=${RUN_RUNTIME_SMOKE} RUN_VARPOS_SMOKE=${RUN_VARPOS_SMOKE} RUN_DUAL_GPU_SMOKE=${RUN_DUAL_GPU_SMOKE} RUN_GREEDY_REGRESSION=${RUN_GREEDY_REGRESSION}"
fi

# Quick mode should run without requiring any model. If caller did not explicitly
# pass a model argument, ignore inherited MODEL_PATH-style env vars to avoid
# failing on stale values (e.g. hub root instead of snapshot dir).
if [[ "${PROFILE}" == "quick" && "${HAS_MODEL_ARG}" == "0" ]]; then
  if [[ -n "${MODEL_PATH:-}${MODEL_PATHS:-}${MODEL_ID:-}" ]]; then
    ci_log "run-local" "quick profile: ignoring inherited MODEL_PATH/MODEL_PATHS/MODEL_ID env vars"
  fi
  unset MODEL_PATH MODEL_PATHS MODEL_ID HUB_ROOT
  unset CI_MODEL_PATH CI_MODEL_PATHS CI_MODEL_ID CI_HUB_ROOT
fi

scripts/ci/dev_check.sh "${mode_arg}" "${extra_args[@]}" "${forward_args[@]}"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/common.sh"
cd "${REPO_ROOT}"

ci_init_common_env

MODE="quick"

print_usage() {
  cat <<'EOF'
Usage: scripts/ci/dev_check.sh [--quick|--full|--perf] [common-options]

Tip:
  Prefer scripts/ci/run_local.sh for profile-based local usage.

What this script does:
  1) Build project
  2) Run CPU tests + CUDA kernel smoke
  3) Optionally run deeper GPU correctness/regression checks

Modes:
  --quick  Build + CPU tests + minimal GPU smoke (default)
  --full   Run additional GPU correctness checks (requires model path)
  --perf   Run lightweight perf-oriented benchmarks

Common options:
  --build-dir DIR
  --model-path DIR
  --model-paths DIR1,DIR2   (for gpu_check)
  --model-id ID
  --model ID_OR_DIR         (alias; same as --model-b)
  --model-b ID_OR_DIR       (alias; id or local dir)
  --hub-root DIR
  --gpus IDS                (example: 1 or 0,1)
  --python-runner MODE      (auto|uv|direct)
  --python-bin PATH
  --require-hf-compare
  --no-require-hf-compare

Toggles (env, legacy):
  RUN_KERNEL_SMOKE=1
  RUN_RUNTIME_SMOKE=1
  RUN_VARPOS_SMOKE=0
  RUN_DUAL_GPU_SMOKE=0
  RUN_LAYER_CHECK=0
  RUN_GPU_CHECK=0
  RUN_GREEDY_REGRESSION=0

Common local recipes:
  # Quick local sanity (no model required)
  scripts/ci/dev_check.sh

  # Full check with HF cache model id
  scripts/ci/dev_check.sh --full --hub-root ~/xilinx/huggingface/hub --gpus 0,1 --model-b Qwen3-8B

  # Full check but skip known-heavy steps (good for 2x11GB cards)
  RUN_RUNTIME_SMOKE=0 RUN_VARPOS_SMOKE=0 RUN_DUAL_GPU_SMOKE=0 RUN_GREEDY_REGRESSION=0 \
  scripts/ci/dev_check.sh --full --hub-root ~/xilinx/huggingface/hub --gpus 0,1 --model-b Qwen3-8B --no-require-hf-compare

Expected result:
  - Exit code 0 and no [FAIL] lines
  - check-mode dumps under debug/layer_check_* and debug/check_*
  - If transformers is not installed and --no-require-hf-compare is used,
    compare steps are skipped by design
EOF
}

rest=()
ci_parse_common_args rest "$@"

while [[ ${#rest[@]} -gt 0 ]]; do
  case "${rest[0]}" in
    --quick)
      MODE="quick"
      rest=("${rest[@]:1}")
      ;;
    --full)
      MODE="full"
      rest=("${rest[@]:1}")
      ;;
    --perf)
      MODE="perf"
      rest=("${rest[@]:1}")
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      ci_die "dev-check" "unknown argument: ${rest[0]}"
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
  RUN_GREEDY_REGRESSION="${RUN_GREEDY_REGRESSION:-1}"
else
  RUN_VARPOS_SMOKE="${RUN_VARPOS_SMOKE:-0}"
  RUN_DUAL_GPU_SMOKE="${RUN_DUAL_GPU_SMOKE:-0}"
  RUN_LAYER_CHECK="${RUN_LAYER_CHECK:-0}"
  RUN_GPU_CHECK="${RUN_GPU_CHECK:-0}"
  RUN_GREEDY_REGRESSION="${RUN_GREEDY_REGRESSION:-0}"
fi

ci_resolve_build_dir "build"

# Resolve model-path from --model-id/--hub-root if needed.
ci_resolve_model_path_if_needed "dev-check" || true

# If only model-paths is provided, use the first one for single-model smoke/regression steps.
if [[ -z "${CI_MODEL_PATH:-}" && -n "${CI_MODEL_PATHS:-}" ]]; then
  IFS=',' read -r first_model _ <<< "${CI_MODEL_PATHS}"
  first_model="$(echo "${first_model}" | xargs)"
  if [[ -n "${first_model}" ]]; then
    CI_MODEL_PATH="$(ci_resolve_path_or_snapshot "${first_model}" || true)"
  fi
fi

common_args=(
  --build-dir "${CI_BUILD_DIR}"
  --gpus "${CI_GPUS}"
  --python-runner "${CI_PYTHON_RUNNER}"
)
if [[ -n "${CI_PYTHON_BIN:-}" ]]; then
  common_args+=(--python-bin "${CI_PYTHON_BIN}")
fi
if [[ "${CI_REQUIRE_HF_COMPARE}" != "0" ]]; then
  common_args+=(--require-hf-compare)
else
  common_args+=(--no-require-hf-compare)
fi
if [[ -n "${CI_HUB_ROOT:-}" ]]; then
  common_args+=(--hub-root "${CI_HUB_ROOT}")
fi
if [[ -n "${CI_MODEL_ID:-}" ]]; then
  common_args+=(--model-id "${CI_MODEL_ID}")
fi

ci_log "dev-check" "Build"
BUILD_DIR="${CI_BUILD_DIR}" scripts/ci/build.sh

ci_log "dev-check" "CPU tests"
"${CI_BUILD_DIR}/ember_tests"

if [[ "${RUN_KERNEL_SMOKE}" != "0" ]]; then
  ci_log "dev-check" "CUDA kernel smoke"
  "${CI_BUILD_DIR}/ember_cuda_kernels_smoke"
fi

SMOKE_DEVICE="$(ci_first_gpu "${CI_GPUS}")"

if [[ -n "${CI_MODEL_PATH:-}" && "${RUN_RUNTIME_SMOKE}" != "0" ]]; then
  ci_log "dev-check" "CUDA runtime smoke (device ${SMOKE_DEVICE})"
  MODEL_PATH="${CI_MODEL_PATH}" EMBER_DEVICE="${SMOKE_DEVICE}" "${CI_BUILD_DIR}/ember_cuda_runtime_smoke"
fi

if [[ -n "${CI_MODEL_PATH:-}" && "${RUN_VARPOS_SMOKE}" != "0" ]]; then
  ci_log "dev-check" "Varpos batch smoke (device ${SMOKE_DEVICE})"
  MODEL_PATH="${CI_MODEL_PATH}" EMBER_DEVICE="${SMOKE_DEVICE}" "${CI_BUILD_DIR}/ember_varpos_batch_smoke"
fi

if [[ -n "${CI_MODEL_PATH:-}" && "${RUN_DUAL_GPU_SMOKE}" != "0" ]]; then
  ci_log "dev-check" "Dual GPU smoke"
  MODEL_PATH="${CI_MODEL_PATH}" "${CI_BUILD_DIR}/ember_dual_gpu_smoke"
fi

if [[ "${RUN_LAYER_CHECK}" != "0" ]]; then
  ci_log "dev-check" "Single-layer check"
  layer_args=("${common_args[@]}")
  if [[ -n "${CI_MODEL_PATH:-}" ]]; then
    layer_args+=(--model-path "${CI_MODEL_PATH}")
  fi
  scripts/ci/layer_check.sh "${layer_args[@]}"
fi

if [[ "${RUN_GPU_CHECK}" != "0" ]]; then
  ci_log "dev-check" "GPU correctness check"
  gpu_args=("${common_args[@]}")
  if [[ -n "${CI_MODEL_PATHS:-}" ]]; then
    gpu_args+=(--model-paths "${CI_MODEL_PATHS}")
  elif [[ -n "${CI_MODEL_PATH:-}" ]]; then
    gpu_args+=(--model-path "${CI_MODEL_PATH}")
  fi
  scripts/ci/gpu_check.sh "${gpu_args[@]}"
fi

if [[ "${RUN_GREEDY_REGRESSION}" != "0" ]]; then
  ci_log "dev-check" "Greedy decode regression"
  greedy_args=("${common_args[@]}")
  if [[ -n "${CI_MODEL_PATH:-}" ]]; then
    greedy_args+=(--model-path "${CI_MODEL_PATH}")
  fi
  scripts/ci/greedy_regression.sh "${greedy_args[@]}"
fi

if [[ "${MODE}" == "perf" ]]; then
  if [[ -x "${CI_BUILD_DIR}/ember_kernel_bench" ]]; then
    ci_log "dev-check" "Kernel microbench"
    "${CI_BUILD_DIR}/ember_kernel_bench" --dtype f16
  fi
  if [[ -n "${CI_MODEL_PATH:-}" && -x "${CI_BUILD_DIR}/ember_stage_breakdown" ]]; then
    ci_log "dev-check" "Stage breakdown (quick)"
    "${CI_BUILD_DIR}/ember_stage_breakdown" \
      --model "${CI_MODEL_PATH}" \
      --gpus "${CI_GPUS}" \
      --prompt-len 128 \
      --decode-steps 64 \
      --iters 1 \
      --warmup 0
  fi
fi

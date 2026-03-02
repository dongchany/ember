#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/common.sh"
cd "${REPO_ROOT}"

ci_init_common_env

PROMPTS_FILE="${GREEDY_PROMPTS_FILE:-}"
BASELINE_PATH="${GREEDY_BASELINE_PATH:-}"
WRITE_BASELINE_PATH="${GREEDY_WRITE_BASELINE_PATH:-}"
MAX_NEW_TOKENS="${GREEDY_MAX_NEW_TOKENS:-16}"
MAX_CTX_LEN="${GREEDY_MAX_CTX_LEN:-512}"
EMBER_DEVICE="${EMBER_DEVICE:-}"

print_usage() {
  cat <<'EOF'
Usage: scripts/ci/greedy_regression.sh [common-options] [options]

What this script does:
  - Runs deterministic greedy decode checks on fixed prompts
  - Supports write/read baseline files for regression checking

Common options:
  --build-dir DIR
  --model-path DIR
  --model-id ID
  --model ID_OR_DIR         (alias; same as --model-b)
  --model-b ID_OR_DIR       (alias; id or local dir)
  --hub-root DIR
  --gpus IDS             (example: 1 or 0,1; first GPU is used)
  --python-runner MODE   (auto|uv|direct)
  --python-bin PATH

Greedy options:
  --prompts-file FILE
  --baseline FILE
  --write-baseline FILE
  --max-new-tokens N
  --max-ctx-len N
  --ember-device ID      (overrides --gpus first GPU)

Environment fallback (legacy):
  MODEL_PATH
  EMBER_DEVICE / EMBER_DEVICES
  GREEDY_PROMPTS_FILE / GREEDY_BASELINE_PATH / GREEDY_WRITE_BASELINE_PATH
  GREEDY_MAX_NEW_TOKENS / GREEDY_MAX_CTX_LEN

Examples:
  scripts/ci/greedy_regression.sh --hub-root ~/xilinx/huggingface/hub --model-b Qwen3-8B --gpus 1 --write-baseline debug/greedy_baseline_qwen3_8b.json
  scripts/ci/greedy_regression.sh --hub-root ~/xilinx/huggingface/hub --model-b Qwen3-8B --gpus 1 --baseline debug/greedy_baseline_qwen3_8b.json
EOF
}

rest=()
ci_parse_common_args rest "$@"

while [[ ${#rest[@]} -gt 0 ]]; do
  case "${rest[0]}" in
    --prompts-file)
      ci_expect_value "greedy-regression" "${rest[0]}" "${rest[1]:-}"
      PROMPTS_FILE="${rest[1]}"
      rest=("${rest[@]:2}")
      ;;
    --baseline)
      ci_expect_value "greedy-regression" "${rest[0]}" "${rest[1]:-}"
      BASELINE_PATH="${rest[1]}"
      rest=("${rest[@]:2}")
      ;;
    --write-baseline)
      ci_expect_value "greedy-regression" "${rest[0]}" "${rest[1]:-}"
      WRITE_BASELINE_PATH="${rest[1]}"
      rest=("${rest[@]:2}")
      ;;
    --max-new-tokens)
      ci_expect_value "greedy-regression" "${rest[0]}" "${rest[1]:-}"
      MAX_NEW_TOKENS="${rest[1]}"
      rest=("${rest[@]:2}")
      ;;
    --max-ctx-len)
      ci_expect_value "greedy-regression" "${rest[0]}" "${rest[1]:-}"
      MAX_CTX_LEN="${rest[1]}"
      rest=("${rest[@]:2}")
      ;;
    --ember-device)
      ci_expect_value "greedy-regression" "${rest[0]}" "${rest[1]:-}"
      EMBER_DEVICE="${rest[1]}"
      rest=("${rest[@]:2}")
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      ci_die "greedy-regression" "unknown argument: ${rest[0]}"
      ;;
  esac
done

if ! ci_resolve_model_path_if_needed "greedy-regression"; then
  ci_log "greedy-regression" "MODEL_PATH is not set; skipping."
  exit 0
fi

ci_resolve_build_dir "build"

if [[ -z "${EMBER_DEVICE}" ]]; then
  EMBER_DEVICE="$(ci_first_gpu "${CI_GPUS}")"
fi

if ! ci_resolve_python_runtime; then
  ci_die "greedy-regression" "Python runtime not found"
fi

args=(
  --model "${CI_MODEL_PATH}"
  --build-dir "${CI_BUILD_DIR}"
  --ember-device "${EMBER_DEVICE}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --max-ctx-len "${MAX_CTX_LEN}"
)

if [[ -n "${PROMPTS_FILE}" ]]; then
  args+=(--prompts-file "${PROMPTS_FILE}")
fi
if [[ -n "${BASELINE_PATH}" ]]; then
  args+=(--baseline "${BASELINE_PATH}")
fi
if [[ -n "${WRITE_BASELINE_PATH}" ]]; then
  args+=(--write-baseline "${WRITE_BASELINE_PATH}")
fi

ci_log "greedy-regression" "Python runner: ${CI_PYTHON_RUNNER} (${CI_PYTHON_BIN})"
ci_log "greedy-regression" "Running scripts/compare_greedy.py"
ci_run_python scripts/compare_greedy.py "${args[@]}"

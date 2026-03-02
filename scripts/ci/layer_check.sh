#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/common.sh"
cd "${REPO_ROOT}"

ci_init_common_env

LAYER="${LAYER:-2}"
PROMPT="${PROMPT:-Hello, my name is}"
DUMP_DIR="${DUMP_DIR:-}"

print_usage() {
  cat <<'EOF'
Usage: scripts/ci/layer_check.sh [common-options] [options]

What this script does:
  - Runs single-layer check-mode dump for debugging where drift starts
  - Optionally compares hidden states with HuggingFace (transformers required)

Common options:
  --build-dir DIR
  --model-path DIR
  --model-id ID
  --model ID_OR_DIR         (alias; same as --model-b)
  --model-b ID_OR_DIR       (alias; id or local dir)
  --hub-root DIR
  --gpus IDS             (example: 1 or 0,1)
  --python-runner MODE   (auto|uv|direct)
  --python-bin PATH
  --require-hf-compare
  --no-require-hf-compare

Layer check options:
  --layer N
  --prompt TEXT
  --dump-dir DIR

Environment fallback (legacy):
  MODEL_PATH
  EMBER_DEVICES
  PYTHON_RUNNER / PYTHON_BIN
  REQUIRE_HF_COMPARE

Example:
  scripts/ci/layer_check.sh --hub-root ~/xilinx/huggingface/hub --model-b Qwen3-8B --gpus 0,1 --layer 2 --no-require-hf-compare
EOF
}

rest=()
ci_parse_common_args rest "$@"

while [[ ${#rest[@]} -gt 0 ]]; do
  case "${rest[0]}" in
    --layer)
      ci_expect_value "layer-check" "${rest[0]}" "${rest[1]:-}"
      LAYER="${rest[1]}"
      rest=("${rest[@]:2}")
      ;;
    --prompt)
      ci_expect_value "layer-check" "${rest[0]}" "${rest[1]:-}"
      PROMPT="${rest[1]}"
      rest=("${rest[@]:2}")
      ;;
    --dump-dir)
      ci_expect_value "layer-check" "${rest[0]}" "${rest[1]:-}"
      DUMP_DIR="${rest[1]}"
      rest=("${rest[@]:2}")
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      ci_die "layer-check" "unknown argument: ${rest[0]}"
      ;;
  esac
done

if ! ci_resolve_model_path_if_needed "layer-check"; then
  ci_log "layer-check" "MODEL_PATH is not set; skipping."
  exit 0
fi

ci_resolve_build_dir "build"
BIN="${CI_BUILD_DIR}/ember"
[[ -x "${BIN}" ]] || ci_die "layer-check" "Missing binary: ${BIN}"

if [[ -z "${DUMP_DIR}" ]]; then
  model_base="$(basename "${CI_MODEL_PATH}")"
  safe_model="${model_base//[^a-zA-Z0-9._-]/_}"
  DUMP_DIR="debug/layer_check_${safe_model}_layer_${LAYER}"
fi

ci_log "layer-check" "Running ember --check (layer ${LAYER})"
"${BIN}" \
  -m "${CI_MODEL_PATH}" \
  --devices "${CI_GPUS}" \
  --check \
  --dump-layer "${LAYER}" \
  --dump-dir "${DUMP_DIR}" \
  -p "${PROMPT}"

if ci_resolve_python_runtime; then
  if ci_python_has_module "transformers"; then
    ci_log "layer-check" "Python runner: ${CI_PYTHON_RUNNER} (${CI_PYTHON_BIN})"
    ci_log "layer-check" "Running compare_hidden.py (layer ${LAYER})"
    args=(--model "${CI_MODEL_PATH}" --debug-dir "${DUMP_DIR}" --layer "${LAYER}")
    if [[ -n "${HIDDEN_MAX_ABS_THRESHOLD:-}" ]]; then
      args+=(--max-abs-threshold "${HIDDEN_MAX_ABS_THRESHOLD}")
    fi
    if [[ -n "${HIDDEN_MEAN_ABS_THRESHOLD:-}" ]]; then
      args+=(--mean-abs-threshold "${HIDDEN_MEAN_ABS_THRESHOLD}")
    fi
    ci_run_python scripts/compare_hidden.py "${args[@]}"
  else
    if [[ "${CI_REQUIRE_HF_COMPARE}" != "0" ]]; then
      ci_die "layer-check" "transformers not available and REQUIRE_HF_COMPARE=1; failing."
    fi
    ci_log "layer-check" "transformers not available; skipping compare_hidden.py"
  fi
else
  if [[ "${CI_REQUIRE_HF_COMPARE}" != "0" ]]; then
    ci_die "layer-check" "Python runtime not found and REQUIRE_HF_COMPARE=1; failing."
  fi
  ci_log "layer-check" "Python runtime not found; skipping compare_hidden.py"
fi

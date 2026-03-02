#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/common.sh"
cd "${REPO_ROOT}"

ci_init_common_env

RUN_HIDDEN_COMPARE="${RUN_HIDDEN_COMPARE:-1}"
HIDDEN_LAYER="${HIDDEN_LAYER:-2}"
PROMPT="${PROMPT:-Hello, my name is}"
LOGITS_MAX_ABS_THRESHOLD="${LOGITS_MAX_ABS_THRESHOLD:-}"
LOGITS_MEAN_ABS_THRESHOLD="${LOGITS_MEAN_ABS_THRESHOLD:-}"
HIDDEN_MAX_ABS_THRESHOLD="${HIDDEN_MAX_ABS_THRESHOLD:-}"
HIDDEN_MEAN_ABS_THRESHOLD="${HIDDEN_MEAN_ABS_THRESHOLD:-}"

print_usage() {
  cat <<'EOF'
Usage: scripts/ci/gpu_check.sh [common-options] [options]

What this script does:
  - Runs ember --check and dumps logits/hidden for one or more models
  - Optionally compares against HuggingFace outputs (transformers required)

Common options:
  --build-dir DIR
  --model-path DIR
  --model-paths DIR1,DIR2
  --model-id ID
  --model ID_OR_DIR         (alias; same as --model-b)
  --model-b ID_OR_DIR       (alias; id or local dir)
  --hub-root DIR
  --gpus IDS             (example: 1 or 0,1)
  --python-runner MODE   (auto|uv|direct)
  --python-bin PATH
  --require-hf-compare
  --no-require-hf-compare

GPU check options:
  --hidden-layer N
  --prompt TEXT
  --run-hidden-compare
  --no-hidden-compare
  --logits-max-abs-threshold F
  --logits-mean-abs-threshold F
  --hidden-max-abs-threshold F
  --hidden-mean-abs-threshold F

Environment fallback (legacy):
  MODEL_PATH / MODEL_PATHS
  EMBER_DEVICES
  PYTHON_RUNNER / PYTHON_BIN
  REQUIRE_HF_COMPARE

Example:
  scripts/ci/gpu_check.sh --hub-root ~/xilinx/huggingface/hub --model-b Qwen3-8B --gpus 0,1 --no-require-hf-compare
EOF
}

rest=()
ci_parse_common_args rest "$@"

while [[ ${#rest[@]} -gt 0 ]]; do
  case "${rest[0]}" in
    --hidden-layer)
      ci_expect_value "gpu-check" "${rest[0]}" "${rest[1]:-}"
      HIDDEN_LAYER="${rest[1]}"
      rest=("${rest[@]:2}")
      ;;
    --prompt)
      ci_expect_value "gpu-check" "${rest[0]}" "${rest[1]:-}"
      PROMPT="${rest[1]}"
      rest=("${rest[@]:2}")
      ;;
    --run-hidden-compare)
      RUN_HIDDEN_COMPARE=1
      rest=("${rest[@]:1}")
      ;;
    --no-hidden-compare)
      RUN_HIDDEN_COMPARE=0
      rest=("${rest[@]:1}")
      ;;
    --logits-max-abs-threshold)
      ci_expect_value "gpu-check" "${rest[0]}" "${rest[1]:-}"
      LOGITS_MAX_ABS_THRESHOLD="${rest[1]}"
      rest=("${rest[@]:2}")
      ;;
    --logits-mean-abs-threshold)
      ci_expect_value "gpu-check" "${rest[0]}" "${rest[1]:-}"
      LOGITS_MEAN_ABS_THRESHOLD="${rest[1]}"
      rest=("${rest[@]:2}")
      ;;
    --hidden-max-abs-threshold)
      ci_expect_value "gpu-check" "${rest[0]}" "${rest[1]:-}"
      HIDDEN_MAX_ABS_THRESHOLD="${rest[1]}"
      rest=("${rest[@]:2}")
      ;;
    --hidden-mean-abs-threshold)
      ci_expect_value "gpu-check" "${rest[0]}" "${rest[1]:-}"
      HIDDEN_MEAN_ABS_THRESHOLD="${rest[1]}"
      rest=("${rest[@]:2}")
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      ci_die "gpu-check" "unknown argument: ${rest[0]}"
      ;;
  esac
done

ci_resolve_build_dir "build"
BIN="${CI_BUILD_DIR}/ember"
[[ -x "${BIN}" ]] || ci_die "gpu-check" "Missing binary: ${BIN}"

models=()
if [[ -n "${CI_MODEL_PATHS:-}" ]]; then
  IFS=',' read -r -a models <<< "${CI_MODEL_PATHS}"
else
  if ci_resolve_model_path_if_needed "gpu-check"; then
    :
  fi
  if [[ -n "${CI_MODEL_PATH:-}" ]]; then
    models=("${CI_MODEL_PATH}")
  fi
fi

if [[ ${#models[@]} -eq 0 ]]; then
  ci_log "gpu-check" "MODEL_PATH(S) not set; skipping."
  exit 0
fi

for raw_model in "${models[@]}"; do
  model_candidate="$(echo "${raw_model}" | xargs)"
  [[ -n "${model_candidate}" ]] || continue

  if ! MODEL_PATH="$(ci_resolve_path_or_snapshot "${model_candidate}")"; then
    ci_die "gpu-check" "MODEL_PATH is invalid or not a local model dir: ${model_candidate}"
  fi

  model_base="$(basename "${MODEL_PATH}")"
  safe_model="${model_base//[^a-zA-Z0-9._-]/_}"
  dump_dir="debug/check_${safe_model}"

  ci_log "gpu-check" "Running ember --check (${model_base})"
  "${BIN}" \
    -m "${MODEL_PATH}" \
    --devices "${CI_GPUS}" \
    --check \
    --dump-layer "${HIDDEN_LAYER}" \
    --dump-dir "${dump_dir}" \
    -p "${PROMPT}"

  if ci_resolve_python_runtime; then
    if ci_python_has_module "transformers"; then
      ci_log "gpu-check" "Python runner: ${CI_PYTHON_RUNNER} (${CI_PYTHON_BIN})"
      ci_log "gpu-check" "Running compare_logits.py (${model_base})"
      args=(--model "${MODEL_PATH}" --debug-dir "${dump_dir}")
      if [[ -n "${LOGITS_MAX_ABS_THRESHOLD}" ]]; then
        args+=(--max-abs-threshold "${LOGITS_MAX_ABS_THRESHOLD}")
      fi
      if [[ -n "${LOGITS_MEAN_ABS_THRESHOLD}" ]]; then
        args+=(--mean-abs-threshold "${LOGITS_MEAN_ABS_THRESHOLD}")
      fi
      ci_run_python scripts/compare_logits.py "${args[@]}"

      if [[ "${RUN_HIDDEN_COMPARE}" != "0" ]]; then
        ci_log "gpu-check" "Running compare_hidden.py (${model_base})"
        args=(--model "${MODEL_PATH}" --debug-dir "${dump_dir}" --layer "${HIDDEN_LAYER}")
        if [[ -n "${HIDDEN_MAX_ABS_THRESHOLD}" ]]; then
          args+=(--max-abs-threshold "${HIDDEN_MAX_ABS_THRESHOLD}")
        fi
        if [[ -n "${HIDDEN_MEAN_ABS_THRESHOLD}" ]]; then
          args+=(--mean-abs-threshold "${HIDDEN_MEAN_ABS_THRESHOLD}")
        fi
        ci_run_python scripts/compare_hidden.py "${args[@]}"
      else
        ci_log "gpu-check" "RUN_HIDDEN_COMPARE=0; skipping compare_hidden.py"
      fi
    else
      if [[ "${CI_REQUIRE_HF_COMPARE}" != "0" ]]; then
        ci_die "gpu-check" "transformers not available and REQUIRE_HF_COMPARE=1; failing."
      fi
      ci_log "gpu-check" "transformers not available; skipping compare logits/hidden."
    fi
  else
    if [[ "${CI_REQUIRE_HF_COMPARE}" != "0" ]]; then
      ci_die "gpu-check" "Python runtime not found and REQUIRE_HF_COMPARE=1; failing."
    fi
    ci_log "gpu-check" "Python runtime not found; skipping compare logits/hidden."
  fi
done

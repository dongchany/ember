#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

BUILD_DIR="${BUILD_DIR:-}"
MODEL_PATHS="${MODEL_PATHS:-}"
MODEL_PATH="${MODEL_PATH:-}"
RUN_HIDDEN_COMPARE="${RUN_HIDDEN_COMPARE:-1}"
HIDDEN_LAYER="${HIDDEN_LAYER:-2}"

if [[ -z "${BUILD_DIR}" ]]; then
  if [[ -x "build/ember" ]]; then
    BUILD_DIR="build"
  elif [[ -x "build-ci/ember" ]]; then
    BUILD_DIR="build-ci"
  else
    BUILD_DIR="build"
  fi
fi

models=()
if [[ -n "${MODEL_PATHS}" ]]; then
  IFS=',' read -r -a models <<< "${MODEL_PATHS}"
elif [[ -n "${MODEL_PATH}" ]]; then
  models=("${MODEL_PATH}")
else
  echo "[gpu-check] MODEL_PATH(S) not set; skipping."
  exit 0
fi

BIN="${BUILD_DIR}/ember"
if [[ ! -x "${BIN}" ]]; then
  echo "[gpu-check] Missing binary: ${BIN}"
  exit 1
fi

for raw_model in "${models[@]}"; do
  MODEL_PATH="$(echo "${raw_model}" | xargs)"
  if [[ -z "${MODEL_PATH}" ]]; then
    continue
  fi
  if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "[gpu-check] MODEL_PATH does not exist: ${MODEL_PATH}"
    exit 1
  fi

  model_base="$(basename "${MODEL_PATH}")"
  safe_model="${model_base//[^a-zA-Z0-9._-]/_}"
  dump_dir="debug/check_${safe_model}"

  echo "[gpu-check] Running ember --check (${model_base})"
  "${BIN}" -m "${MODEL_PATH}" --check --dump-layer "${HIDDEN_LAYER}" --dump-dir "${dump_dir}" -p "Hello, my name is"

  if command -v python3 >/dev/null 2>&1; then
    if python3 - <<'PY' >/dev/null 2>&1
import importlib
import sys
sys.exit(0 if importlib.util.find_spec("transformers") else 1)
PY
    then
      echo "[gpu-check] Running compare_logits.py (${model_base})"
      args=(--model "${MODEL_PATH}" --debug-dir "${dump_dir}")
      if [[ -n "${LOGITS_MAX_ABS_THRESHOLD:-}" ]]; then
        args+=(--max-abs-threshold "${LOGITS_MAX_ABS_THRESHOLD}")
      fi
      if [[ -n "${LOGITS_MEAN_ABS_THRESHOLD:-}" ]]; then
        args+=(--mean-abs-threshold "${LOGITS_MEAN_ABS_THRESHOLD}")
      fi
      python3 scripts/compare_logits.py "${args[@]}"

      if [[ "${RUN_HIDDEN_COMPARE}" != "0" ]]; then
        echo "[gpu-check] Running compare_hidden.py (${model_base})"
        args=(--model "${MODEL_PATH}" --debug-dir "${dump_dir}" --layer "${HIDDEN_LAYER}")
        if [[ -n "${HIDDEN_MAX_ABS_THRESHOLD:-}" ]]; then
          args+=(--max-abs-threshold "${HIDDEN_MAX_ABS_THRESHOLD}")
        fi
        if [[ -n "${HIDDEN_MEAN_ABS_THRESHOLD:-}" ]]; then
          args+=(--mean-abs-threshold "${HIDDEN_MEAN_ABS_THRESHOLD}")
        fi
        python3 scripts/compare_hidden.py "${args[@]}"
      else
        echo "[gpu-check] RUN_HIDDEN_COMPARE=0; skipping compare_hidden.py"
      fi
    else
      echo "[gpu-check] transformers not available; skipping compare_logits.py/compare_hidden.py"
    fi
  else
    echo "[gpu-check] python3 not available; skipping compare_logits.py/compare_hidden.py"
  fi
done

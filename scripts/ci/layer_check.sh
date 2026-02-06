#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

BUILD_DIR="${BUILD_DIR:-build}"
MODEL_PATH="${MODEL_PATH:-}"
LAYER="${LAYER:-2}"
PROMPT="${PROMPT:-Hello, my name is}"
DUMP_DIR="${DUMP_DIR:-}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "[layer-check] MODEL_PATH is not set; skipping."
  exit 0
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[layer-check] MODEL_PATH does not exist: ${MODEL_PATH}"
  exit 1
fi

BIN="${BUILD_DIR}/ember"
if [[ ! -x "${BIN}" ]]; then
  echo "[layer-check] Missing binary: ${BIN}"
  exit 1
fi

if [[ -z "${DUMP_DIR}" ]]; then
  model_base="$(basename "${MODEL_PATH}")"
  safe_model="${model_base//[^a-zA-Z0-9._-]/_}"
  DUMP_DIR="debug/layer_check_${safe_model}_layer_${LAYER}"
fi

echo "[layer-check] Running ember --check (layer ${LAYER})"
"${BIN}" -m "${MODEL_PATH}" --check --dump-layer "${LAYER}" --dump-dir "${DUMP_DIR}" -p "${PROMPT}"

if command -v python3 >/dev/null 2>&1; then
  if python3 - <<'PY' >/dev/null 2>&1; then
import importlib
import sys
sys.exit(0 if importlib.util.find_spec("transformers") else 1)
PY
  then
    echo "[layer-check] Running compare_hidden.py (layer ${LAYER})"
    args=(--model "${MODEL_PATH}" --debug-dir "${DUMP_DIR}" --layer "${LAYER}")
    if [[ -n "${HIDDEN_MAX_ABS_THRESHOLD:-}" ]]; then
      args+=(--max-abs-threshold "${HIDDEN_MAX_ABS_THRESHOLD}")
    fi
    if [[ -n "${HIDDEN_MEAN_ABS_THRESHOLD:-}" ]]; then
      args+=(--mean-abs-threshold "${HIDDEN_MEAN_ABS_THRESHOLD}")
    fi
    python3 scripts/compare_hidden.py "${args[@]}"
  else
    echo "[layer-check] transformers not available; skipping compare_hidden.py"
  fi
else
  echo "[layer-check] python3 not available; skipping compare_hidden.py"
fi

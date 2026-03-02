#!/usr/bin/env bash
# shellcheck shell=bash

ci_log() {
  local tag="$1"
  shift
  echo "[${tag}] $*"
}

ci_err() {
  local tag="$1"
  shift
  echo "[${tag}] $*" >&2
}

ci_die() {
  local tag="$1"
  shift
  ci_err "${tag}" "$*"
  exit 1
}

ci_expect_value() {
  local tag="$1"
  local opt="$2"
  local val="${3:-}"
  if [[ -z "${val}" ]]; then
    ci_die "${tag}" "Missing value for ${opt}"
  fi
}

ci_init_common_env() {
  CI_BUILD_DIR="${CI_BUILD_DIR:-${BUILD_DIR:-}}"
  CI_MODEL_PATH="${CI_MODEL_PATH:-${MODEL_PATH:-}}"
  CI_MODEL_PATHS="${CI_MODEL_PATHS:-${MODEL_PATHS:-}}"
  CI_MODEL_ID="${CI_MODEL_ID:-${MODEL_ID:-}}"
  CI_HUB_ROOT="${CI_HUB_ROOT:-${HUB_ROOT:-}}"
  CI_GPUS="${CI_GPUS:-${EMBER_DEVICES:-${EMBER_DEVICE:-1}}}"
  CI_PYTHON_BIN="${CI_PYTHON_BIN:-${PYTHON_BIN:-}}"
  CI_PYTHON_RUNNER="${CI_PYTHON_RUNNER:-${PYTHON_RUNNER:-auto}}"
  CI_REQUIRE_HF_COMPARE="${CI_REQUIRE_HF_COMPARE:-${REQUIRE_HF_COMPARE:-1}}"
  CI_ARG_MODEL_PATH_SET=0
  CI_ARG_MODEL_PATHS_SET=0
  CI_ARG_MODEL_ID_SET=0
  CI_ARG_HUB_ROOT_SET=0
}

ci_apply_common_precedence() {
  # Explicit CLI options must win over inherited env vars.
  if [[ "${CI_ARG_MODEL_PATHS_SET}" == "1" ]]; then
    CI_MODEL_PATH=""
  fi
  if [[ "${CI_ARG_MODEL_PATH_SET}" == "1" ]]; then
    CI_MODEL_PATHS=""
  fi
  if [[ "${CI_ARG_MODEL_ID_SET}" == "1" && "${CI_ARG_MODEL_PATH_SET}" != "1" && "${CI_ARG_MODEL_PATHS_SET}" != "1" ]]; then
    CI_MODEL_PATH=""
    CI_MODEL_PATHS=""
  fi
}

ci_set_model_arg() {
  local raw="$1"
  local expanded
  expanded="$(ci_expand_path "${raw}")"
  if [[ -d "${expanded}" ]]; then
    CI_MODEL_PATH="${raw}"
    CI_ARG_MODEL_PATH_SET=1
  else
    CI_MODEL_ID="${raw}"
    CI_ARG_MODEL_ID_SET=1
  fi
}

ci_parse_common_args() {
  local -n _out_rest="$1"
  shift
  _out_rest=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --build-dir)
        ci_expect_value "ci-common" "$1" "${2:-}"
        CI_BUILD_DIR="$2"
        shift 2
        ;;
      --model-path)
        ci_expect_value "ci-common" "$1" "${2:-}"
        CI_MODEL_PATH="$2"
        CI_ARG_MODEL_PATH_SET=1
        shift 2
        ;;
      --model-paths)
        ci_expect_value "ci-common" "$1" "${2:-}"
        CI_MODEL_PATHS="$2"
        CI_ARG_MODEL_PATHS_SET=1
        shift 2
        ;;
      --model-id)
        ci_expect_value "ci-common" "$1" "${2:-}"
        CI_MODEL_ID="$2"
        CI_ARG_MODEL_ID_SET=1
        shift 2
        ;;
      --model|--model-b)
        ci_expect_value "ci-common" "$1" "${2:-}"
        ci_set_model_arg "$2"
        shift 2
        ;;
      --hub-root)
        ci_expect_value "ci-common" "$1" "${2:-}"
        CI_HUB_ROOT="$2"
        CI_ARG_HUB_ROOT_SET=1
        shift 2
        ;;
      --gpus|--devices)
        ci_expect_value "ci-common" "$1" "${2:-}"
        CI_GPUS="$2"
        shift 2
        ;;
      --python-bin)
        ci_expect_value "ci-common" "$1" "${2:-}"
        CI_PYTHON_BIN="$2"
        shift 2
        ;;
      --python-runner)
        ci_expect_value "ci-common" "$1" "${2:-}"
        CI_PYTHON_RUNNER="$2"
        shift 2
        ;;
      --require-hf-compare)
        CI_REQUIRE_HF_COMPARE=1
        shift
        ;;
      --no-require-hf-compare)
        CI_REQUIRE_HF_COMPARE=0
        shift
        ;;
      --)
        shift
        _out_rest+=("$@")
        break
        ;;
      *)
        _out_rest+=("$1")
        shift
        ;;
    esac
  done
  ci_apply_common_precedence
}

ci_resolve_build_dir() {
  local default_dir="${1:-build}"
  if [[ -n "${CI_BUILD_DIR:-}" ]]; then
    return 0
  fi
  if [[ -x "build/ember" ]]; then
    CI_BUILD_DIR="build"
  elif [[ -x "build-ci/ember" ]]; then
    CI_BUILD_DIR="build-ci"
  else
    CI_BUILD_DIR="${default_dir}"
  fi
}

ci_expand_path() {
  local p="$1"
  if [[ "${p}" == "~" ]]; then
    printf '%s\n' "${HOME}"
    return 0
  fi
  if [[ "${p}" == "~/"* ]]; then
    printf '%s/%s\n' "${HOME}" "${p#~/}"
    return 0
  fi
  printf '%s\n' "${p}"
}

ci_default_hub_root() {
  printf '%s\n' "${HOME}/.cache/huggingface/hub"
}

ci_model_id_with_default_org() {
  local model_id="$1"
  if [[ "${model_id}" == */* ]]; then
    printf '%s\n' "${model_id}"
  else
    printf 'Qwen/%s\n' "${model_id}"
  fi
}

ci_latest_snapshot_dir() {
  local root="$1"
  local snap_root="${root}/snapshots"
  [[ -d "${snap_root}" ]] || return 1

  local best=""
  local best_mtime=-1
  local d mtime
  shopt -s nullglob
  for d in "${snap_root}"/*; do
    [[ -d "${d}" ]] || continue
    mtime="$(stat -c %Y "${d}" 2>/dev/null || printf '0')"
    if (( mtime > best_mtime )); then
      best_mtime="${mtime}"
      best="${d}"
    fi
  done
  shopt -u nullglob

  [[ -n "${best}" ]] || return 1
  printf '%s\n' "${best}"
}

ci_is_model_dir() {
  local p="$1"
  [[ -d "${p}" && -f "${p}/config.json" ]] || return 1
  shopt -s nullglob
  local files=("${p}"/*.safetensors)
  shopt -u nullglob
  [[ ${#files[@]} -gt 0 ]]
}

ci_resolve_path_or_snapshot() {
  local raw="$1"
  local p
  p="$(ci_expand_path "${raw}")"
  [[ -d "${p}" ]] || return 1

  if ci_is_model_dir "${p}"; then
    printf '%s\n' "${p}"
    return 0
  fi

  local snap
  if snap="$(ci_latest_snapshot_dir "${p}")" && ci_is_model_dir "${snap}"; then
    printf '%s\n' "${snap}"
    return 0
  fi
  return 1
}

ci_resolve_model_path_if_needed() {
  local tag="$1"
  if [[ -n "${CI_MODEL_PATH:-}" ]]; then
    local resolved
    if ! resolved="$(ci_resolve_path_or_snapshot "${CI_MODEL_PATH}")"; then
      ci_die "${tag}" "MODEL_PATH is invalid or not a local model dir: ${CI_MODEL_PATH}"
    fi
    CI_MODEL_PATH="${resolved}"
    return 0
  fi

  if [[ -z "${CI_MODEL_ID:-}" ]]; then
    return 1
  fi

  local hub_root
  if [[ -n "${CI_HUB_ROOT:-}" ]]; then
    hub_root="$(ci_expand_path "${CI_HUB_ROOT}")"
  else
    hub_root="$(ci_default_hub_root)"
  fi
  local full_id
  full_id="$(ci_model_id_with_default_org "${CI_MODEL_ID}")"
  local cache_dir="${hub_root}/models--${full_id//\//--}"

  local resolved
  if ! resolved="$(ci_resolve_path_or_snapshot "${cache_dir}")"; then
    ci_die "${tag}" "Failed to resolve model from hub cache: id=${full_id}, hub_root=${hub_root}"
  fi
  CI_MODEL_PATH="${resolved}"
  return 0
}

ci_resolve_python_runtime() {
  if [[ "${CI_PYTHON_RUNNER}" == "auto" ]]; then
    if command -v uv >/dev/null 2>&1; then
      CI_PYTHON_RUNNER="uv"
    else
      CI_PYTHON_RUNNER="direct"
    fi
  fi

  if [[ "${CI_PYTHON_RUNNER}" != "uv" && "${CI_PYTHON_RUNNER}" != "direct" ]]; then
    return 1
  fi

  if [[ "${CI_PYTHON_RUNNER}" == "uv" ]] && ! command -v uv >/dev/null 2>&1; then
    return 1
  fi

  if [[ -n "${CI_PYTHON_BIN}" ]]; then
    CI_PYTHON_BIN="$(ci_expand_path "${CI_PYTHON_BIN}")"
    [[ -x "${CI_PYTHON_BIN}" ]] || return 1
    return 0
  fi

  if [[ -x "${PWD}/torch-env/bin/python" ]]; then
    CI_PYTHON_BIN="${PWD}/torch-env/bin/python"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    CI_PYTHON_BIN="$(command -v python3)"
    return 0
  fi
  return 1
}

ci_run_python() {
  if [[ "${CI_PYTHON_RUNNER}" == "uv" ]]; then
    uv run --python "${CI_PYTHON_BIN}" -- python "$@"
  else
    "${CI_PYTHON_BIN}" "$@"
  fi
}

ci_python_has_module() {
  local module_name="$1"
  ci_run_python -c "import importlib,sys;sys.exit(0 if importlib.util.find_spec('${module_name}') else 1)" >/dev/null 2>&1
}

ci_first_gpu() {
  local gpus="${1:-${CI_GPUS:-1}}"
  local first
  IFS=',' read -r first _ <<< "${gpus}"
  first="${first//[[:space:]]/}"
  if [[ -z "${first}" ]]; then
    first="0"
  fi
  printf '%s\n' "${first}"
}

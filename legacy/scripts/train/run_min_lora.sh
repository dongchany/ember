#!/usr/bin/env bash
set -euo pipefail

# Wrapper for minimal LoRA training with regular Python environment.
# Usage:
#   scripts/train/run_min_lora.sh --output-dir reports/adapters/qwen3_4b_min

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"
python scripts/train/train_min_lora_adapter.py "$@"


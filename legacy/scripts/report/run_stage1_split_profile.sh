#!/usr/bin/env bash
set -euo pipefail

python3 "$(dirname "$0")/run_stage1_split_profile.py" "$@"

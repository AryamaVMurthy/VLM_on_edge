#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

source "${ROOT_DIR}/2.42.0.251225/qnn_env/bin/activate"

python "${ROOT_DIR}/handoff/scripts/qaihub_compile_fp16.py"

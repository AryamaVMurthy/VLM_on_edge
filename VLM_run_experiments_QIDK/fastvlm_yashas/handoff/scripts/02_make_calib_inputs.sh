#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CALIB_DIR="${CALIB_DIR:-${ROOT_DIR}/models/fastvlm-calib-raw}"
IMAGE_SIZE="${IMAGE_SIZE:-1024}"
MAX_SAMPLES="${MAX_SAMPLES:-8}"

source "${ROOT_DIR}/2.42.0.251225/qnn_env/bin/activate"

mkdir -p "${CALIB_DIR}"

if [[ $# -gt 0 ]]; then
  echo "Generating calibration inputs from provided images..."
  python "${ROOT_DIR}/handoff/scripts/make_calib_inputs.py" \
    --images "$@" \
    --out-dir "${CALIB_DIR}" \
    --image-size "${IMAGE_SIZE}" \
    --max-samples "${MAX_SAMPLES}"
else
  echo "Generating random calibration inputs..."
  python "${ROOT_DIR}/handoff/scripts/make_calib_inputs.py" \
    --random \
    --out-dir "${CALIB_DIR}" \
    --image-size "${IMAGE_SIZE}" \
    --max-samples "${MAX_SAMPLES}"
fi

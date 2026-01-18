#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  run_fastvlm_device.sh <image_path> <prompt> [max_new_tokens] [image_size] [decoder_precision] [vision_precision]

Runs:
  - Vision encoder (int8) on Android NPU (HTP)
  - Text embedder (fp16) on Android NPU (HTP)
  - Decoder (fp32/z8/q4, GenAiTransformer) on Android CPU via Genie

Profiling:
  Always enabled. Genie profile stats are printed from the profile JSON.

Notes:
  Set max_new_tokens to 0 to auto-cap at context limit (prevents overflow).
  Set SKIP_PUSH=1 to skip re-pushing libraries/binaries (inputs are still pushed).

Example:
  ./run_fastvlm_device.sh ./models/fastvlm_test.jpg "Describe the image in detail." 250 1024 fp32 fp16
EOF
}

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

IMAGE_PATH="$1"
PROMPT="$2"
MAX_NEW_TOKENS="${3:-0}"
IMAGE_SIZE="${4:-1024}"
DECODER_PRECISION="${5:-${DECODER_PRECISION:-fp32}}"
VISION_PRECISION="${6:-${VISION_PRECISION:-int8}}"
PROFILE="1"

if [[ ! -f "${IMAGE_PATH}" ]]; then
  echo "Image not found: ${IMAGE_PATH}" >&2
  exit 1
fi

source "${ROOT_DIR}/2.42.0.251225/qnn_env/bin/activate"
set +u
source "${ROOT_DIR}/2.42.0.251225/bin/envsetup.sh"
set -u

if [[ -z "${ANDROID_NDK_ROOT:-}" ]]; then
  echo "ANDROID_NDK_ROOT is not set. Please export it before running." >&2
  exit 1
fi

if ! adb get-state >/dev/null 2>&1; then
  echo "No adb device detected. Run 'adb devices' and ensure one device is connected." >&2
  exit 1
fi

MODEL_BIN="${ROOT_DIR}/models/fastvlm-genie/fastvlm_model.bin"
case "${DECODER_PRECISION}" in
  fp32)
    MODEL_BIN="${ROOT_DIR}/models/fastvlm-genie/fastvlm_model.bin"
    ;;
  int8|z8)
    MODEL_BIN="${ROOT_DIR}/models/fastvlm-genie/fastvlm_model_z8.bin"
    ;;
  int4|q4)
    MODEL_BIN="${ROOT_DIR}/models/fastvlm-genie/fastvlm_model_q4.bin"
    ;;
  *)
    echo "Unsupported decoder_precision: ${DECODER_PRECISION} (use fp32, int8, or int4)" >&2
    exit 1
    ;;
esac

case "${VISION_PRECISION}" in
  int8|fp16)
    ;;
  *)
    echo "Unsupported vision_precision: ${VISION_PRECISION} (use int8 or fp16)" >&2
    exit 1
    ;;
esac

if [[ ! -f "${MODEL_BIN}" ]]; then
  echo "Decoder bin not found: ${MODEL_BIN}" >&2
  exit 1
fi

SCRIPT="${ROOT_DIR}/handoff/scripts/run_e2e_int8_device.sh"
if [[ ! -x "${SCRIPT}" ]]; then
  chmod +x "${SCRIPT}"
fi

IMAGE_PATH="${IMAGE_PATH}" \
PROMPT="${PROMPT}" \
MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
IMAGE_SIZE="${IMAGE_SIZE}" \
PROFILE="${PROFILE}" \
MODEL_BIN="${MODEL_BIN}" \
VISION_PRECISION="${VISION_PRECISION}" \
SKIP_PUSH="${SKIP_PUSH:-0}" \
LUT_BIN="${ROOT_DIR}/models/fastvlm-genie/LUT.bin" \
"${SCRIPT}"

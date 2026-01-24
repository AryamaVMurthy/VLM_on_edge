#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE_DIR="${DEVICE_DIR:-/data/local/tmp/fastvlm}"

OUT_DIR="${OUT_DIR:-${ROOT_DIR}/host_outputs}"
mkdir -p "${OUT_DIR}/vision" "${OUT_DIR}/text"

adb pull "${DEVICE_DIR}/outputs/vision/Result_0/output_0.raw" "${OUT_DIR}/vision/vision_output.raw"
adb pull "${DEVICE_DIR}/outputs/text/Result_0/output_0.raw" "${OUT_DIR}/text/text_output.raw"

echo "==> Outputs pulled to ${OUT_DIR}"

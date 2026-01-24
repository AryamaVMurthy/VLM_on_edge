#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE_DIR="${DEVICE_DIR:-/data/local/tmp/fastvlm}"

VISION_SRC="${VISION_SRC:-${ROOT_DIR}/host_inputs/vision}"
TEXT_SRC="${TEXT_SRC:-${ROOT_DIR}/host_inputs/text}"

if [[ ! -d "${VISION_SRC}" ]]; then
  echo "Missing vision input dir: ${VISION_SRC}"
  exit 1
fi
if [[ ! -d "${TEXT_SRC}" ]]; then
  echo "Missing text input dir: ${TEXT_SRC}"
  exit 1
fi

adb shell "mkdir -p ${DEVICE_DIR}/inputs/vision ${DEVICE_DIR}/inputs/text"
adb push "${VISION_SRC}/pixel_values.raw" "${DEVICE_DIR}/inputs/vision/"
adb push "${VISION_SRC}/input_list.txt" "${DEVICE_DIR}/inputs/vision/"

adb push "${TEXT_SRC}/input_ids.raw" "${DEVICE_DIR}/inputs/text/"
adb push "${TEXT_SRC}/input_list.txt" "${DEVICE_DIR}/inputs/text/"

echo "==> Inputs pushed"

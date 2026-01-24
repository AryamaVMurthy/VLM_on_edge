#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE_DIR="${DEVICE_DIR:-/data/local/tmp/fastvlm}"

VISION_BIN="${VISION_BIN:-${ROOT_DIR}/../../../fastVLM-bins/fastvlm/vision_encoder_fp16.bin}"
TEXT_BIN="${TEXT_BIN:-${ROOT_DIR}/../../../fastVLM-bins/fastvlm/text_embedder_fp16.bin}"

if [[ ! -f "${VISION_BIN}" ]]; then
  echo "Vision encoder binary not found: ${VISION_BIN}"
  exit 1
fi
if [[ ! -f "${TEXT_BIN}" ]]; then
  echo "Text embedder binary not found: ${TEXT_BIN}"
  exit 1
fi

echo "==> Pushing encoder binaries"
adb shell "mkdir -p ${DEVICE_DIR}/encoders"
adb push "${VISION_BIN}" "${DEVICE_DIR}/encoders/vision_encoder_fp16.bin"
adb push "${TEXT_BIN}" "${DEVICE_DIR}/encoders/text_embedder_fp16.bin"

echo "==> Encoder binaries pushed"

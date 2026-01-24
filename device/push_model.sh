#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE_DIR="${DEVICE_DIR:-/data/local/tmp/fastvlm}"

LUT_FILE="${LUT_FILE:-}"
if [[ -z "${LUT_FILE}" ]]; then
  if [[ -f "${ROOT_DIR}/embedding.bin" ]]; then
    LUT_FILE="${ROOT_DIR}/embedding.bin"
  elif [[ -f "${ROOT_DIR}/embedding_fp16.bin" ]]; then
    LUT_FILE="${ROOT_DIR}/embedding_fp16.bin"
  else
    echo "Missing embedding LUT. Provide embedding.bin or embedding_fp16.bin."
    exit 1
  fi
fi

# Pick latest compiled bin directory
BIN_DIR="${BIN_DIR:-}"
if [[ -z "${BIN_DIR}" ]]; then
  for candidate in $(ls -td ${ROOT_DIR}/qaihub_bins/fastvlm_full_* 2>/dev/null); do
    if ls -1 "${candidate}"/*.bin >/dev/null 2>&1; then
      BIN_DIR="${candidate}"
      break
    fi
  done
fi
if [[ -z "${BIN_DIR}" ]]; then
  for candidate in $(ls -td ${ROOT_DIR}/qaihub_bins/full_model_* 2>/dev/null); do
    if ls -1 "${candidate}"/*.bin >/dev/null 2>&1; then
      BIN_DIR="${candidate}"
      break
    fi
  done
fi
if [[ -z "${BIN_DIR}" ]]; then
  echo "No compiled bin directory found under ${ROOT_DIR}/qaihub_bins/"
  exit 1
fi

BIN_FILE="${BIN_DIR}/fastvlm_full.bin"
if [[ ! -f "${BIN_FILE}" ]]; then
  # Older bins use job_* naming; pick the first .bin if present.
  CANDIDATE="$(ls -1 ${BIN_DIR}/*.bin 2>/dev/null | head -n 1 || true)"
  if [[ -n "${CANDIDATE}" ]]; then
    BIN_FILE="${CANDIDATE}"
  fi
fi
if [[ ! -f "${BIN_FILE}" ]]; then
  echo "Missing ${BIN_FILE}. Run compile_final.py and ensure output was renamed."
  exit 1
fi

echo "==> Pushing model artifacts from ${BIN_DIR}"
adb shell "mkdir -p ${DEVICE_DIR}"
adb push "${BIN_FILE}" "${DEVICE_DIR}/fastvlm_full.bin"
adb push "${ROOT_DIR}/fastvlm_genie_npu.json" "${DEVICE_DIR}/"
if [[ -f "${ROOT_DIR}/fastvlm_genie_npu_prefill.json" ]]; then
  adb push "${ROOT_DIR}/fastvlm_genie_npu_prefill.json" "${DEVICE_DIR}/"
fi
adb push "${ROOT_DIR}/tokenizer.json" "${DEVICE_DIR}/"
adb push "${LUT_FILE}" "${DEVICE_DIR}/"

echo "==> Model artifacts pushed"

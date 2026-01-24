#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE_DIR="${DEVICE_DIR:-/data/local/tmp/fastvlm}"
TOKENS_SRC="${TOKENS_SRC:-${ROOT_DIR}/host_outputs/combined/prefill_tokens}"
COMBINED_SRC="${COMBINED_SRC:-${ROOT_DIR}/host_outputs/combined/combined_embeddings.raw}"
PUSH_TOKEN_FILES="${PUSH_TOKEN_FILES:-1}"

adb shell "mkdir -p ${DEVICE_DIR}/inputs/prefill"
adb shell "rm -f ${DEVICE_DIR}/inputs/prefill/token_*.raw"
if [[ "${PUSH_TOKEN_FILES}" != "0" ]]; then
  if [[ ! -d "${TOKENS_SRC}" ]]; then
    echo "Missing tokens directory: ${TOKENS_SRC}"
    exit 1
  fi
  adb push "${TOKENS_SRC}"/* "${DEVICE_DIR}/inputs/prefill/"
else
  echo "==> Skipping per-token files (PUSH_TOKEN_FILES=0)"
fi
if [[ -f "${COMBINED_SRC}" ]]; then
  adb push "${COMBINED_SRC}" "${DEVICE_DIR}/inputs/prefill/combined_embeddings.raw"
fi

echo "==> Prefill tokens pushed"

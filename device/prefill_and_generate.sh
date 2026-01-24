#!/usr/bin/env bash
set -euo pipefail

DEVICE_DIR="${DEVICE_DIR:-/data/local/tmp/fastvlm}"
TOKEN_DIR="${TOKEN_DIR:-${DEVICE_DIR}/inputs/prefill}"
STATE_DIR="${STATE_DIR:-${DEVICE_DIR}/state}"
CONFIG="${CONFIG:-${DEVICE_DIR}/fastvlm_genie_npu.json}"
PREFILL_CONFIG="${PREFILL_CONFIG:-${DEVICE_DIR}/fastvlm_genie_npu_prefill.json}"
LUT="${LUT:-}"
if [[ -z "${LUT}" ]]; then
  if adb shell "test -f ${DEVICE_DIR}/embedding.bin" >/dev/null 2>&1; then
    LUT="${DEVICE_DIR}/embedding.bin"
  else
    LUT="${DEVICE_DIR}/embedding_fp16.bin"
  fi
fi
EMBED_FILE="${EMBED_FILE:-${TOKEN_DIR}/combined_embeddings.raw}"
GENIE_LOG="${GENIE_LOG:-}"
PROFILE_OUT="${PROFILE_OUT:-}"
EMBEDDING_FILE_DTYPE="${EMBEDDING_FILE_DTYPE:-}"
EMBEDDING_FILE_SCALE="${EMBEDDING_FILE_SCALE:-1}"
EMBEDDING_FILE_OFFSET="${EMBEDDING_FILE_OFFSET:-0}"
LUT_DTYPE="${LUT_DTYPE:-}"
LUT_SCALE="${LUT_SCALE:-1}"
LUT_OFFSET="${LUT_OFFSET:-0}"
FORCE_TOKEN_PREFILL="${FORCE_TOKEN_PREFILL:-0}"
SAVE_STATE="${SAVE_STATE:-}"
RESTORE_STATE="${RESTORE_STATE:-}"
PREFILL_ONLY="${PREFILL_ONLY:-0}"
EMBEDDING_QUERY_OUTPUT_TYPE="${EMBEDDING_QUERY_OUTPUT_TYPE:-text}"

format_embed_arg() {
  local path="$1"
  if [[ -n "${EMBEDDING_FILE_DTYPE}" ]]; then
    echo "${path},${EMBEDDING_FILE_DTYPE},${EMBEDDING_FILE_SCALE},${EMBEDDING_FILE_OFFSET}"
  else
    echo "${path}"
  fi
}

format_lut_arg() {
  local path="$1"
  if [[ -n "${LUT_DTYPE}" ]]; then
    echo "${path},${LUT_DTYPE},${LUT_SCALE},${LUT_OFFSET}"
  else
    echo "${path}"
  fi
}

adb shell "mkdir -p ${STATE_DIR}"

STATE_PATH="${STATE_DIR}/dialog_state.bin"
if [[ -n "${SAVE_STATE}" ]]; then
  STATE_PATH="${SAVE_STATE}"
fi
STATE_DIRNAME="$(dirname "${STATE_PATH}")"
adb shell "mkdir -p ${STATE_DIRNAME}"
adb shell "mkdir -p ${STATE_PATH}"

BASE_ARGS=""
if [[ -n "${GENIE_LOG}" ]]; then
  BASE_ARGS="${BASE_ARGS} --log ${GENIE_LOG}"
fi

use_combined=1
if [[ "${FORCE_TOKEN_PREFILL}" != "0" ]]; then
  use_combined=0
elif ! adb shell "test -f ${EMBED_FILE}" >/dev/null 2>&1; then
  use_combined=0
fi

if [[ "${use_combined}" -eq 1 ]]; then
  echo "==> Prefill + generate (combined embeddings)"
  EMBED_FILE_ARG="$(format_embed_arg "${EMBED_FILE}")"
  LUT_ARG="$(format_lut_arg "${LUT}")"
  RESTORE_ARG=""
  SAVE_ARG=""
  if [[ -n "${RESTORE_STATE}" ]]; then
    RESTORE_ARG="--restore ${RESTORE_STATE}"
  fi
  if [[ -n "${SAVE_STATE}" ]]; then
    SAVE_ARG="--save ${STATE_PATH}"
  fi
  PROFILE_ARG=""
  if [[ -n "${PROFILE_OUT}" ]]; then
    PROFILE_ARG="--profile ${PROFILE_OUT}"
  fi
  adb shell "export ADSP_LIBRARY_PATH=${DEVICE_DIR} && \
    export LD_LIBRARY_PATH=${DEVICE_DIR}:/vendor/lib64 && \
    cd ${DEVICE_DIR} && \
    ./genie-t2t-run \
      --config ${CONFIG} \
      ${RESTORE_ARG} \
      --embedding_file ${EMBED_FILE_ARG} \
      --embedding_table ${LUT_ARG} \
      --embedding_query_output_type ${EMBEDDING_QUERY_OUTPUT_TYPE} \
      ${SAVE_ARG} \
      ${BASE_ARGS} \
      ${PROFILE_ARG}" | tr -d '\r'
else
  TOKEN_LIST="$(adb shell "ls -1 ${TOKEN_DIR}/token_*.raw 2>/dev/null" | tr -d '\r')"
  if [[ -z "${TOKEN_LIST}" ]]; then
    echo "No token_*.raw found in ${TOKEN_DIR}"
    exit 1
  fi

  TOKENS=( ${TOKEN_LIST} )
  LAST_INDEX=$(( ${#TOKENS[@]} - 1 ))
  echo "==> Prefill: ${#TOKENS[@]} tokens"
  for idx in "${!TOKENS[@]}"; do
    token_path="${TOKENS[$idx]}"
    is_last=0
    if [[ "${idx}" -eq "${LAST_INDEX}" ]]; then
      is_last=1
    fi

    if [[ "${idx}" -eq 0 ]]; then
      if [[ -n "${RESTORE_STATE}" ]]; then
        RESTORE_ARG="--restore ${RESTORE_STATE}"
      else
        RESTORE_ARG=""
      fi
    else
      RESTORE_ARG="--restore ${STATE_PATH}"
    fi

    if [[ "${is_last}" -eq 1 ]]; then
      if [[ "${PREFILL_ONLY}" != "0" ]]; then
        OUTPUT_TYPE="token"
        if adb shell "test -f ${PREFILL_CONFIG}" >/dev/null 2>&1; then
          ACTIVE_CONFIG="${PREFILL_CONFIG}"
        else
          ACTIVE_CONFIG="${CONFIG}"
        fi
      else
        OUTPUT_TYPE="text"
        ACTIVE_CONFIG="${CONFIG}"
      fi
    else
      OUTPUT_TYPE="token"
      if adb shell "test -f ${PREFILL_CONFIG}" >/dev/null 2>&1; then
        ACTIVE_CONFIG="${PREFILL_CONFIG}"
      else
        ACTIVE_CONFIG="${CONFIG}"
      fi
    fi

    TOKEN_ARG="$(format_embed_arg "${token_path}")"
    LUT_ARG="$(format_lut_arg "${LUT}")"
    PROFILE_ARG=""
    if [[ -n "${PROFILE_OUT}" ]]; then
      base="${PROFILE_OUT}"
      ext=""
      if [[ "${base}" == *.json ]]; then
        ext=".json"
        base="${base%.json}"
      fi
      PROFILE_ARG="--profile ${base}_token_${idx}${ext}"
    fi
    adb shell "export ADSP_LIBRARY_PATH=${DEVICE_DIR} && \
      export LD_LIBRARY_PATH=${DEVICE_DIR}:/vendor/lib64 && \
      cd ${DEVICE_DIR} && \
      ./genie-t2t-run \
        --config ${ACTIVE_CONFIG} \
        ${RESTORE_ARG} \
        --embedding_file ${TOKEN_ARG} \
        --embedding_table ${LUT_ARG} \
        --embedding_query_output_type ${OUTPUT_TYPE} \
        --save ${STATE_PATH} \
        ${BASE_ARGS} \
        ${PROFILE_ARG}" | tr -d '\r'
  done
fi

echo "==> Prefill+generate complete"

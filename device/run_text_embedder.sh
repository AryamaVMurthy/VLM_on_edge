#!/usr/bin/env bash
set -euo pipefail

DEVICE_DIR="${DEVICE_DIR:-/data/local/tmp/fastvlm}"
INPUT_DIR="${INPUT_DIR:-${DEVICE_DIR}/inputs/text}"
OUTPUT_DIR="${OUTPUT_DIR:-${DEVICE_DIR}/outputs/text}"

adb shell "mkdir -p ${OUTPUT_DIR}"

adb shell "export ADSP_LIBRARY_PATH=${DEVICE_DIR} && \
  export LD_LIBRARY_PATH=${DEVICE_DIR}:/vendor/lib64 && \
  cd ${DEVICE_DIR} && \
  ./qnn-net-run \
    --backend libQnnHtp.so \
    --retrieve_context encoders/text_embedder_fp16.bin \
    --input_list ${INPUT_DIR}/input_list.txt \
    --use_native_input_files \
    --use_native_output_files \
    --output_dir ${OUTPUT_DIR} \
    --num_inferences 1"

echo "==> Text embedder finished"

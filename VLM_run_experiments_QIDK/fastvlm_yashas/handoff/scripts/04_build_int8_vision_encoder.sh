#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

source "${ROOT_DIR}/2.42.0.251225/qnn_env/bin/activate"
set +u
source "${ROOT_DIR}/2.42.0.251225/bin/envsetup.sh"
set -u

SDK_ROOT="${QNN_SDK_ROOT:-${QAIRT_SDK_ROOT:-}}"
if [[ -z "${SDK_ROOT}" ]]; then
  echo "QNN_SDK_ROOT or QAIRT_SDK_ROOT must be set by envsetup.sh" >&2
  exit 1
fi

VISION_ONNX="${VISION_ONNX:-${ROOT_DIR}/models/fastvlm-onnx-fp32/vision_encoder_fp32.onnx}"
CALIB_LIST="${CALIB_LIST:-${ROOT_DIR}/models/fastvlm-calib-raw/input_list.txt}"
QNN_INT8_DIR="${QNN_INT8_DIR:-${ROOT_DIR}/models/fastvlm-qnn-int8}"
MODEL_PREFIX="${MODEL_PREFIX:-vision_encoder_int8_qnnmodel}"
OUT_PREFIX="${QNN_INT8_DIR}/${MODEL_PREFIX}"

LOCAL_CTX_DIR="${LOCAL_CTX_DIR:-${ROOT_DIR}/models/fastvlm-genie/local_context/vision_int8_qnnmodel}"
CTX_OUTPUT_DIR="${CTX_OUTPUT_DIR:-${LOCAL_CTX_DIR}/output}"
CTX_BIN="${CTX_BIN:-fastvlm_vision_encoder_int8_htp.bin}"
CTX_INFO_JSON="${CTX_INFO_JSON:-${LOCAL_CTX_DIR}/vision_int8_ctx_info.json}"

if [[ ! -f "${VISION_ONNX}" ]]; then
  echo "Missing ONNX: ${VISION_ONNX}" >&2
  exit 1
fi
if [[ ! -f "${CALIB_LIST}" ]]; then
  echo "Missing calibration list: ${CALIB_LIST}" >&2
  exit 1
fi

mkdir -p "${QNN_INT8_DIR}"
mkdir -p "${LOCAL_CTX_DIR}"
mkdir -p "${CTX_OUTPUT_DIR}"

echo "Converting ONNX -> QNN (int8)..."
qnn-onnx-converter \
  --input_network "${VISION_ONNX}" \
  --input_dim "pixel_values" "1,3,1024,1024" \
  --input_layout "pixel_values" "NCHW" \
  --input_list "${CALIB_LIST}" \
  --act_bitwidth 8 \
  --weights_bitwidth 8 \
  --bias_bitwidth 8 \
  --act_quantizer tf \
  --act_quantizer_calibration min-max \
  --act_quantizer_schema asymmetric \
  --param_quantizer_calibration min-max \
  --param_quantizer_schema asymmetric \
  --float_bitwidth 32 \
  --float_bias_bitwidth 32 \
  --out_node "image_features" \
  -o "${OUT_PREFIX}"

echo "Building QNN model libraries..."
qnn-model-lib-generator \
  -c "${OUT_PREFIX}.cpp" \
  -b "${OUT_PREFIX}.bin" \
  -t aarch64-android x86_64-linux-clang \
  -l "${MODEL_PREFIX}" \
  -o "${LOCAL_CTX_DIR}"

echo "Generating HTP context binary..."
qnn-context-binary-generator \
  --backend "${SDK_ROOT}/lib/x86_64-linux-clang/libQnnHtp.so" \
  --model "${LOCAL_CTX_DIR}/x86_64-linux-clang/lib${MODEL_PREFIX}.so" \
  --output_dir "${CTX_OUTPUT_DIR}" \
  --binary_file "${CTX_BIN}"

echo "Writing context info JSON..."
qnn-context-binary-utility \
  --context_binary "${CTX_OUTPUT_DIR}/${CTX_BIN}" \
  --json_file "${CTX_INFO_JSON}"

echo "Done."

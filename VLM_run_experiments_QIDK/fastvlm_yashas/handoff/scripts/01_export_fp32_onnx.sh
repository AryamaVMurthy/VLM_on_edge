#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/models/apple--FastVLM-0.5B}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/models/fastvlm-onnx-fp32}"
IMAGE_SIZE="${IMAGE_SIZE:-1024}"
SEQ_LEN="${SEQ_LEN:-512}"
OPSET="${OPSET:-14}"

source "${ROOT_DIR}/2.42.0.251225/qnn_env/bin/activate"

mkdir -p "${OUT_DIR}"

echo "Exporting vision encoder fp32 ONNX..."
python "${ROOT_DIR}/handoff/scripts/export_fastvlm_fp16.py" \
  --model-dir "${MODEL_DIR}" \
  --out-dir "${OUT_DIR}" \
  --image-size "${IMAGE_SIZE}" \
  --seq-len "${SEQ_LEN}" \
  --opset "${OPSET}" \
  --dtype fp32 \
  --no-constant-folding \
  --only vision

echo "Exporting embed tokens fp32 ONNX..."
python "${ROOT_DIR}/handoff/scripts/export_fastvlm_fp16.py" \
  --model-dir "${MODEL_DIR}" \
  --out-dir "${OUT_DIR}" \
  --image-size "${IMAGE_SIZE}" \
  --seq-len "${SEQ_LEN}" \
  --opset "${OPSET}" \
  --dtype fp32 \
  --no-constant-folding \
  --only embed

echo "Exporting decoder fixed KV fp32 ONNX..."
python "${ROOT_DIR}/handoff/scripts/export_fastvlm_decoder_fixed_kv.py" \
  --model-dir "${MODEL_DIR}" \
  --out "${OUT_DIR}/decoder_fixed_kv_fp32.onnx" \
  --cache-len "${SEQ_LEN}" \
  --decode-len 1 \
  --opset "${OPSET}" \
  --dtype fp32

echo "Done."

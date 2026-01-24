#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE_PATH="${1:-}"
PROMPT="${2:-}"
PROMPT_FORMAT="${PROMPT_FORMAT:-raw}"
COMBINE_ORDER="${COMBINE_ORDER:-vision-text}"
ADD_BOS="${ADD_BOS:-0}"
BOS_TOKEN="${BOS_TOKEN:-151644}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-512}"
VISION_TOKENS="${VISION_TOKENS:-256}"
VISION_STRIDE="${VISION_STRIDE:-1}"
ADD_IMAGE_TOKEN="${ADD_IMAGE_TOKEN:-1}"
USE_FP16_LUT="${USE_FP16_LUT:-0}"
BIN_DIR="${BIN_DIR:-}"
EMBEDDING_DTYPE="${EMBEDDING_DTYPE:-}"
EMBEDDING_SCALE="${EMBEDDING_SCALE:-1}"
EMBEDDING_OFFSET="${EMBEDDING_OFFSET:-0}"
FORCE_TOKEN_PREFILL="${FORCE_TOKEN_PREFILL:-0}"
DEVICE_DIR="${DEVICE_DIR:-/data/local/tmp/fastvlm}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${ROOT_DIR}}"
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-${ROOT_DIR}/fastvlm_genie_npu.json}"
SKIP_PUSH="${SKIP_PUSH:-0}"

if [[ -z "${IMAGE_PATH}" || -z "${PROMPT}" ]]; then
  echo "Usage: $0 /path/to/image \"Describe this image\""
  exit 1
fi

LUT_PATH="${LUT_PATH:-}"
if [[ -n "${LUT_PATH}" ]]; then
  if [[ "${LUT_PATH}" != /* ]]; then
    LUT_PATH="${ROOT_DIR}/${LUT_PATH}"
  fi
elif [[ "${USE_FP16_LUT}" != "0" && -f "${ROOT_DIR}/embedding_fp16.bin" ]]; then
  LUT_PATH="${ROOT_DIR}/embedding_fp16.bin"
elif [[ -f "${ROOT_DIR}/embedding.bin" ]]; then
  LUT_PATH="${ROOT_DIR}/embedding.bin"
elif [[ -f "${ROOT_DIR}/embedding_fp16.bin" ]]; then
  LUT_PATH="${ROOT_DIR}/embedding_fp16.bin"
else
  echo "Missing embedding LUT (embedding.bin or embedding_fp16.bin)."
  exit 1
fi

BIN_DIR_ABS="${BIN_DIR}"
if [[ -n "${BIN_DIR_ABS}" && "${BIN_DIR_ABS}" != /* ]]; then
  BIN_DIR_ABS="${ROOT_DIR}/${BIN_DIR_ABS}"
fi

PYTHON="${PYTHON:-${ROOT_DIR}/venv/bin/python3}"
if [[ ! -x "${PYTHON}" ]]; then
  echo "Python not found at ${PYTHON}. Activate venv or set PYTHON."
  exit 1
fi

echo "==> Preparing inputs"
"${PYTHON}" "${ROOT_DIR}/host/prepare_vision_input.py" \
  --image "${IMAGE_PATH}" \
  --out-dir "${ROOT_DIR}/host_inputs/vision" \
  --device-dir "/data/local/tmp/fastvlm"
TEXT_ARGS=()
if [[ "${ADD_IMAGE_TOKEN}" != "0" ]]; then
  TEXT_ARGS+=(--add-image-token)
fi
"${PYTHON}" "${ROOT_DIR}/host/prepare_text_input.py" \
  --prompt "${PROMPT}" \
  --tokenizer-dir "${TOKENIZER_DIR}" \
  --out-dir "${ROOT_DIR}/host_inputs/text" \
  --device-dir "/data/local/tmp/fastvlm" \
  --format "${PROMPT_FORMAT}" \
  "${TEXT_ARGS[@]}"

if [[ "${SKIP_PUSH}" == "0" ]]; then
  echo "==> Pushing runtime + models"
  bash "${ROOT_DIR}/device/push_runtime.sh"
  if [[ -n "${BIN_DIR_ABS}" ]]; then
    BIN_DIR="${BIN_DIR_ABS}" LUT_FILE="${LUT_PATH}" bash "${ROOT_DIR}/device/push_model.sh"
  else
    LUT_FILE="${LUT_PATH}" bash "${ROOT_DIR}/device/push_model.sh"
  fi
  bash "${ROOT_DIR}/device/push_encoders.sh"
else
  echo "==> SKIP_PUSH=1 (using existing runtime/models on device)"
fi

echo "==> Pushing inputs"
bash "${ROOT_DIR}/device/push_inputs.sh"

echo "==> Running encoders on device"
bash "${ROOT_DIR}/device/run_vision_encoder.sh"
bash "${ROOT_DIR}/device/run_text_embedder.sh"

echo "==> Pulling encoder outputs"
bash "${ROOT_DIR}/device/pull_outputs.sh"

echo "==> Combining embeddings (truncates to fit KV cache)"
BOS_ARGS=()
if [[ "${ADD_BOS}" != "0" ]]; then
  BOS_ARGS=(--add-bos --bos-token "${BOS_TOKEN}")
fi
"${PYTHON}" "${ROOT_DIR}/host/combine_embeddings.py" \
  --vision-raw "${ROOT_DIR}/host_outputs/vision/vision_output.raw" \
  --text-raw "${ROOT_DIR}/host_outputs/text/text_output.raw" \
  --text-meta "${ROOT_DIR}/host_inputs/text/meta.json" \
  --out-dir "${ROOT_DIR}/host_outputs/combined" \
  --max-prompt-tokens "${MAX_PROMPT_TOKENS}" \
  --vision-tokens "${VISION_TOKENS}" \
  --vision-stride "${VISION_STRIDE}" \
  --lut "${LUT_PATH}" \
  --order "${COMBINE_ORDER}" \
  "${BOS_ARGS[@]}"

echo "==> Preparing runtime GENIE config (max-num-tokens based on prompt length)"
RUNTIME_CONFIG="${ROOT_DIR}/host_outputs/combined/fastvlm_genie_npu_runtime.json"
"${PYTHON}" - <<PY
import json
from pathlib import Path

root = Path("${ROOT_DIR}")
meta = json.loads((root / "host_outputs" / "combined" / "meta.json").read_text())
cfg = json.loads(Path("${CONFIG_TEMPLATE}").read_text())

ctx_size = int(cfg["dialog"]["context"]["size"])
total_tokens = int(meta.get("total_tokens", 0))
max_gen = max(ctx_size - total_tokens, 0)

cfg["dialog"]["max-num-tokens"] = int(max_gen)

out = Path("${RUNTIME_CONFIG}")
out.write_text(json.dumps(cfg, indent=4) + "\\n")
print(f"Wrote {out} (max-num-tokens={max_gen}, total_tokens={total_tokens}, ctx_size={ctx_size})")
PY

echo "==> Pushing prefill tokens"
PUSH_TOKEN_FILES="${PUSH_TOKEN_FILES:-}"
if [[ -z "${PUSH_TOKEN_FILES}" ]]; then
  if [[ "${FORCE_TOKEN_PREFILL}" != "0" ]]; then
    PUSH_TOKEN_FILES=1
  else
    PUSH_TOKEN_FILES=0
  fi
fi
PUSH_TOKEN_FILES="${PUSH_TOKEN_FILES}" bash "${ROOT_DIR}/device/push_prefill_tokens.sh"

echo "==> Prefill + generate on device"
DEVICE_LUT="${DEVICE_DIR}/$(basename "${LUT_PATH}")"
adb push "${RUNTIME_CONFIG}" "${DEVICE_DIR}/fastvlm_genie_npu_runtime.json" >/dev/null
LUT="${DEVICE_LUT}" \
CONFIG="${DEVICE_DIR}/fastvlm_genie_npu_runtime.json" \
EMBEDDING_FILE_DTYPE="${EMBEDDING_DTYPE}" \
EMBEDDING_FILE_SCALE="${EMBEDDING_SCALE}" \
EMBEDDING_FILE_OFFSET="${EMBEDDING_OFFSET}" \
LUT_DTYPE="${EMBEDDING_DTYPE}" \
LUT_SCALE="${EMBEDDING_SCALE}" \
LUT_OFFSET="${EMBEDDING_OFFSET}" \
FORCE_TOKEN_PREFILL="${FORCE_TOKEN_PREFILL}" \
bash "${ROOT_DIR}/device/prefill_and_generate.sh"

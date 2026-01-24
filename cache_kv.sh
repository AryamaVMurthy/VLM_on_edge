#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE_PATH="${1:-}"
SYSTEM_PROMPT="${2-${SYSTEM_PROMPT-}}"
ALLOW_EMPTY_SYSTEM_PROMPT="${ALLOW_EMPTY_SYSTEM_PROMPT:-0}"
if [[ -z "${SYSTEM_PROMPT}" && "${ALLOW_EMPTY_SYSTEM_PROMPT}" == "0" ]]; then
  SYSTEM_PROMPT="You are a helpful assistant."
fi
ADD_IMAGE_TOKEN="${ADD_IMAGE_TOKEN:-1}"
USER_PREFIX_DEFAULT="<image>\n"
if [[ "${ADD_IMAGE_TOKEN}" == "0" ]]; then
  USER_PREFIX_DEFAULT=""
fi
USER_PREFIX="${USER_PREFIX:-${USER_PREFIX_DEFAULT}}"

if [[ -z "${IMAGE_PATH}" ]]; then
  echo "Usage: $0 /path/to/image [\"system prompt with optional <image>\"]"
  exit 1
fi

PROMPT_FORMAT="${PROMPT_FORMAT:-raw}"
INCLUDE_USER_START="${INCLUDE_USER_START:-1}"
ALLOW_EMPTY_PROMPT="${ALLOW_EMPTY_PROMPT:-0}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-512}"
VISION_TOKENS="${VISION_TOKENS:-256}"
VISION_STRIDE="${VISION_STRIDE:-1}"
ADD_BOS="${ADD_BOS:-0}"
BOS_TOKEN="${BOS_TOKEN:-151644}"
USE_FP16_LUT="${USE_FP16_LUT:-1}"
LUT_PATH="${LUT_PATH:-}"
DEVICE_DIR="${DEVICE_DIR:-/data/local/tmp/fastvlm}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${ROOT_DIR}}"
BIN_DIR="${BIN_DIR:-}"
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-${ROOT_DIR}/fastvlm_genie_npu.json}"
CACHE_STATE="${CACHE_STATE:-${DEVICE_DIR}/state/static_dialog_state}"
CACHE_META="${CACHE_META:-${ROOT_DIR}/host_outputs/cache_state.json}"
CACHE_MODE="${CACHE_MODE:-combined}"
EMBEDDING_QUERY_OUTPUT_TYPE="${EMBEDDING_QUERY_OUTPUT_TYPE:-token}"

PYTHON="${PYTHON:-${ROOT_DIR}/venv/bin/python3}"
if [[ ! -x "${PYTHON}" ]]; then
  echo "Python not found at ${PYTHON}. Activate venv or set PYTHON."
  exit 1
fi

LUT_ABS="${LUT_PATH}"
if [[ -n "${LUT_ABS}" && "${LUT_ABS}" != /* ]]; then
  LUT_ABS="${ROOT_DIR}/${LUT_ABS}"
fi

if [[ -z "${LUT_ABS}" ]]; then
  if [[ "${USE_FP16_LUT}" != "0" && -f "${ROOT_DIR}/embedding_fp16.bin" ]]; then
    LUT_ABS="${ROOT_DIR}/embedding_fp16.bin"
  elif [[ -f "${ROOT_DIR}/embedding.bin" ]]; then
    LUT_ABS="${ROOT_DIR}/embedding.bin"
  elif [[ -f "${ROOT_DIR}/embedding_fp16.bin" ]]; then
    LUT_ABS="${ROOT_DIR}/embedding_fp16.bin"
  else
    echo "Missing embedding LUT (embedding.bin or embedding_fp16.bin)."
    exit 1
  fi
fi

BIN_DIR_ABS="${BIN_DIR}"
if [[ -n "${BIN_DIR_ABS}" && "${BIN_DIR_ABS}" != /* ]]; then
  BIN_DIR_ABS="${ROOT_DIR}/${BIN_DIR_ABS}"
fi

echo "==> Preparing inputs (cache prefix)"
"${PYTHON}" - <<PY
from pathlib import Path

system = "${SYSTEM_PROMPT}"
user_prefix = "${USER_PREFIX}"
add_image_token = "${ADD_IMAGE_TOKEN}" != "0"

parts = []
if system:
    parts.append(f"<|im_start|>system\\n{system}<|im_end|>\\n")
if "${INCLUDE_USER_START}" != "0":
    parts.append("<|im_start|>user\\n")
if add_image_token and "<image>" not in user_prefix:
    user_prefix = "<image>\\n" + user_prefix
parts.append(user_prefix)
prefix = "".join(parts)

out_dir = Path("${ROOT_DIR}") / "host_outputs" / "append"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "prefix_prompt.txt").write_text(prefix)
print(prefix)
PY

"${PYTHON}" "${ROOT_DIR}/host/prepare_vision_input.py" \
  --image "${IMAGE_PATH}" \
  --out-dir "${ROOT_DIR}/host_inputs/vision" \
  --device-dir "${DEVICE_DIR}"

"${PYTHON}" "${ROOT_DIR}/host/prepare_text_input.py" \
  --prompt "$(cat "${ROOT_DIR}/host_outputs/append/prefix_prompt.txt")" \
  --tokenizer-dir "${TOKENIZER_DIR}" \
  --out-dir "${ROOT_DIR}/host_inputs/text" \
  --device-dir "${DEVICE_DIR}" \
  --format "${PROMPT_FORMAT}" \
  $( [[ "${ALLOW_EMPTY_PROMPT}" != "0" ]] && echo "--allow-empty-prompt" )

echo "==> Pushing runtime + models"
bash "${ROOT_DIR}/device/push_runtime.sh"
if [[ -n "${BIN_DIR_ABS}" ]]; then
  BIN_DIR="${BIN_DIR_ABS}" LUT_FILE="${LUT_ABS}" bash "${ROOT_DIR}/device/push_model.sh"
else
  LUT_FILE="${LUT_ABS}" bash "${ROOT_DIR}/device/push_model.sh"
fi
bash "${ROOT_DIR}/device/push_encoders.sh"

echo "==> Pushing inputs"
bash "${ROOT_DIR}/device/push_inputs.sh"

echo "==> Running encoders on device"
bash "${ROOT_DIR}/device/run_vision_encoder.sh"
bash "${ROOT_DIR}/device/run_text_embedder.sh"

echo "==> Pulling encoder outputs"
bash "${ROOT_DIR}/device/pull_outputs.sh"

echo "==> Combining embeddings (cache prefix)"
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
  --lut "${LUT_ABS}" \
  --order "vision-text" \
  "${BOS_ARGS[@]}"

echo "==> Preparing runtime GENIE config (max-num-tokens=0)"
RUNTIME_CONFIG="${ROOT_DIR}/host_outputs/combined/fastvlm_genie_npu_runtime.json"
"${PYTHON}" - <<PY
import json
from pathlib import Path

root = Path("${ROOT_DIR}")
meta = json.loads((root / "host_outputs" / "combined" / "meta.json").read_text())
cfg = json.loads(Path("${CONFIG_TEMPLATE}").read_text())

ctx_size = int(cfg["dialog"]["context"]["size"])
total_tokens = int(meta.get("total_tokens", 0))
cfg["dialog"]["max-num-tokens"] = 0

out = Path("${RUNTIME_CONFIG}")
out.write_text(json.dumps(cfg, indent=4) + "\\n")
cache_meta = {
    "cached_tokens": total_tokens,
    "ctx_size": ctx_size,
    "state_path": "${CACHE_STATE}",
}
Path("${CACHE_META}").write_text(json.dumps(cache_meta, indent=2) + "\\n")
print(f"Wrote {out} (max-num-tokens=0, total_tokens={total_tokens}, ctx_size={ctx_size})")
print(f"Wrote {Path('${CACHE_META}')} (cached_tokens={total_tokens})")
PY

echo "==> Pushing prefill tokens"
PUSH_TOKEN_FILES="${PUSH_TOKEN_FILES:-}"
if [[ -z "${PUSH_TOKEN_FILES}" ]]; then
  if [[ "${CACHE_MODE}" == "token" ]]; then
    PUSH_TOKEN_FILES=1
  else
    PUSH_TOKEN_FILES=0
  fi
fi
PUSH_TOKEN_FILES="${PUSH_TOKEN_FILES}" bash "${ROOT_DIR}/device/push_prefill_tokens.sh"

echo "==> Caching dialog state to ${CACHE_STATE}"
adb shell "rm -rf ${CACHE_STATE}"
adb shell "mkdir -p ${CACHE_STATE}"
DEVICE_LUT="${DEVICE_DIR}/$(basename "${LUT_ABS}")"
adb push "${RUNTIME_CONFIG}" "${DEVICE_DIR}/fastvlm_genie_npu_runtime.json" >/dev/null
if [[ -n "${PROFILE_OUT:-}" ]]; then
  adb shell "rm -f ${PROFILE_OUT}"
fi
FORCE_PREFILL=0
if [[ "${CACHE_MODE}" == "token" ]]; then
  FORCE_PREFILL=1
fi
LUT="${DEVICE_LUT}" \
CONFIG="${DEVICE_DIR}/fastvlm_genie_npu_runtime.json" \
SAVE_STATE="${CACHE_STATE}" \
PREFILL_ONLY=1 \
EMBEDDING_QUERY_OUTPUT_TYPE="${EMBEDDING_QUERY_OUTPUT_TYPE}" \
FORCE_TOKEN_PREFILL="${FORCE_PREFILL}" \
bash "${ROOT_DIR}/device/prefill_and_generate.sh"

if ! adb shell "test -e ${CACHE_STATE}" >/dev/null 2>&1; then
  echo "State not found after combined prefill; retrying with token prefill."
  LUT="${DEVICE_LUT}" \
  CONFIG="${DEVICE_DIR}/fastvlm_genie_npu_runtime.json" \
  SAVE_STATE="${CACHE_STATE}" \
  PREFILL_ONLY=1 \
  EMBEDDING_QUERY_OUTPUT_TYPE=token \
  FORCE_TOKEN_PREFILL=1 \
  bash "${ROOT_DIR}/device/prefill_and_generate.sh"
fi

echo "==> Cache complete"
adb shell "test -e ${CACHE_STATE}" && echo "Saved state: ${CACHE_STATE}" || echo "Warning: state path not found on device."

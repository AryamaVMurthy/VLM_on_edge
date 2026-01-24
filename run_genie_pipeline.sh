#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE_PATH="${1:-}"
PROMPT="${2:-Describe the image in 2-3 sentences.}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-You are a helpful assistant.}"

if [[ -z "${IMAGE_PATH}" ]]; then
  echo "Usage: $0 /path/to/image [\"prompt\"]"
  exit 1
fi

DEVICE_DIR="${DEVICE_DIR:-/data/local/tmp/fastvlm}"
QCOM_ROOT="${QCOM_ROOT:-}"
if [[ -z "${QCOM_ROOT}" ]]; then
  if [[ -d "${ROOT_DIR}/qcom_ai_stack_2.41" ]]; then
    QCOM_ROOT="${ROOT_DIR}/qcom_ai_stack_2.41"
  else
    QCOM_ROOT="${ROOT_DIR}/../../../qcom_ai_stack"
  fi
fi

PYTHON="${PYTHON:-${ROOT_DIR}/venv/bin/python3}"
VISION_INPUT_FORMAT="${VISION_INPUT_FORMAT:-}"
VISION_INPUT_MODE="${VISION_INPUT_MODE:-raw}"
VISION_BIN="${VISION_BIN:-}"
TEXT_BIN="${TEXT_BIN:-}"

HWC_BIN="$(ls -1 ${ROOT_DIR}/qaihub_bins/fastvlm_vision_encoder_hwc_*.bin 2>/dev/null | head -n 1 || true)"
FP16IN_FP32OUT_BIN="$(ls -1 ${ROOT_DIR}/qaihub_bins/fastvlm_vision_encoder_fp16in_fp32out_flat_hwc_*.bin 2>/dev/null | head -n 1 || true)"
U8_FLAT_BIN="$(ls -1 ${ROOT_DIR}/qaihub_bins/fastvlm_vision_encoder_u8_flat_hwc*.bin 2>/dev/null | head -n 1 || true)"
FLAT_BIN="$(ls -1 ${ROOT_DIR}/qaihub_bins/fastvlm_vision_encoder_*flat_hwc*.bin 2>/dev/null | head -n 1 || true)"
if [[ -z "${VISION_BIN}" && -n "${FP16IN_FP32OUT_BIN}" ]]; then
  VISION_BIN="${FP16IN_FP32OUT_BIN}"
  if [[ -z "${VISION_INPUT_FORMAT}" ]]; then
    VISION_INPUT_FORMAT="float16_hwc_flat"
  fi
elif [[ -z "${VISION_BIN}" && -n "${U8_FLAT_BIN}" ]]; then
  VISION_BIN="${U8_FLAT_BIN}"
  if [[ -z "${VISION_INPUT_FORMAT}" ]]; then
    VISION_INPUT_FORMAT="uint8_hwc_flat"
  fi
elif [[ -z "${VISION_BIN}" && -n "${FLAT_BIN}" ]]; then
  VISION_BIN="${FLAT_BIN}"
  if [[ -z "${VISION_INPUT_FORMAT}" ]]; then
    if [[ "${VISION_BIN}" == *u8* ]]; then
      VISION_INPUT_FORMAT="uint8_hwc_flat"
    elif [[ "${VISION_BIN}" == *fp16* ]]; then
      VISION_INPUT_FORMAT="float16_hwc_flat"
    else
      VISION_INPUT_FORMAT="float32_hwc_flat"
    fi
  fi
elif [[ -z "${VISION_BIN}" && -n "${HWC_BIN}" ]]; then
  VISION_BIN="${HWC_BIN}"
  if [[ -z "${VISION_INPUT_FORMAT}" ]]; then
    if [[ "${VISION_BIN}" == *u8* ]]; then
      VISION_INPUT_FORMAT="uint8_hwc"
    elif [[ "${VISION_BIN}" == *fp16* ]]; then
      VISION_INPUT_FORMAT="float16_hwc_nobatch"
    else
      VISION_INPUT_FORMAT="float32_hwc_nobatch"
    fi
  fi
fi
if [[ -z "${VISION_INPUT_FORMAT}" && -n "${VISION_BIN}" ]]; then
  if [[ "${VISION_BIN}" == *flat_hwc* ]]; then
    if [[ "${VISION_BIN}" == *u8* ]]; then
      VISION_INPUT_FORMAT="uint8_hwc_flat"
    elif [[ "${VISION_BIN}" == *fp16* ]]; then
      VISION_INPUT_FORMAT="float16_hwc_flat"
    else
      VISION_INPUT_FORMAT="float32_hwc_flat"
    fi
  elif [[ "${VISION_BIN}" == *hwc* ]]; then
    if [[ "${VISION_BIN}" == *u8* ]]; then
      VISION_INPUT_FORMAT="uint8_hwc"
    elif [[ "${VISION_BIN}" == *fp16* ]]; then
      VISION_INPUT_FORMAT="float16_hwc_nobatch"
    else
      VISION_INPUT_FORMAT="float32_hwc_nobatch"
    fi
  fi
fi
if [[ -z "${VISION_INPUT_FORMAT}" ]]; then
  VISION_INPUT_FORMAT="float32_nchw"
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "Python not found at ${PYTHON}. Activate venv or set PYTHON."
  exit 1
fi

# Prepare vision input
if [[ "${VISION_INPUT_MODE}" == "image" ]]; then
  PREP_FORMAT="image_png"
  INPUT_FILE_NAME="image.png"
else
  PREP_FORMAT="${VISION_INPUT_FORMAT}"
  INPUT_FILE_NAME="pixel_values.raw"
fi

"${PYTHON}" "${ROOT_DIR}/host/prepare_vision_input.py" \
  --image "${IMAGE_PATH}" \
  --out-dir "${ROOT_DIR}/host_inputs/vision" \
  --device-dir "${DEVICE_DIR}" \
  --format "${PREP_FORMAT}"

# Push runtime + genie-app
QCOM_ROOT="${QCOM_ROOT}" DEVICE_DIR="${DEVICE_DIR}" bash "${ROOT_DIR}/device/push_runtime.sh"
adb push "${QCOM_ROOT}/bin/aarch64-android/genie-app" "${DEVICE_DIR}/" >/dev/null

# Push model + encoders (decoder + LUT + tokenizer + vision encoder)
DEVICE_DIR="${DEVICE_DIR}" bash "${ROOT_DIR}/device/push_model.sh"
ALT_MODEL_DIR="${ROOT_DIR}/../../VLM_run_experiments_QIDK/fastvlm_yashas/models/fastvlm-genie"
if [[ -z "${VISION_BIN}" && -f "${ALT_MODEL_DIR}/vision_encoder_fp16.bin" ]]; then
  VISION_BIN="${ALT_MODEL_DIR}/vision_encoder_fp16.bin"
fi
if [[ -z "${TEXT_BIN}" && -f "${ALT_MODEL_DIR}/text_embedder_fp16.bin" ]]; then
  TEXT_BIN="${ALT_MODEL_DIR}/text_embedder_fp16.bin"
fi
if [[ -n "${VISION_BIN}" || -n "${TEXT_BIN}" ]]; then
  VISION_BIN="${VISION_BIN}" TEXT_BIN="${TEXT_BIN}" DEVICE_DIR="${DEVICE_DIR}" bash "${ROOT_DIR}/device/push_encoders.sh"
else
  DEVICE_DIR="${DEVICE_DIR}" bash "${ROOT_DIR}/device/push_encoders.sh"
fi

# Push image input
adb shell "mkdir -p ${DEVICE_DIR}/inputs/vision"
adb push "${ROOT_DIR}/host_inputs/vision/${INPUT_FILE_NAME}" "${DEVICE_DIR}/inputs/vision/" >/dev/null

# Push pipeline configs (allow vision-param override)
VISION_PARAM_HEIGHT="${VISION_PARAM_HEIGHT:-1024}"
VISION_PARAM_WIDTH="${VISION_PARAM_WIDTH:-1024}"
HOST_PIPE_DIR="${ROOT_DIR}/host_outputs/pipeline"
mkdir -p "${HOST_PIPE_DIR}"

IMAGE_CFG_RUNTIME="${HOST_PIPE_DIR}/fastvlm_image_encoder.runtime.json"
TEXT_CFG_RUNTIME="${HOST_PIPE_DIR}/fastvlm_text_encoder.runtime.json"
GEN_CFG_RUNTIME="${HOST_PIPE_DIR}/fastvlm_text_generator.runtime.json"

"${PYTHON}" - <<PY
import json
from pathlib import Path

root = Path("${ROOT_DIR}")
height = int("${VISION_PARAM_HEIGHT}")
width = int("${VISION_PARAM_WIDTH}")

img_cfg = json.loads((root / "genie_pipeline" / "fastvlm_image_encoder.json").read_text())
img_cfg["image-encoder"]["engine"]["model"]["vision-param"]["height"] = height
img_cfg["image-encoder"]["engine"]["model"]["vision-param"]["width"] = width
Path("${IMAGE_CFG_RUNTIME}").write_text(json.dumps(img_cfg, indent=2) + "\\n")

text_cfg = (root / "genie_pipeline" / "fastvlm_text_encoder.json").read_text()
Path("${TEXT_CFG_RUNTIME}").write_text(text_cfg)

gen_cfg = (root / "genie_pipeline" / "fastvlm_text_generator.json").read_text()
Path("${GEN_CFG_RUNTIME}").write_text(gen_cfg)
PY

adb shell "mkdir -p ${DEVICE_DIR}/pipeline"
adb push "${IMAGE_CFG_RUNTIME}" "${DEVICE_DIR}/pipeline/fastvlm_image_encoder.json" >/dev/null
adb push "${TEXT_CFG_RUNTIME}" "${DEVICE_DIR}/pipeline/fastvlm_text_encoder.json" >/dev/null
adb push "${GEN_CFG_RUNTIME}" "${DEVICE_DIR}/pipeline/fastvlm_text_generator.json" >/dev/null

# Build genie-app script (embed prompt with escaped newlines)
PIPE_SCRIPT="${HOST_PIPE_DIR}/fastvlm_pipeline.genie"
GENIE_LOG_LEVEL="${GENIE_LOG_LEVEL:-}"
GENIE_LOG_FILE="${GENIE_LOG_FILE:-${DEVICE_DIR}/genie_log.txt}"
PROMPT="${PROMPT}" SYSTEM_PROMPT="${SYSTEM_PROMPT}" DEVICE_DIR="${DEVICE_DIR}" INPUT_FILE_NAME="${INPUT_FILE_NAME}" PIPE_SCRIPT="${PIPE_SCRIPT}" GENIE_LOG_LEVEL="${GENIE_LOG_LEVEL}" GENIE_LOG_FILE="${GENIE_LOG_FILE}" \
"${PYTHON}" - <<'PY'
from pathlib import Path
import os

device_dir = os.environ["DEVICE_DIR"]
input_file = os.environ["INPUT_FILE_NAME"]
prompt = os.environ["PROMPT"]
system_prompt = os.environ["SYSTEM_PROMPT"]
script_path = Path(os.environ["PIPE_SCRIPT"])
log_level = os.environ.get("GENIE_LOG_LEVEL", "").strip()
log_file = os.environ.get("GENIE_LOG_FILE", "").strip()

text = (
    f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    f"<|im_start|>user\n{prompt}<|im_end|>\n"
    f"<|im_start|>assistant\n"
)

def esc(val: str) -> str:
    return val.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n")

text_esc = esc(text)

log_cmds = ""
if log_level:
    log_path = log_file or f"{device_dir}/genie_log.txt"
    log_cmds = (
        f"log create genieLog {log_level} {log_path}\n"
        "node config bind log imageEncoderConfig genieLog\n"
        "node config bind log textEncoderConfig genieLog\n"
        "node config bind log textGeneratorConfig genieLog\n"
    )

script = f"""version
pipeline config create pipelineConfig
pipeline create fastVLM pipelineConfig

node config create imageEncoderConfig {device_dir}/pipeline/fastvlm_image_encoder.json
node config create textEncoderConfig {device_dir}/pipeline/fastvlm_text_encoder.json
node config create textGeneratorConfig {device_dir}/pipeline/fastvlm_text_generator.json
{log_cmds}node create imageEncoder imageEncoderConfig

node create textEncoder textEncoderConfig

node create textGenerator textGeneratorConfig
node set textCallback textGenerator GENIE_NODE_TEXT_GENERATOR_TEXT_OUTPUT

pipeline add fastVLM imageEncoder
pipeline add fastVLM textEncoder
pipeline add fastVLM textGenerator

pipeline connect fastVLM imageEncoder GENIE_NODE_IMAGE_ENCODER_EMBEDDING_OUTPUT textGenerator GENIE_NODE_TEXT_GENERATOR_EMBEDDING_INPUT
pipeline connect fastVLM textEncoder GENIE_NODE_TEXT_ENCODER_EMBEDDING_OUTPUT textGenerator GENIE_NODE_TEXT_GENERATOR_EMBEDDING_INPUT

node set image imageEncoder GENIE_NODE_IMAGE_ENCODER_IMAGE_INPUT {device_dir}/inputs/vision/{input_file}
node set text textEncoder GENIE_NODE_TEXT_ENCODER_TEXT_INPUT "{text_esc}"

pipeline execute fastVLM

node free imageEncoder
node free textEncoder
node free textGenerator
pipeline free fastVLM
"""

script_path.write_text(script)
print(f"Wrote {script_path}")
PY

adb push "${PIPE_SCRIPT}" "${DEVICE_DIR}/pipeline/fastvlm_pipeline.genie" >/dev/null

# Run pipeline on device
adb shell "export ADSP_LIBRARY_PATH=${DEVICE_DIR} && \
  export LD_LIBRARY_PATH=${DEVICE_DIR}:/vendor/lib64 && \
  export PATH=${DEVICE_DIR}:\$PATH && \
  cd ${DEVICE_DIR} && \
  ./genie-app -s ${DEVICE_DIR}/pipeline/fastvlm_pipeline.genie" | tr -d '\r'

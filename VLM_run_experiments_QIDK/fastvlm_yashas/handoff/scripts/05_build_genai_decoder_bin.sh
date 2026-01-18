#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

source "${ROOT_DIR}/2.42.0.251225/qnn_env/bin/activate"
set +u
source "${ROOT_DIR}/2.42.0.251225/bin/envsetup.sh"
set -u

MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/models/apple--FastVLM-0.5B}"
CONFIG_JSON="${CONFIG_JSON:-${ROOT_DIR}/handoff/configs/genai_fastvlm_config.json}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/models/fastvlm-genie}"
QUANTIZE="${QUANTIZE:-}"
OUT_BIN="${OUT_BIN:-}"

mkdir -p "${OUT_DIR}"

if [[ -z "${OUT_BIN}" ]]; then
  case "${QUANTIZE}" in
    Z8|z8)
      OUT_BIN="${OUT_DIR}/fastvlm_model_z8.bin"
      ;;
    Q4|q4)
      OUT_BIN="${OUT_DIR}/fastvlm_model_q4.bin"
      ;;
    *)
      OUT_BIN="${OUT_DIR}/fastvlm_model.bin"
      ;;
  esac
fi

COMPOSE_ARGS=(
  --model "${MODEL_DIR}"
  --config_file "${CONFIG_JSON}"
  --outfile "${OUT_BIN}"
  --dump_lut
  --export_tokenizer_json
)
if [[ -n "${QUANTIZE}" ]]; then
  COMPOSE_ARGS+=(--quantize "${QUANTIZE}")
fi

pushd "${OUT_DIR}" >/dev/null
qnn-genai-transformer-composer "${COMPOSE_ARGS[@]}"
popd >/dev/null

echo "Done."

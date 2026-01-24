#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-${ROOT_DIR}/venv/bin/python3}"

usage() {
  cat <<'EOF'
FastVLM consolidated entrypoint

Usage:
  scripts/fastvlm.sh e2e <image_path> "<prompt>"
  scripts/fastvlm.sh cache <image_path> "<prompt>" ["system prompt"]
  scripts/fastvlm.sh export-decoder [export_decoder_fp16.py args...]
  scripts/fastvlm.sh compile [compile_final.py args...]
  scripts/fastvlm.sh quantize-compile [quantize_and_compile.py args...]

Examples:
  scripts/fastvlm.sh e2e /path/to/image.jpg "Describe the image."
  scripts/fastvlm.sh cache /path/to/image.jpg "Describe the image."
  scripts/fastvlm.sh export-decoder --model-dir /path/to/ckpt --output-dir .
  scripts/fastvlm.sh compile --onnx fastvlm_full_fp16_embedded_renamed.onnx
EOF
}

cmd="${1:-help}"
shift || true

case "${cmd}" in
  e2e)
    IMAGE_PATH="${1:-}"
    PROMPT="${2:-}"
    if [[ -z "${IMAGE_PATH}" || -z "${PROMPT}" ]]; then
      usage
      exit 1
    fi
    shift 2 || true
    exec "${ROOT_DIR}/run_e2e_vlm.sh" "${IMAGE_PATH}" "${PROMPT}" "$@"
    ;;
  cache)
    IMAGE_PATH="${1:-}"
    PROMPT="${2:-}"
    SYSTEM_PROMPT="${3:-${SYSTEM_PROMPT:-}}"
    if [[ -z "${IMAGE_PATH}" || -z "${PROMPT}" ]]; then
      usage
      exit 1
    fi
    shift 3 || true
    "${ROOT_DIR}/cache_kv.sh" "${IMAGE_PATH}" "${SYSTEM_PROMPT}"
    exec "${ROOT_DIR}/query_with_cache.sh" "${PROMPT}"
    ;;
  export-decoder)
    if [[ ! -x "${PYTHON}" ]]; then
      echo "Python not found at ${PYTHON}. Set PYTHON or create venv." >&2
      exit 1
    fi
    exec "${PYTHON}" "${ROOT_DIR}/export_decoder_fp16.py" "$@"
    ;;
  compile)
    if [[ ! -x "${PYTHON}" ]]; then
      echo "Python not found at ${PYTHON}. Set PYTHON or create venv." >&2
      exit 1
    fi
    exec "${PYTHON}" "${ROOT_DIR}/compile_final.py" "$@"
    ;;
  quantize-compile)
    if [[ ! -x "${PYTHON}" ]]; then
      echo "Python not found at ${PYTHON}. Set PYTHON or create venv." >&2
      exit 1
    fi
    exec "${PYTHON}" "${ROOT_DIR}/quantize_and_compile.py" "$@"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown command: ${cmd}" >&2
    usage
    exit 1
    ;;
esac

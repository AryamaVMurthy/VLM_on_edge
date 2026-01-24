#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROMPT="${1:-}"
if [[ -z "${PROMPT}" ]]; then
  echo "Usage: $0 \"your question about the image\""
  exit 1
fi

if [[ "${PROMPT}" == *"<image>"* ]]; then
  echo "Error: do not include <image> in the query when using cached KV (image already in cache)."
  exit 1
fi

PROMPT_FORMAT="${PROMPT_FORMAT:-raw}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${ROOT_DIR}}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-512}"
MAX_GEN_TOKENS="${MAX_GEN_TOKENS:-}"
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-${ROOT_DIR}/fastvlm_genie_npu.json}"
DEVICE_DIR="${DEVICE_DIR:-/data/local/tmp/fastvlm}"
CACHE_STATE="${CACHE_STATE:-${DEVICE_DIR}/state/static_dialog_state}"
CACHE_META="${CACHE_META:-${ROOT_DIR}/host_outputs/cache_state.json}"
SAVE_STATE="${SAVE_STATE:-${CACHE_STATE}}"
PROFILE_OUT="${PROFILE_OUT:-}"
EMBEDDING_QUERY_OUTPUT_TYPE="${EMBEDDING_QUERY_OUTPUT_TYPE:-text}"

PYTHON="${PYTHON:-${ROOT_DIR}/venv/bin/python3}"
if [[ ! -x "${PYTHON}" ]]; then
  echo "Python not found at ${PYTHON}. Activate venv or set PYTHON."
  exit 1
fi

if [[ ! -f "${CACHE_META}" ]]; then
  echo "Missing cache metadata: ${CACHE_META}. Run cache_kv.sh first."
  exit 1
fi
if ! adb shell "test -e ${CACHE_STATE}" >/dev/null 2>&1; then
  echo "Missing cache state on device: ${CACHE_STATE}. Run cache_kv.sh first."
  exit 1
fi

echo "==> Preparing text input (query)"
ROOT_DIR_ENV="${ROOT_DIR}" PROMPT_ENV="${PROMPT}" "${PYTHON}" - <<'PY'
from pathlib import Path
import os

suffix = os.environ.get("PROMPT_ENV", "")
suffix = suffix + "<|im_end|>\\n<|im_start|>assistant\\n"
root_dir = Path(os.environ["ROOT_DIR_ENV"])
out_dir = root_dir / "host_outputs" / "append"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "suffix_prompt.txt").write_text(suffix)
print(suffix)
PY

"${PYTHON}" "${ROOT_DIR}/host/prepare_text_input.py" \
  --prompt "$(cat "${ROOT_DIR}/host_outputs/append/suffix_prompt.txt")" \
  --tokenizer-dir "${TOKENIZER_DIR}" \
  --out-dir "${ROOT_DIR}/host_inputs/text" \
  --device-dir "${DEVICE_DIR}" \
  --format "${PROMPT_FORMAT}"

echo "==> Pushing text inputs"
adb shell "mkdir -p ${DEVICE_DIR}/inputs/text"
adb push "${ROOT_DIR}/host_inputs/text/input_ids.raw" "${DEVICE_DIR}/inputs/text/"
adb push "${ROOT_DIR}/host_inputs/text/input_list.txt" "${DEVICE_DIR}/inputs/text/"

echo "==> Running text embedder on device"
bash "${ROOT_DIR}/device/run_text_embedder.sh"

echo "==> Pulling text embeddings"
mkdir -p "${ROOT_DIR}/host_outputs/text"
adb pull "${DEVICE_DIR}/outputs/text/Result_0/output_0.raw" "${ROOT_DIR}/host_outputs/text/text_output.raw"

echo "==> Truncating embeddings to token_count"
TOKEN_COUNT="$("${PYTHON}" - <<PY
import json
from pathlib import Path
import numpy as np

root = Path("${ROOT_DIR}")
meta = json.loads((root / "host_inputs" / "text" / "meta.json").read_text())
cfg = json.loads(Path("${CONFIG_TEMPLATE}").read_text())
count = int(meta.get("token_count", 0))
raw_path = root / "host_outputs" / "text" / "text_output.raw"
out_path = root / "host_outputs" / "append" / "combined_embeddings.raw"
out_path.parent.mkdir(parents=True, exist_ok=True)

embedding_cfg = cfg.get("dialog", {}).get("embedding", {})
lut_path = str(embedding_cfg.get("lut-path", ""))
dtype_name = str(embedding_cfg.get("datatype", "native")).lower()
use_fp16 = ("fp16" in lut_path) or (dtype_name in ("float16", "fp16"))
out_dtype = np.float16 if use_fp16 else np.float32

# Text embedder output is float32 regardless of LUT dtype.
data = np.fromfile(raw_path, dtype=np.float32)
emb_dim = 896
total_tokens = data.size // emb_dim
data = data.reshape(total_tokens, emb_dim)
data[:count].astype(out_dtype, copy=False).tofile(out_path)
print(count)
PY
)"

echo "==> Preparing runtime GENIE config (max-num-tokens based on cache)"
RUNTIME_CONFIG="${ROOT_DIR}/host_outputs/append/fastvlm_genie_npu_runtime.json"
"${PYTHON}" - <<PY
import json
from pathlib import Path

root = Path("${ROOT_DIR}")
cache = json.loads(Path("${CACHE_META}").read_text())
cfg = json.loads(Path("${CONFIG_TEMPLATE}").read_text())
meta = json.loads((root / "host_inputs" / "text" / "meta.json").read_text())

cached_tokens = int(cache.get("cached_tokens", 0))
ctx_size = int(cache.get("ctx_size", cfg["dialog"]["context"]["size"]))
new_tokens = int(meta.get("token_count", 0))
total = cached_tokens + new_tokens
max_gen = max(ctx_size - total, 0)
cap = "${MAX_GEN_TOKENS}"
if cap:
    try:
        cap_val = int(cap)
    except ValueError:
        cap_val = 0
    if cap_val > 0:
        max_gen = min(max_gen, cap_val)

cfg["dialog"]["max-num-tokens"] = int(max_gen)
Path("${RUNTIME_CONFIG}").write_text(json.dumps(cfg, indent=4) + "\\n")
print(f"Wrote {Path('${RUNTIME_CONFIG}')} (max-num-tokens={max_gen}, total_tokens={total}, ctx_size={ctx_size})")
PY

echo "==> Pushing combined embeddings (query)"
adb shell "mkdir -p ${DEVICE_DIR}/inputs/prefill"
adb shell "rm -f ${DEVICE_DIR}/inputs/prefill/token_*.raw"
adb push "${ROOT_DIR}/host_outputs/append/combined_embeddings.raw" "${DEVICE_DIR}/inputs/prefill/combined_embeddings.raw"

echo "==> Prefill + generate from cached state"
if [[ -n "${PROFILE_OUT:-}" ]]; then
  adb shell "rm -f ${PROFILE_OUT}"
fi
adb push "${RUNTIME_CONFIG}" "${DEVICE_DIR}/fastvlm_genie_npu_runtime.json" >/dev/null
CONFIG="${DEVICE_DIR}/fastvlm_genie_npu_runtime.json" \
RESTORE_STATE="${CACHE_STATE}" \
SAVE_STATE="${SAVE_STATE}" \
FORCE_TOKEN_PREFILL="${FORCE_TOKEN_PREFILL:-0}" \
PROFILE_OUT="${PROFILE_OUT}" \
EMBEDDING_QUERY_OUTPUT_TYPE="${EMBEDDING_QUERY_OUTPUT_TYPE}" \
bash "${ROOT_DIR}/device/prefill_and_generate.sh"

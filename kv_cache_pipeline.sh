#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE_PATH="${1:-}"
PROMPT="${2:-Describe the image in 2-3 sentences.}"

if [[ -z "${IMAGE_PATH}" ]]; then
  echo "Usage: $0 /path/to/image [\"prompt\"]"
  exit 1
fi

QCOM_ROOT="${QCOM_ROOT:-}"
BIN_DIR="${BIN_DIR:-}"
if [[ -z "${QCOM_ROOT}" || ! -d "${QCOM_ROOT}" ]]; then
  echo "Set QCOM_ROOT to a valid qcom_ai_stack (e.g., qcom_ai_stack_2.41)."
  exit 1
fi
if [[ -z "${BIN_DIR}" || ! -d "${BIN_DIR}" ]]; then
  echo "Set BIN_DIR to a valid compiled bin directory."
  exit 1
fi

PROMPT_FORMAT="${PROMPT_FORMAT:-raw}"
VISION_TOKENS="${VISION_TOKENS:-256}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-512}"
MAX_GEN_TOKENS="${MAX_GEN_TOKENS:-64}"
USE_FP16_LUT="${USE_FP16_LUT:-0}"

DEVICE_DIR="${DEVICE_DIR:-/data/local/tmp/fastvlm}"
CACHE_PROFILE="${CACHE_PROFILE:-${DEVICE_DIR}/genie_profile_cache_pipeline.json}"
QUERY_PROFILE="${QUERY_PROFILE:-${DEVICE_DIR}/genie_profile_query_pipeline.json}"

HOST_CACHE_PROFILE="${HOST_CACHE_PROFILE:-${ROOT_DIR}/host_outputs/genie_profile_cache_pipeline.json}"
HOST_QUERY_PROFILE="${HOST_QUERY_PROFILE:-${ROOT_DIR}/host_outputs/genie_profile_query_pipeline.json}"

echo "==> Clearing device profiles"
adb shell "rm -f ${CACHE_PROFILE} ${QUERY_PROFILE}"

echo "==> Cache prefix (image + system)"
QCOM_ROOT="${QCOM_ROOT}" \
BIN_DIR="${BIN_DIR}" \
USE_FP16_LUT="${USE_FP16_LUT}" \
PROMPT_FORMAT="${PROMPT_FORMAT}" \
VISION_TOKENS="${VISION_TOKENS}" \
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS}" \
PROFILE_OUT="${CACHE_PROFILE}" \
"${ROOT_DIR}/cache_kv.sh" "${IMAGE_PATH}"

echo "==> Query with cache"
QCOM_ROOT="${QCOM_ROOT}" \
PROMPT_FORMAT="${PROMPT_FORMAT}" \
MAX_GEN_TOKENS="${MAX_GEN_TOKENS}" \
PROFILE_OUT="${QUERY_PROFILE}" \
"${ROOT_DIR}/query_with_cache.sh" "${PROMPT}"

echo "==> Pulling profiles"
adb pull "${CACHE_PROFILE}" "${HOST_CACHE_PROFILE}" >/dev/null
adb pull "${QUERY_PROFILE}" "${HOST_QUERY_PROFILE}" >/dev/null

echo "==> Summary (Genie dialog)"
python3 - <<PY
import json
from pathlib import Path

def summarize(path):
    data = json.loads(Path(path).read_text())
    ev_create = ev_query = None
    for comp in data.get("components", []):
        for e in comp.get("events", []):
            if e.get("type") == "GenieDialog_create":
                ev_create = e
            elif e.get("type") == "GenieDialog_query":
                ev_query = e
    def val(ev, key):
        v = ev.get(key) if ev else None
        if isinstance(v, dict):
            return v.get("value")
        return v
    create_us = val(ev_create, "duration") or 0
    ttfs_us = val(ev_query, "time-to-first-token") or 0
    prompt_tokens = val(ev_query, "num-prompt-tokens") or 0
    prompt_rate = val(ev_query, "prompt-processing-rate") or 0
    gen_rate = val(ev_query, "token-generation-rate") or 0
    return create_us, ttfs_us, prompt_tokens, prompt_rate, gen_rate

cache_create, cache_ttfs, cache_tokens, cache_rate, _ = summarize("${HOST_CACHE_PROFILE}")
query_create, query_ttfs, query_tokens, query_rate, query_gen_rate = summarize("${HOST_QUERY_PROFILE}")

print(f"Cache prefill: tokens={cache_tokens}, prompt_rate={cache_rate:.2f} tok/s, ttfs={cache_ttfs/1e6:.3f}s")
print(f"Query (cached): tokens={query_tokens}, prompt_rate={query_rate:.2f} tok/s, ttfs={query_ttfs/1e6:.3f}s, gen_rate={query_gen_rate:.2f} tok/s")
print(f"Query dialog create: {query_create/1e6:.3f}s")
print(f"Query total (create + ttfs): {(query_create + query_ttfs)/1e6:.3f}s")

ok = (query_ttfs/1e6) < 1.0
if ok:
    print("PASS: cached prefill TTFS < 1s")
else:
    print("FAIL: cached prefill TTFS >= 1s")
    raise SystemExit(1)
PY

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# User-configurable defaults
IMAGE_PATH="${IMAGE_PATH:-${ROOT_DIR}/models/fastvlm_test.jpg}"
PROMPT="${PROMPT:-Describe the image in detail.}"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/models/apple--FastVLM-0.5B}"
IMAGE_SIZE="${IMAGE_SIZE:-1024}"
CACHE_LEN="${CACHE_LEN:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-50}"

# QNN/QAIRT env
source "${ROOT_DIR}/2.42.0.251225/qnn_env/bin/activate"
set +u
source "${ROOT_DIR}/2.42.0.251225/bin/envsetup.sh"
set -u

VISION_CTX="${ROOT_DIR}/models/fastvlm-genie/local_context/vision_int8_qnnmodel/output/fastvlm_vision_encoder_int8_htp.bin"
EMBED_CTX="${ROOT_DIR}/models/fastvlm-qaihub-fp16-context/fastvlm_embed_tokens_fp32/job_jgdvx0n6g_optimized_bin_mn4yxgx0n.bin"
VISION_CTX_INFO="${ROOT_DIR}/models/fastvlm-genie/local_context/vision_int8_qnnmodel/vision_int8_ctx_info.json"
LUT_BIN="${ROOT_DIR}/models/fastvlm-genie/LUT.bin"
MODEL_BIN="${ROOT_DIR}/models/fastvlm-genie/fastvlm_model.bin"
TOKENIZER_JSON="${ROOT_DIR}/models/fastvlm-genie/tokenizer.json"
GENIE_BIN="${ROOT_DIR}/2.42.0.251225/bin/x86_64-linux-clang/genie-t2t-run"

OUT_BASE="${ROOT_DIR}/models/fastvlm-genie/run_outputs"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_BASE}/${RUN_ID}"
mkdir -p "${OUT_DIR}"

VISION_INPUT_RAW="${OUT_DIR}/pixel_values_u8.raw"
EMBED_INPUT_RAW="${OUT_DIR}/input_ids.raw"
TOKEN_META_JSON="${OUT_DIR}/tokens.json"

DEVICE_DIR="/data/local/tmp/fastvlm_e2e_cpu"
DEVICE_VISION_INPUT="${DEVICE_DIR}/inputs/vision/pixel_values_u8.raw"
DEVICE_EMBED_INPUT="${DEVICE_DIR}/inputs/embed/input_ids.raw"
DEVICE_VISION_LIST="${DEVICE_DIR}/inputs/vision/input_list.txt"
DEVICE_EMBED_LIST="${DEVICE_DIR}/inputs/embed/input_list.txt"

VISION_LIST_LOCAL="${OUT_DIR}/vision_input_list.txt"
EMBED_LIST_LOCAL="${OUT_DIR}/embed_input_list.txt"

VISION_OUT_LOCAL="${OUT_DIR}/vision_out"
EMBED_OUT_LOCAL="${OUT_DIR}/embed_out"

COMBINED_EMB="${OUT_DIR}/combined_embeddings_fp32.raw"
GENIE_CONFIG="${OUT_DIR}/fastvlm_genie_cpu_config.json"

if [[ ! -f "${VISION_CTX}" ]]; then
  echo "Missing vision int8 context: ${VISION_CTX}" >&2
  exit 1
fi
if [[ ! -f "${EMBED_CTX}" ]]; then
  echo "Missing embedder context: ${EMBED_CTX}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_BIN}" ]]; then
  echo "Missing Genie model bin: ${MODEL_BIN}" >&2
  exit 1
fi
if [[ ! -f "${LUT_BIN}" ]]; then
  echo "Missing LUT.bin: ${LUT_BIN}" >&2
  exit 1
fi
if [[ ! -f "${GENIE_BIN}" ]]; then
  echo "Missing genie-t2t-run: ${GENIE_BIN}" >&2
  exit 1
fi

echo "Preparing image + prompt inputs..."
IMAGE_PATH="${IMAGE_PATH}" PROMPT="${PROMPT}" MODEL_DIR="${MODEL_DIR}" \
IMAGE_SIZE="${IMAGE_SIZE}" CACHE_LEN="${CACHE_LEN}" \
VISION_INPUT_RAW="${VISION_INPUT_RAW}" EMBED_INPUT_RAW="${EMBED_INPUT_RAW}" \
TOKEN_META_JSON="${TOKEN_META_JSON}" python - <<'PY'
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from transformers import AutoTokenizer

image_path = os.environ["IMAGE_PATH"]
prompt = os.environ["PROMPT"]
model_dir = os.environ["MODEL_DIR"]
image_size = int(os.environ["IMAGE_SIZE"])
cache_len = int(os.environ["CACHE_LEN"])
vision_out = Path(os.environ["VISION_INPUT_RAW"])
embed_out = Path(os.environ["EMBED_INPUT_RAW"])
meta_out = Path(os.environ["TOKEN_META_JSON"])

img = Image.open(image_path).convert("RGB")
img = img.resize((image_size, image_size), Image.BICUBIC)
arr = np.asarray(img, dtype=np.uint8)
arr.tofile(vision_out)

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
pad_id = tokenizer.pad_token_id
if pad_id is None:
    pad_id = tokenizer.eos_token_id
token_ids = tokenizer(prompt, add_special_tokens=True).input_ids

input_ids = np.full((1, cache_len), pad_id, dtype=np.int32)
input_ids[0, : len(token_ids)] = np.array(token_ids, dtype=np.int32)
input_ids.tofile(embed_out)

meta = {
    "prompt": prompt,
    "token_ids": token_ids,
    "pad_id": int(pad_id),
    "cache_len": cache_len,
    "image_size": image_size,
}
meta_out.write_text(json.dumps(meta, indent=2))
PY

echo "${DEVICE_VISION_INPUT}" > "${VISION_LIST_LOCAL}"
echo "${DEVICE_EMBED_INPUT}" > "${EMBED_LIST_LOCAL}"

echo "Pushing inputs and binaries to device..."
adb shell "mkdir -p ${DEVICE_DIR}/inputs/vision ${DEVICE_DIR}/inputs/embed ${DEVICE_DIR}/outputs/vision ${DEVICE_DIR}/outputs/embed"
adb push "${ROOT_DIR}/2.42.0.251225/bin/aarch64-android/qnn-net-run" "${DEVICE_DIR}/"
adb push "${ROOT_DIR}/2.42.0.251225/lib/aarch64-android/libQnnHtp.so" "${DEVICE_DIR}/"
adb push "${ROOT_DIR}/2.42.0.251225/lib/aarch64-android/libQnnHtpPrepare.so" "${DEVICE_DIR}/"
adb push "${ROOT_DIR}/2.42.0.251225/lib/aarch64-android/libQnnHtpV79Stub.so" "${DEVICE_DIR}/"
adb push "${ROOT_DIR}/2.42.0.251225/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so" "${DEVICE_DIR}/"
adb push "${ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so" "${DEVICE_DIR}/"
adb push "${VISION_CTX}" "${DEVICE_DIR}/fastvlm_vision_encoder_int8_htp.bin"
adb push "${EMBED_CTX}" "${DEVICE_DIR}/fastvlm_embed_tokens_fp16_htp.bin"
adb push "${VISION_INPUT_RAW}" "${DEVICE_VISION_INPUT}"
adb push "${EMBED_INPUT_RAW}" "${DEVICE_EMBED_INPUT}"
adb push "${VISION_LIST_LOCAL}" "${DEVICE_VISION_LIST}"
adb push "${EMBED_LIST_LOCAL}" "${DEVICE_EMBED_LIST}"

RUN_ENV="export ADSP_LIBRARY_PATH=\"${DEVICE_DIR};/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp\" && \
export LD_LIBRARY_PATH=\"${DEVICE_DIR}:/vendor/dsp/cdsp:/vendor/lib64:\$LD_LIBRARY_PATH\" && \
cd ${DEVICE_DIR}"

echo "Running int8 vision encoder on device (HTP)..."
adb shell "${RUN_ENV} && ./qnn-net-run --backend libQnnHtp.so --retrieve_context fastvlm_vision_encoder_int8_htp.bin --input_list ${DEVICE_VISION_LIST} --use_native_input_files --use_native_output_files --output_dir outputs/vision --num_inferences 1"

echo "Running embedder on device (HTP)..."
adb shell "${RUN_ENV} && ./qnn-net-run --backend libQnnHtp.so --retrieve_context fastvlm_embed_tokens_fp16_htp.bin --input_list ${DEVICE_EMBED_LIST} --use_native_input_files --use_native_output_files --output_dir outputs/embed --num_inferences 1"

echo "Pulling device outputs..."
adb pull "${DEVICE_DIR}/outputs/vision" "${VISION_OUT_LOCAL}" >/dev/null
adb pull "${DEVICE_DIR}/outputs/embed" "${EMBED_OUT_LOCAL}" >/dev/null

if [[ ! -f "${VISION_CTX_INFO}" ]]; then
  echo "Generating vision context info..."
  qnn-context-binary-utility --context_binary "${VISION_CTX}" --json_file "${VISION_CTX_INFO}"
fi

echo "Combining vision + text embeddings..."
VISION_OUT_LOCAL="${VISION_OUT_LOCAL}" EMBED_OUT_LOCAL="${EMBED_OUT_LOCAL}" \
VISION_CTX_INFO="${VISION_CTX_INFO}" TOKEN_META_JSON="${TOKEN_META_JSON}" \
COMBINED_EMB="${COMBINED_EMB}" python - <<'PY'
import json
import os
from pathlib import Path

import numpy as np

vision_out_dir = Path(os.environ["VISION_OUT_LOCAL"])
embed_out_dir = Path(os.environ["EMBED_OUT_LOCAL"])
ctx_info = Path(os.environ["VISION_CTX_INFO"])
token_meta = Path(os.environ["TOKEN_META_JSON"])
combined_out = Path(os.environ["COMBINED_EMB"])

def find_raw(dir_path: Path) -> Path:
    if not dir_path.exists():
        raise FileNotFoundError(f"Missing output dir: {dir_path}")
    result_dirs = sorted([p for p in dir_path.iterdir() if p.is_dir() and p.name.startswith("Result_")])
    if not result_dirs:
        raise FileNotFoundError(f"No Result_* in {dir_path}")
    raw_files = list(result_dirs[-1].glob("*.raw"))
    if not raw_files:
        raise FileNotFoundError(f"No .raw outputs in {result_dirs[-1]}")
    # Prefer tensor-named file when available
    for name in ["image_features_native.raw", "output_0_native.raw", "output_0.raw"]:
        p = result_dirs[-1] / name
        if p.exists():
            return p
    return raw_files[0]

vision_raw = find_raw(vision_out_dir)
embed_raw = find_raw(embed_out_dir)

ctx = json.loads(ctx_info.read_text())
graph = ctx["info"]["graphs"][0]["info"]
out_info = graph["graphOutputs"][0]["info"]
scale = out_info["quantizeParams"]["scaleOffset"]["scale"]
offset = out_info["quantizeParams"]["scaleOffset"]["offset"]

vision_u8 = np.fromfile(vision_raw, dtype=np.uint8).reshape(1, 256, 896)
vision_f = (vision_u8.astype(np.float32) + float(offset)) * float(scale)

embed_data = np.fromfile(embed_raw, dtype=np.float32)
if embed_data.size != 1 * 512 * 896:
    embed_data = np.fromfile(embed_raw, dtype=np.float16).astype(np.float32)
embed = embed_data.reshape(1, 512, 896)

meta = json.loads(token_meta.read_text())
num_text = len(meta["token_ids"])
text_embeds = embed[:, :num_text, :]
combined = np.concatenate([vision_f, text_embeds], axis=1).astype(np.float32)

combined_out.parent.mkdir(parents=True, exist_ok=True)
combined.tofile(combined_out)
PY

echo "Preparing Genie CPU config..."
MODEL_BIN="${MODEL_BIN}" TOKENIZER_JSON="${TOKENIZER_JSON}" MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
GENIE_CONFIG="${GENIE_CONFIG}" python - <<'PY'
import json
import os

model_bin = os.environ["MODEL_BIN"]
tokenizer_json = os.environ["TOKENIZER_JSON"]
max_new = int(os.environ["MAX_NEW_TOKENS"])
out_path = os.environ["GENIE_CONFIG"]

config = {
    "dialog": {
        "version": 1,
        "type": "basic",
        "max-num-tokens": max_new,
        "embedding": {"version": 1, "size": 896, "datatype": "float32"},
        "context": {
            "version": 1,
            "size": 512,
            "n-vocab": 151936,
            "bos-token": 151643,
            "eos-token": -1,
            "pad-token": 151643,
        },
        "sampler": {
            "version": 1,
            "seed": 42,
            "temp": 0.8,
            "top-k": 40,
            "top-p": 0.95,
            "greedy": False,
        },
        "tokenizer": {"version": 1, "path": tokenizer_json},
        "engine": {
            "version": 1,
            "n-threads": 6,
            "backend": {
                "version": 1,
                "type": "QnnGenAiTransformer",
                "QnnGenAiTransformer": {
                    "version": 1,
                    "model-input": "embeddings",
                    "use-mmap": True,
                    "n-layer": 24,
                    "n-embd": 896,
                    "n-heads": 14,
                    "n-kv-heads": 2,
                },
            },
            "model": {"version": 1, "type": "library", "library": {"version": 1, "model-bin": model_bin}},
        },
    }
}

with open(out_path, "w") as f:
    json.dump(config, f, indent=2)
PY

echo "Decoding with Genie CPU backend..."
# Avoid pulling in non-x86_64 libs from the QAIRT env when running Genie on host.
export LD_LIBRARY_PATH="${ROOT_DIR}/2.42.0.251225/lib/x86_64-linux-clang"

GENIE_BIN="${GENIE_BIN}" GENIE_CONFIG="${GENIE_CONFIG}" COMBINED_EMB="${COMBINED_EMB}" \
LUT_BIN="${LUT_BIN}" MODEL_DIR="${MODEL_DIR}" python - <<'PY'
import os
import re
import subprocess
import sys
import time

from transformers import AutoTokenizer

genie_bin = os.environ["GENIE_BIN"]
config = os.environ["GENIE_CONFIG"]
emb_file = os.environ["COMBINED_EMB"]
lut_bin = os.environ["LUT_BIN"]
model_dir = os.environ["MODEL_DIR"]

cmd = [
    genie_bin,
    "--config",
    config,
    "--embedding_file",
    emb_file,
    "--embedding_table",
    lut_bin,
    "--embedding_query_output_type",
    "text",
]

proc = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=0,
)

start = time.time()
first_token_time = None
seen_begin = False
text_started = False
capture = ""

while True:
    ch = proc.stdout.read(1)
    if ch == "" and proc.poll() is not None:
        break
    if ch:
        sys.stdout.write(ch)
        sys.stdout.flush()
        capture += ch
        if not seen_begin:
            if "[BEGIN]:" in capture:
                seen_begin = True
                # Treat [BEGIN] emission as first-token time for robust timing.
                if first_token_time is None:
                    first_token_time = time.time()
                idx = capture.index("[BEGIN]:") + len("[BEGIN]:")
                trailing = capture[idx:]
                if trailing.strip():
                    text_started = True
        elif not text_started:
            if re.search(r"\\S", ch):
                first_token_time = time.time()
                text_started = True

end = time.time()

gen_text = ""
if "[BEGIN]:" in capture:
    gen_text = capture.split("[BEGIN]:", 1)[1]
elif "[COMPLETE]:" in capture:
    gen_text = capture.split("[COMPLETE]:", 1)[1]
if "[END]" in gen_text:
    gen_text = gen_text.split("[END]", 1)[0]
gen_text = gen_text.strip()

ttfb = None
tps = None
if first_token_time is not None:
    ttfb = first_token_time - start
    decode_time = max(end - first_token_time, 1e-9)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    token_count = len(tokenizer.encode(gen_text, add_special_tokens=False))
    tps = token_count / decode_time

print("\\n\\n=== Decode Metrics ===")
if ttfb is not None:
    print(f"TTFB (s): {ttfb:.3f}")
else:
    print("TTFB (s): n/a")
if tps is not None:
    print(f"Tokens/sec: {tps:.2f}")
else:
    print("Tokens/sec: n/a")
if gen_text:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    token_count = len(tokenizer.encode(gen_text, add_special_tokens=False))
    print(f"Generated tokens (approx): {token_count}")
    print("\\n=== Generated Text ===")
    print(gen_text)
PY

echo "Outputs saved to: ${OUT_DIR}"

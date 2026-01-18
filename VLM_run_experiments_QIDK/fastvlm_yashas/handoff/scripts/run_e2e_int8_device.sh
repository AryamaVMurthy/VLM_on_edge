#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# User-configurable defaults
IMAGE_PATH="${IMAGE_PATH:-${ROOT_DIR}/models/fastvlm_test.jpg}"
PROMPT="${PROMPT:-Describe the image in detail.}"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/models/apple--FastVLM-0.5B}"
IMAGE_SIZE="${IMAGE_SIZE:-1024}"
CACHE_LEN="${CACHE_LEN:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-0}"
PROFILE="1"
VISION_PRECISION="${VISION_PRECISION:-int8}"
SKIP_PUSH="${SKIP_PUSH:-0}"

# QNN/QAIRT env
source "${ROOT_DIR}/2.42.0.251225/qnn_env/bin/activate"
set +u
source "${ROOT_DIR}/2.42.0.251225/bin/envsetup.sh"
set -u

VISION_CTX_INT8="${ROOT_DIR}/models/fastvlm-genie/local_context/vision_int8_qnnmodel/output/fastvlm_vision_encoder_int8_htp.bin"
VISION_CTX_INFO_INT8="${ROOT_DIR}/models/fastvlm-genie/local_context/vision_int8_qnnmodel/vision_int8_ctx_info.json"
VISION_CTX_FP16="${ROOT_DIR}/models/fastvlm-qaihub-fp16-context/fastvlm_vision_encoder_fp32/job_jpevd2o85_optimized_bin_mm57jlj6n.bin"
VISION_CTX_INFO_FP16="${ROOT_DIR}/models/fastvlm-qaihub-fp16-context/vision_ctx_info_unified.json"
EMBED_CTX="${ROOT_DIR}/models/fastvlm-qaihub-fp16-context/fastvlm_embed_tokens_fp32/job_jgdvx0n6g_optimized_bin_mn4yxgx0n.bin"
LUT_BIN="${ROOT_DIR}/models/fastvlm-genie/LUT.bin"
MODEL_BIN="${MODEL_BIN:-${ROOT_DIR}/models/fastvlm-genie/fastvlm_model_z8.bin}"
TOKENIZER_JSON="${ROOT_DIR}/models/fastvlm-genie/tokenizer.json"

OUT_BASE="${ROOT_DIR}/models/fastvlm-genie/run_outputs"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_BASE}/${RUN_ID}"
mkdir -p "${OUT_DIR}"

VISION_INPUT_RAW="${OUT_DIR}/pixel_values.raw"
EMBED_INPUT_RAW="${OUT_DIR}/input_ids.raw"
TOKEN_META_JSON="${OUT_DIR}/tokens.json"

VISION_LIST_LOCAL="${OUT_DIR}/vision_input_list.txt"
EMBED_LIST_LOCAL="${OUT_DIR}/embed_input_list.txt"

VISION_OUT_LOCAL="${OUT_DIR}/vision_out"
EMBED_OUT_LOCAL="${OUT_DIR}/embed_out"

COMBINED_EMB="${OUT_DIR}/combined_embeddings_fp32.raw"
GENIE_CONFIG="${OUT_DIR}/fastvlm_genie_device_config.json"
PROFILE_OUT="${OUT_DIR}/profile.json"

DEVICE_DIR="/data/local/tmp/fastvlm_e2e_device"
DEVICE_VISION_INPUT="${DEVICE_DIR}/inputs/vision/pixel_values.raw"
DEVICE_EMBED_INPUT="${DEVICE_DIR}/inputs/embed/input_ids.raw"
DEVICE_VISION_LIST="${DEVICE_DIR}/inputs/vision/input_list.txt"
DEVICE_EMBED_LIST="${DEVICE_DIR}/inputs/embed/input_list.txt"
DEVICE_VISION_OUT="${DEVICE_DIR}/outputs/vision"
DEVICE_EMBED_OUT="${DEVICE_DIR}/outputs/embed"
DEVICE_COMBINED="${DEVICE_DIR}/combined_embeddings_fp32.raw"
DEVICE_CONFIG="${DEVICE_DIR}/fastvlm_genie_device_config.json"
DEVICE_MODEL_BIN="${DEVICE_DIR}/fastvlm_model.bin"
DEVICE_LUT_BIN="${DEVICE_DIR}/LUT.bin"
DEVICE_TOKENIZER_JSON="${DEVICE_DIR}/tokenizer.json"
DEVICE_PROFILE="${DEVICE_DIR}/profile_${RUN_ID}.json"
DEVICE_VISION_CTX="${DEVICE_DIR}/fastvlm_vision_encoder_${VISION_PRECISION}.bin"

VISION_CTX=""
VISION_CTX_INFO=""
case "${VISION_PRECISION}" in
  int8)
    VISION_CTX="${VISION_CTX_INT8}"
    VISION_CTX_INFO="${VISION_CTX_INFO_INT8}"
    ;;
  fp16)
    VISION_CTX="${VISION_CTX_FP16}"
    VISION_CTX_INFO="${VISION_CTX_INFO_FP16}"
    ;;
  *)
    echo "Unsupported VISION_PRECISION: ${VISION_PRECISION} (use int8 or fp16)" >&2
    exit 1
    ;;
esac

if [[ ! -f "${VISION_CTX}" ]]; then
  echo "Missing vision ${VISION_PRECISION} context: ${VISION_CTX}" >&2
  exit 1
fi
if [[ "${VISION_PRECISION}" == "fp16" && ! -f "${VISION_CTX_INFO}" ]]; then
  echo "Missing vision fp16 context info: ${VISION_CTX_INFO}" >&2
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

echo "Preparing image + prompt inputs..."
IMAGE_PATH="${IMAGE_PATH}" PROMPT="${PROMPT}" MODEL_DIR="${MODEL_DIR}" \
IMAGE_SIZE="${IMAGE_SIZE}" CACHE_LEN="${CACHE_LEN}" VISION_PRECISION="${VISION_PRECISION}" \
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
vision_precision = os.environ["VISION_PRECISION"]
vision_out = Path(os.environ["VISION_INPUT_RAW"])
embed_out = Path(os.environ["EMBED_INPUT_RAW"])
meta_out = Path(os.environ["TOKEN_META_JSON"])

img = Image.open(image_path).convert("RGB")
img = img.resize((image_size, image_size), Image.BICUBIC)
if vision_precision == "int8":
    arr = np.asarray(img, dtype=np.uint8)
    arr.tofile(vision_out)
else:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    arr.tofile(vision_out)

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
pad_id = tokenizer.pad_token_id
if pad_id is None:
    pad_id = tokenizer.eos_token_id

token_ids = None
if getattr(tokenizer, "apply_chat_template", None):
    messages = [{"role": "user", "content": prompt}]
    try:
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    except TypeError:
        token_ids = None
if token_ids is None:
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
    "used_chat_template": bool(getattr(tokenizer, "chat_template", None)),
}
meta_out.write_text(json.dumps(meta, indent=2))
PY

echo "${DEVICE_VISION_INPUT}" > "${VISION_LIST_LOCAL}"
echo "${DEVICE_EMBED_INPUT}" > "${EMBED_LIST_LOCAL}"

echo "Preparing device directories..."
adb shell "mkdir -p ${DEVICE_DIR}/inputs/vision ${DEVICE_DIR}/inputs/embed ${DEVICE_DIR}/outputs/vision ${DEVICE_DIR}/outputs/embed"

if [[ "${SKIP_PUSH}" == "0" ]]; then
  echo "Pushing binaries and libraries to device..."
  adb push "${ROOT_DIR}/2.42.0.251225/bin/aarch64-android/qnn-net-run" "${DEVICE_DIR}/"
  adb push "${ROOT_DIR}/2.42.0.251225/bin/aarch64-android/genie-t2t-run" "${DEVICE_DIR}/"
  adb push "${ROOT_DIR}/2.42.0.251225/lib/aarch64-android/libQnnHtp.so" "${DEVICE_DIR}/"
  adb push "${ROOT_DIR}/2.42.0.251225/lib/aarch64-android/libQnnHtpPrepare.so" "${DEVICE_DIR}/"
  adb push "${ROOT_DIR}/2.42.0.251225/lib/aarch64-android/libQnnHtpV79Stub.so" "${DEVICE_DIR}/"
  adb push "${ROOT_DIR}/2.42.0.251225/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so" "${DEVICE_DIR}/"
  adb push "${ROOT_DIR}/2.42.0.251225/lib/aarch64-android/libGenie.so" "${DEVICE_DIR}/"
  adb push "${ROOT_DIR}/2.42.0.251225/lib/aarch64-android/libQnnCpu.so" "${DEVICE_DIR}/"
  adb push "${ROOT_DIR}/2.42.0.251225/lib/aarch64-android/libQnnSystem.so" "${DEVICE_DIR}/"
  adb push "${ROOT_DIR}/2.42.0.251225/lib/aarch64-android/libQnnGenAiTransformerCpuOpPkg.so" "${DEVICE_DIR}/"
  adb push "${ROOT_DIR}/2.42.0.251225/lib/aarch64-android/libQnnGenAiTransformerModel.so" "${DEVICE_DIR}/"
  adb push "${ROOT_DIR}/2.42.0.251225/lib/aarch64-android/libQnnGenAiTransformer.so" "${DEVICE_DIR}/"
  adb push "${ANDROID_NDK_ROOT}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so" "${DEVICE_DIR}/"
  adb push "${VISION_CTX}" "${DEVICE_VISION_CTX}"
  adb push "${EMBED_CTX}" "${DEVICE_DIR}/fastvlm_embed_tokens_fp16_htp.bin"
  adb push "${MODEL_BIN}" "${DEVICE_MODEL_BIN}"
  adb push "${TOKENIZER_JSON}" "${DEVICE_TOKENIZER_JSON}"
  adb push "${LUT_BIN}" "${DEVICE_LUT_BIN}"
else
  echo "Skipping binary/library pushes (SKIP_PUSH=1)."
fi

echo "Pushing inputs to device..."
adb push "${VISION_INPUT_RAW}" "${DEVICE_VISION_INPUT}"
adb push "${EMBED_INPUT_RAW}" "${DEVICE_EMBED_INPUT}"
adb push "${VISION_LIST_LOCAL}" "${DEVICE_VISION_LIST}"
adb push "${EMBED_LIST_LOCAL}" "${DEVICE_EMBED_LIST}"

RUN_ENV="export ADSP_LIBRARY_PATH=\"${DEVICE_DIR};/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp\" && \
export LD_LIBRARY_PATH=\"${DEVICE_DIR}:/vendor/dsp/cdsp:/vendor/lib64\" && \
cd ${DEVICE_DIR}"

echo "Running vision encoder (${VISION_PRECISION}) on device (HTP)..."
adb shell "${RUN_ENV} && ./qnn-net-run --backend libQnnHtp.so --retrieve_context $(basename "${DEVICE_VISION_CTX}") --input_list ${DEVICE_VISION_LIST} --use_native_input_files --use_native_output_files --output_dir outputs/vision --num_inferences 1"

echo "Running embedder on device (HTP)..."
adb shell "${RUN_ENV} && ./qnn-net-run --backend libQnnHtp.so --retrieve_context fastvlm_embed_tokens_fp16_htp.bin --input_list ${DEVICE_EMBED_LIST} --use_native_input_files --use_native_output_files --output_dir outputs/embed --num_inferences 1"

echo "Pulling device outputs..."
adb pull "${DEVICE_VISION_OUT}" "${VISION_OUT_LOCAL}" >/dev/null
adb pull "${DEVICE_EMBED_OUT}" "${EMBED_OUT_LOCAL}" >/dev/null

if [[ "${VISION_PRECISION}" == "int8" && ! -f "${VISION_CTX_INFO}" ]]; then
  echo "Generating vision context info..."
  qnn-context-binary-utility --context_binary "${VISION_CTX}" --json_file "${VISION_CTX_INFO}"
fi

echo "Combining vision + text embeddings (host) and pushing to device..."
VISION_OUT_LOCAL="${VISION_OUT_LOCAL}" EMBED_OUT_LOCAL="${EMBED_OUT_LOCAL}" \
VISION_CTX_INFO="${VISION_CTX_INFO}" VISION_PRECISION="${VISION_PRECISION}" \
TOKEN_META_JSON="${TOKEN_META_JSON}" \
COMBINED_EMB="${COMBINED_EMB}" python - <<'PY'
import json
import os
from pathlib import Path

import numpy as np

vision_out_dir = Path(os.environ["VISION_OUT_LOCAL"])
embed_out_dir = Path(os.environ["EMBED_OUT_LOCAL"])
ctx_info = Path(os.environ["VISION_CTX_INFO"])
vision_precision = os.environ["VISION_PRECISION"]
token_meta = Path(os.environ["TOKEN_META_JSON"])
combined_out = Path(os.environ["COMBINED_EMB"])

def find_raw(dir_path: Path, preferred_names) -> Path:
    if not dir_path.exists():
        raise FileNotFoundError(f"Missing output dir: {dir_path}")
    result_dirs = sorted([p for p in dir_path.iterdir() if p.is_dir() and p.name.startswith("Result_")])
    if not result_dirs:
        raise FileNotFoundError(f"No Result_* in {dir_path}")
    raw_files = list(result_dirs[-1].glob("*.raw"))
    if not raw_files:
        raise FileNotFoundError(f"No .raw outputs in {result_dirs[-1]}")
    for name in preferred_names:
        p = result_dirs[-1] / name
        if p.exists():
            return p
    return raw_files[0]

vision_pref = ["image_features_native.raw", "output_0_native.raw", "output_0.raw"]
if vision_precision == "fp16":
    vision_pref = ["output_0.raw", "output_0_native.raw", "image_features_native.raw"]

vision_raw = find_raw(vision_out_dir, vision_pref)
embed_raw = find_raw(embed_out_dir, ["output_0.raw", "output_0_native.raw"])

if vision_precision == "int8":
    ctx = json.loads(ctx_info.read_text())
    graph = ctx["info"]["graphs"][0]["info"]
    out_info = graph["graphOutputs"][0]["info"]
    scale = out_info["quantizeParams"]["scaleOffset"]["scale"]
    offset = out_info["quantizeParams"]["scaleOffset"]["offset"]
    dims = out_info["dimensions"]
    vision_u8 = np.fromfile(vision_raw, dtype=np.uint8).reshape(dims)
    vision_f = (vision_u8.astype(np.float32) + float(offset)) * float(scale)
else:
    ctx = json.loads(ctx_info.read_text()) if ctx_info.exists() else None
    dims = None
    if ctx is not None:
        if "graphs" in ctx and ctx["graphs"]:
            out_info = ctx["graphs"][0]["graphOutputs"][0]
            dims = out_info.get("dimensions")
    if not dims:
        dims = [1, 256, 896]
    vision_f = np.fromfile(vision_raw, dtype=np.float32).reshape(dims)

embed_data = np.fromfile(embed_raw, dtype=np.float32)
if embed_data.size != 1 * 512 * 896:
    embed_data = np.fromfile(embed_raw, dtype=np.float16).astype(np.float32)
embed = embed_data.reshape(1, 512, 896)

meta = json.loads(token_meta.read_text())
num_text = len(meta["token_ids"])
text_embeds = embed[:, :num_text, :]
combined = np.concatenate([vision_f, text_embeds], axis=1).astype(np.float32)

meta["combined_len"] = int(combined.shape[1])
token_meta.write_text(json.dumps(meta, indent=2))

combined_out.parent.mkdir(parents=True, exist_ok=True)
combined.tofile(combined_out)
PY

adb push "${COMBINED_EMB}" "${DEVICE_COMBINED}" >/dev/null

echo "Preparing Genie device config..."
DEVICE_MODEL_BIN="${DEVICE_MODEL_BIN}" DEVICE_TOKENIZER_JSON="${DEVICE_TOKENIZER_JSON}" \
TOKEN_META_JSON="${TOKEN_META_JSON}" MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
GENIE_CONFIG="${GENIE_CONFIG}" python - <<'PY'
import json
import os
from pathlib import Path

model_bin = os.environ["DEVICE_MODEL_BIN"]
tokenizer_json = os.environ["DEVICE_TOKENIZER_JSON"]
max_new = int(os.environ["MAX_NEW_TOKENS"])
out_path = os.environ["GENIE_CONFIG"]
token_meta = Path(os.environ["TOKEN_META_JSON"])

ctx_size = 512
combined_len = None
if token_meta.exists():
    meta = json.loads(token_meta.read_text())
    combined_len = meta.get("combined_len")

if combined_len is not None and combined_len >= ctx_size:
    raise SystemExit(f"Combined embeddings length {combined_len} exceeds context size {ctx_size}")

if max_new <= 0:
    if combined_len is not None:
        max_new = max(1, ctx_size - combined_len)
    else:
        max_new = 50

config = {
    "dialog": {
        "version": 1,
        "type": "basic",
        "embedding": {"version": 1, "size": 896, "datatype": "float32"},
        "context": {
            "version": 1,
            "size": ctx_size,
            "n-vocab": 151936,
            "bos-token": 151644,
            "eos-token": 151645,
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

config["dialog"]["max-num-tokens"] = max_new

with open(out_path, "w") as f:
    json.dump(config, f, indent=2)
PY

adb push "${GENIE_CONFIG}" "${DEVICE_CONFIG}" >/dev/null

echo "Decoding with Genie CPU backend on device..."
GENIE_CMD="export LD_LIBRARY_PATH=\"${DEVICE_DIR}:/vendor/dsp/cdsp:/vendor/lib64\" && \
cd ${DEVICE_DIR} && \
./genie-t2t-run --config ${DEVICE_CONFIG} \
--embedding_file ${DEVICE_COMBINED} \
--embedding_table ${DEVICE_LUT_BIN} \
--embedding_query_output_type text"

PROFILE_ARG=""
if [[ "${PROFILE}" != "0" ]]; then
  PROFILE_ARG="--profile ${DEVICE_PROFILE}"
  adb shell "rm -f ${DEVICE_PROFILE}" >/dev/null 2>&1 || true
fi
if [[ -n "${PROFILE_ARG}" ]]; then
  GENIE_CMD="${GENIE_CMD} ${PROFILE_ARG}"
fi

GENIE_CMD="${GENIE_CMD}" python - <<'PY'
import os
import subprocess
import sys
import time

genie_cmd = os.environ["GENIE_CMD"]

proc = subprocess.Popen(
    ["adb", "shell", genie_cmd],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=0,
)

start = time.time()
first_token_time = None
seen_begin = False
capture = ""

while True:
    ch = proc.stdout.read(1)
    if ch == "" and proc.poll() is not None:
        break
    if ch:
        sys.stdout.write(ch)
        sys.stdout.flush()
        capture += ch
        if not seen_begin and "[BEGIN]:" in capture:
            seen_begin = True
            if first_token_time is None:
                first_token_time = time.time()

end = time.time()

gen_text = ""
if "[BEGIN]:" in capture:
    gen_text = capture.split("[BEGIN]:", 1)[1]
elif "[COMPLETE]:" in capture:
    gen_text = capture.split("[COMPLETE]:", 1)[1]
if "[END]" in gen_text:
    gen_text = gen_text.split("[END]", 1)[0]
gen_text = gen_text.strip()

if gen_text:
    print("\\n\\n=== Generated Text ===")
    print(gen_text)
PY

adb pull "${DEVICE_PROFILE}" "${PROFILE_OUT}" >/dev/null || true
if [[ -f "${PROFILE_OUT}" ]]; then
  echo "Profile saved to: ${PROFILE_OUT}"
  PROFILE_OUT="${PROFILE_OUT}" python - <<'PY'
import json
import os

path = os.environ["PROFILE_OUT"]
data = json.load(open(path))
events = data.get("components", [])[0].get("events", []) if data.get("components") else []
query = None
for e in events:
    if e.get("type") == "GenieDialog_query":
        query = e
        break
if not query:
    print("Genie stats not found in profile.")
    raise SystemExit(0)

def fmt(field, unit=""):
    val = field.get("value") if isinstance(field, dict) else None
    if val is None:
        return "n/a"
    if unit:
        return f"{val} {unit}".strip()
    return str(val)

ttfb = query.get("time-to-first-token", {})
prompt_rate = query.get("prompt-processing-rate", {})
gen_rate = query.get("token-generation-rate", {})
num_prompt = query.get("num-prompt-tokens", {})
num_gen = query.get("num-generated-tokens", {})
gen_time = query.get("token-generation-time", {})

print("=== Genie Profile Stats ===")
print(f"Prompt tokens: {fmt(num_prompt)}")
print(f"Prompt rate: {fmt(prompt_rate, prompt_rate.get('unit', ''))}")
print(f"TTFB: {fmt(ttfb, ttfb.get('unit', ''))}")
print(f"Generated tokens: {fmt(num_gen)}")
print(f"Gen rate: {fmt(gen_rate, gen_rate.get('unit', ''))}")
print(f"Gen time: {fmt(gen_time, gen_time.get('unit', ''))}")
PY
else
  echo "Profile not found at: ${PROFILE_OUT}" >&2
fi

echo "Outputs saved to: ${OUT_DIR}"

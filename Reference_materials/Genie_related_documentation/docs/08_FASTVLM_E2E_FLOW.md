# FastVLM End-to-End Flow on NPU

This document details the complete workflow for running Apple's FastVLM on Qualcomm's HTP NPU using GENIE.

## Source Files

| Component | Location |
|-----------|----------|
| E2E Script | `/home/aryamavmurthy/work/QIDK/VLM_on_edge/VLM_run_experiments_QIDK/fastvlm_yashas/handoff/scripts/run_e2e_int8_device.sh` |
| Export Project | `/home/aryamavmurthy/work/QIDK/VLM_on_edge/fastvlm_npu_export_project/` |
| Model Artifacts | `/home/aryamavmurthy/work/QIDK/VLM_on_edge/VLM_run_experiments_QIDK/fastvlm_yashas/handoff/models/` |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        HOST (Linux x86_64)                          │
├─────────────────────────────────────────────────────────────────────┤
│  1. Image Preprocessing (PIL/numpy)                                 │
│  2. Tokenization (HuggingFace Transformers)                         │
│  3. Create input files (.raw)                                       │
│  4. Push to device via ADB                                          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ adb push
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   DEVICE (Snapdragon 8 Elite)                       │
├─────────────────────────────────────────────────────────────────────┤
│  ┌────────────────────┐                                             │
│  │  Vision Encoder    │  qnn-net-run --backend libQnnHtp.so         │
│  │  (SigLIP INT8)     │  → vision_embeddings.raw                    │
│  └────────────────────┘                                             │
│            │                                                        │
│            ▼                                                        │
│  ┌────────────────────┐                                             │
│  │  Text Embedder     │  qnn-net-run --backend libQnnHtp.so         │
│  │  (LUT Lookup)      │  → text_embeddings.raw                      │
│  └────────────────────┘                                             │
│            │                                                        │
│            ▼                                                        │
│  ┌────────────────────┐  (Concatenation done on host, pushed back)  │
│  │  Combined Embeds   │  [vision_emb | text_emb]                    │
│  └────────────────────┘                                             │
│            │                                                        │
│            ▼                                                        │
│  ┌────────────────────┐                                             │
│  │  LLM Decoder       │  genie-t2t-run --embedding_file             │
│  │  (Qwen2 via GENIE) │  → Generated text                           │
│  └────────────────────┘                                             │
└─────────────────────────────────────────────────────────────────────┘
```

## Step-by-Step Workflow

### Step 1: Environment Setup

```bash
# Source QAIRT environment
source "${ROOT_DIR}/2.42.0.251225/qnn_env/bin/activate"
source "${ROOT_DIR}/2.42.0.251225/bin/envsetup.sh"
```

### Step 2: Prepare Image Input

```python
# From run_e2e_int8_device.sh (embedded Python)
from PIL import Image
import numpy as np

# Resize to model size (1024x1024 for FastVLM)
img = Image.open(image_path).convert("RGB")
img = img.resize((image_size, image_size), Image.BICUBIC)

# For INT8 vision encoder: raw uint8
if vision_precision == "int8":
    arr = np.asarray(img, dtype=np.uint8)
    arr.tofile(vision_out)

# For FP16 vision encoder: normalized float32 NCHW
else:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    arr = np.expand_dims(arr, axis=0)    # Add batch dimension
    arr.tofile(vision_out)
```

### Step 3: Prepare Text Input

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

# Apply chat template if available
if getattr(tokenizer, "apply_chat_template", None):
    messages = [{"role": "user", "content": prompt}]
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
else:
    token_ids = tokenizer(prompt, add_special_tokens=True).input_ids

# Pad to context length
input_ids = np.full((1, cache_len), pad_id, dtype=np.int32)
input_ids[0, :len(token_ids)] = np.array(token_ids, dtype=np.int32)
input_ids.tofile(embed_out)
```

### Step 4: Push Artifacts to Device

```bash
# Create directories
adb shell "mkdir -p ${DEVICE_DIR}/inputs/vision ${DEVICE_DIR}/inputs/embed"

# Push binaries and libraries
adb push genie-t2t-run ${DEVICE_DIR}/
adb push qnn-net-run ${DEVICE_DIR}/
adb push libQnnHtp.so ${DEVICE_DIR}/
adb push libQnnHtpV79Skel.so ${DEVICE_DIR}/
adb push libGenie.so ${DEVICE_DIR}/

# Push model context binaries
adb push fastvlm_vision_encoder_int8_htp.bin ${DEVICE_DIR}/
adb push fastvlm_embed_tokens_fp16_htp.bin ${DEVICE_DIR}/
adb push fastvlm_model_z8.bin ${DEVICE_DIR}/
adb push tokenizer.json ${DEVICE_DIR}/
adb push LUT.bin ${DEVICE_DIR}/

# Push input files
adb push pixel_values.raw ${DEVICE_DIR}/inputs/vision/
adb push input_ids.raw ${DEVICE_DIR}/inputs/embed/
```

### Step 5: Run Vision Encoder on HTP

```bash
# Set up environment
export ADSP_LIBRARY_PATH="${DEVICE_DIR};/vendor/dsp/cdsp"
export LD_LIBRARY_PATH="${DEVICE_DIR}:/vendor/lib64"

# Run vision encoder
./qnn-net-run \
    --backend libQnnHtp.so \
    --retrieve_context fastvlm_vision_encoder_int8.bin \
    --input_list ${DEVICE_VISION_LIST} \
    --use_native_input_files \
    --use_native_output_files \
    --output_dir outputs/vision \
    --num_inferences 1
```

**Output**: `image_features_native.raw` (vision embeddings)

### Step 6: Run Text Embedder on HTP

```bash
./qnn-net-run \
    --backend libQnnHtp.so \
    --retrieve_context fastvlm_embed_tokens_fp16_htp.bin \
    --input_list ${DEVICE_EMBED_LIST} \
    --use_native_input_files \
    --use_native_output_files \
    --output_dir outputs/embed \
    --num_inferences 1
```

**Output**: `output_0.raw` (text embeddings)

### Step 7: Combine Vision + Text Embeddings (Host)

```python
# Pull outputs from device
adb pull ${DEVICE_VISION_OUT} ${VISION_OUT_LOCAL}
adb pull ${DEVICE_EMBED_OUT} ${EMBED_OUT_LOCAL}

# Dequantize INT8 vision embeddings
if vision_precision == "int8":
    # Read context binary info for quant params
    ctx = json.loads(ctx_info.read_text())
    scale = ctx["info"]["graphs"][0]["info"]["graphOutputs"][0]["info"]["quantizeParams"]["scaleOffset"]["scale"]
    offset = ctx["info"]["graphs"][0]["info"]["graphOutputs"][0]["info"]["quantizeParams"]["scaleOffset"]["offset"]
    
    vision_u8 = np.fromfile(vision_raw, dtype=np.uint8).reshape(dims)
    vision_f = (vision_u8.astype(np.float32) + float(offset)) * float(scale)
else:
    vision_f = np.fromfile(vision_raw, dtype=np.float32).reshape(dims)

# Load text embeddings
embed = np.fromfile(embed_raw, dtype=np.float32).reshape(1, 512, 896)

# Trim to actual token count (remove padding)
num_text = len(meta["token_ids"])
text_embeds = embed[:, :num_text, :]

# CONCATENATE: Vision embeddings first, then text embeddings
combined = np.concatenate([vision_f, text_embeds], axis=1).astype(np.float32)
# Shape: [1, num_vision_tokens + num_text_tokens, embedding_dim]

combined.tofile(combined_out)
```

**This is the key step**: Vision tokens come BEFORE text tokens in the sequence.

### Step 8: Generate Genie Configuration

```python
config = {
    "dialog": {
        "version": 1,
        "type": "basic",
        "embedding": {
            "version": 1, 
            "size": 896,          # FastVLM embedding dimension
            "datatype": "float32"
        },
        "context": {
            "version": 1,
            "size": 512,          # Context length
            "n-vocab": 151936,    # Qwen2 vocabulary
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
                "type": "QnnGenAiTransformer",  # CPU backend for decoder
                "QnnGenAiTransformer": {
                    "version": 1,
                    "model-input": "embeddings",  # Takes embeddings, not token IDs
                    "use-mmap": True,
                    "n-layer": 24,
                    "n-embd": 896,
                    "n-heads": 14,
                    "n-kv-heads": 2,
                },
            },
            "model": {
                "version": 1, 
                "type": "library", 
                "library": {"version": 1, "model-bin": model_bin}
            },
        },
    }
}
```

### Step 9: Run GENIE Decoder

```bash
./genie-t2t-run \
    --config ${DEVICE_CONFIG} \
    --embedding_file ${DEVICE_COMBINED} \
    --embedding_table ${DEVICE_LUT_BIN} \
    --embedding_query_output_type text \
    --profile ${DEVICE_PROFILE}
```

**Key flags:**
- `--embedding_file`: Pre-computed combined embeddings
- `--embedding_table`: LUT for token-to-embedding during generation
- `--embedding_query_output_type text`: Output decoded text (not tokens)

### Step 10: Stream Output

```python
# Parse Genie output
# Format: [BEGIN]:token1 token2...[END]

if "[BEGIN]:" in capture:
    gen_text = capture.split("[BEGIN]:", 1)[1]
if "[END]" in gen_text:
    gen_text = gen_text.split("[END]", 1)[0]
    
print("=== Generated Text ===")
print(gen_text.strip())
```

## Model Artifacts

### Required Files

| File | Description | Size |
|------|-------------|------|
| `fastvlm_vision_encoder_int8_htp.bin` | Vision encoder context binary | ~100MB |
| `fastvlm_embed_tokens_fp16_htp.bin` | Token embedder context binary | ~50MB |
| `fastvlm_model_z8.bin` | LLM decoder model | ~500MB |
| `LUT.bin` | Embedding lookup table | ~50MB |
| `tokenizer.json` | Tokenizer configuration | ~5MB |

### Context Binary Generation

```bash
# Vision encoder (INT8 quantized)
qnn-context-binary-generator \
    --model fastvlm_vision_encoder.onnx \
    --backend libQnnHtp.so \
    --htp_arch v79 \
    --output_dir ./vision_ctx \
    --binary_file fastvlm_vision_encoder_int8_htp.bin

# Text embedder
qnn-context-binary-generator \
    --model fastvlm_embed_tokens.onnx \
    --backend libQnnHtp.so \
    --htp_arch v79 \
    --output_dir ./embed_ctx \
    --binary_file fastvlm_embed_tokens_fp16_htp.bin
```

## Performance Profiling

```python
# Parse profile.json output
events = data["components"][0]["events"]
for e in events:
    if e["type"] == "GenieDialog_query":
        print(f"TTFB: {e['time-to-first-token']['value']} ms")
        print(f"Gen rate: {e['token-generation-rate']['value']} tokens/s")
        print(f"Prompt tokens: {e['num-prompt-tokens']['value']}")
        print(f"Generated tokens: {e['num-generated-tokens']['value']}")
```

## Key Insights

### 1. Embedding Concatenation Order
Vision embeddings MUST come before text embeddings:
```
[vision_token_0, vision_token_1, ..., text_token_0, text_token_1, ...]
```

### 2. Quantization Strategy
- Vision Encoder: INT8 (important for image processing speed)
- Text Embedder: FP16 (maintains precision)
- LLM Decoder: W4A16 or INT8 (balance of size and quality)

### 3. KV Cache Size
```
Combined prompt length = num_vision_tokens + num_text_tokens
Max generation = context_size - combined_prompt_length
```

### 4. Backend Choice
The E2E script uses:
- `QnnHtp` for vision encoder and embedder (NPU)
- `QnnGenAiTransformer` for decoder (CPU reference)

For production, decoder should also use `QnnHtp` with proper context binaries.

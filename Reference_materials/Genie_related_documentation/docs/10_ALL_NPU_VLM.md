# Running Full VLM on NPU (Vision Encoder + Projection + Text Decoder)

## Goal

Run **entire FastVLM model** on Qualcomm HTP NPU:
- Vision Encoder (SigLIP) → NPU ✓
- Projection/Connector → NPU (part of decoder model)
- Text Generation Decoder (Qwen2) → NPU (currently CPU in E2E, needs to be NPU)

## Current vs Target Architecture

### Current E2E Flow (Hybrid)
```
Vision Encoder (NPU) ───┐
                       │
Text Embedder (NPU) ───┼──> Combined Embeddings ──> LLM Decoder (CPU) ──> Text
                       │                                   QnnGenAiTransformer
```

### Target Flow (All NPU)
```
Vision Encoder (NPU) ───┐
                       │
Text Embedder (NPU) ───┼──> Combined Embeddings ──> LLM Decoder (NPU) ──> Text
                       │                              QnnHtp
Projection Layer (part of decoder, compiled in)
```

---

## Prerequisites

### 1. Model Export Requirements

The full decoder must be exported with the **projection layer included**. In FastVLM, the projection is typically:
- A small MLP (2 layers)
- Maps vision embeddings → LLM space
- Must be compiled as part of decoder

### 2. Context Binary for Decoder

When exporting the decoder for HTP, the projection layer outputs must be accessible:
- Input: Combined `[vision_embeddings, text_embeddings]`
- Output: `hidden_states` (ready for attention layers)

---

## Modified Export Workflow

### Step 1: Export Full Decoder with Projection

Current export script (`export_decoder.py`) must include:
1. Vision encoder → ONNX
2. **Projection/Connector layer** → ONNX
3. Text decoder → ONNX
4. Fusion of all components

Modified export structure:
```python
# Modified export_decoder.py for full NPU

class FastVLMDecoderExport:
    def export_with_projection(self):
        # 1. Export vision encoder
        vision_encoder = self.create_vision_encoder()
        
        # 2. Export projection layer (NEW!)
        projection = self.create_projection_layer()
        
        # 3. Export text decoder with projection input
        text_decoder = self.create_text_decoder(
            input_with_projection=True  # Key change!
        )
        
        # 4. Compile combined model
        combined_model = self.combine([
            vision_encoder,
            projection,      # Added projection
            text_decoder
        ])
        
        # Export to ONNX
        torch.onnx.export(
            combined_model,
            (pixel_values, input_ids),
            (hidden_states,),
            "fastvlm_full_npu.onnx"
            input_names=["pixel_values", "input_ids"],
            output_names=["hidden_states"]
        )
```

### Step 2: Context Binary Configuration

The decoder context binary must expect **embeddings as input**:

```bash
qnn-context-binary-generator \
    --model fastvlm_full_decoder.onnx \
    --backend libQnnHtp.so \
    --htp_arch v79 \
    --input_embeddings      # CRITICAL: Expect embedding input
    --binary_file fastvlm_full_decoder_htp.bin
```

**Required modifications:**
1. Projection weights must be quantized (INT8/W4)
2. Model must be compiled with `--input_embeddings` flag
3. KV cache configured for NPU execution

---

## GENIE Configuration for All-NPU

### Updated Text Generator Config

```json
{
  "text-generator": {
    "version": 1,
    "type": "basic",
    "accumulator-size": 400000000,
    "embedding": {
      "version": 1,
      "type": "lut",
      "lut-path": "embedding_int8_lut.bin",
      "size": 896,
      "datatype": "float32"
    },
    "context": {
      "version": 1,
      "size": 4096,
      "n-vocab": 151936,
      "bos-token": 151644,
      "eos-token": 151645,
      "pad-token": 151643
    },
    "tokenizer": {
      "version": 1,
      "path": "tokenizer.json"
    },
    "sampler": {
      "version": 1,
      "seed": 42,
      "temp": 0.8,
      "top-k": 40,
      "top-p": 0.95
    },
    "engine": {
      "version": 1,
      "n-threads": 6,
      "backend": {
        "version": 1,
        "type": "QnnHtp",        # CHANGED from QnnGenAiTransformer
        "QnnHtp": {
          "version": 1,
          "spill-fill-bufsize": 0,
          "use-mmap": true,
          "mmap-budget": 0,
          "poll": false,
          "cpu-mask": "0xe0",
          "kv-dim": 128,
          "enable-graph-switching": true
        }
      },
      "model": {
        "version": 1,
        "type": "binary",
        "binary": {
          "version": 1,
          "ctx-bins": [
            "fastvlm_full_decoder_part1_htp.bin",
            "fastvlm_full_decoder_part2_htp.bin"
          ]
        },
        "positional-encoding": {
          "type": "rope",
          "rope-dim": 64,
          "rope-theta": 10000.0
        }
      }
    }
  }
}
```

**Key changes from E2E config:**
1. `backend.type`: `"QnnGenAiTransformer"` → `"QnnHtp"` (NPU backend)
2. `backend.QnnHtp`: Add HTP-specific configuration
3. `model.input`: `"embeddings"` (embeddings input, not token IDs)

---

## Projection Layer Handling

### What is the Projection Layer?

In VLMs, the projection layer (also called connector, adapter, or projector) maps:
- Vision embeddings (e.g., 1024 or 1280 dim)
- → LLM embedding dimension (e.g., 896 for Qwen2)

### Projection in FastVLM

Typical FastVLM projection:
```python
# Typical FastVLM architecture

class FastVLMProjection(nn.Module):
    def __init__(self, vision_dim=1024, llm_dim=896):
        self.gate = nn.Linear(vision_dim, llm_dim)
        self.up = nn.Linear(vision_dim, llm_dim)
        
    def forward(self, vision_features, text_features):
        # Vision gate mechanism
        gate = torch.sigmoid(self.gate(vision_features))
        
        # Combine vision and text
        combined = gate * vision_features + text_features
        
        return combined
```

### Compiled into Context Binary

The projection layer is compiled into the decoder context binary:
```
fastvlm_full_decoder_htp.bin
├── [Projection Layer]  ← Part of the graph
│   ├── projection weights (INT8 quantized)
│   └── projection activations
├── [Attention Layers 1-24]
├── [FFN Layers 1-24]
└── [KV Cache Management]
```

---

## Data Flow for All-NPU VLM

### Complete Execution Flow

```
1. Image Preprocessing (Host)
   ↓ PIL resize to 1024x1024
   ↓ Normalize to uint8 or float32

2. Vision Encoding (NPU - qnn-net-run)
   ↓ Input: pixel_values (1024x1024x3)
   ↓ Output: vision_embeddings [1, num_vision_tokens, 1024]

3. Tokenization (Host)
   ↓ Apply chat template
   ↓ Tokenize prompt

4. Text Embedding (NPU - qnn-net-run)
   ↓ Input: token_ids
   ↓ Output: text_embeddings [1, num_text_tokens, 896]

5. Projection (NPU - Part of decoder!)
   ↓ Input: vision_embeddings (1024 dim)
   ↓ Output: projected_vision [1, num_vision_tokens, 896]

6. Concatenation (Host or NPU)
   ↓ Combined: [projected_vision | text_embeddings]
   ↓ Shape: [1, total_tokens, 896]

7. Autoregressive Decoding (NPU - genie-t2t-run)
   ↓ Input: combined_embeddings
   ↓ Output: Generated text (token by token)
   ↓ KV Cache updates automatically on NPU
```

---

## Modified Execution Script

### run_all_npu.sh (New)

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Environment
source "${ROOT_DIR}/2.42.0.251225/qnn_env/bin/activate"
source "${ROOT_DIR}/2.42.0.251225/bin/envsetup.sh"

# Model Paths
VISION_CTX="${ROOT_DIR}/models/vision_encoder_int8_htp.bin"
DECODER_CTX="${ROOT_DIR}/models/fastvlm_full_decoder_htp.bin"  # CHANGED
LUT_BIN="${ROOT_DIR}/models/LUT.bin"
TOKENIZER_JSON="${ROOT_DIR}/models/tokenizer.json"

# Output paths
OUT_DIR="${ROOT_DIR}/run_outputs/all_npu"
mkdir -p "${OUT_DIR}"

# Step 1: Prepare image
echo "Preprocessing image..."
python <<'PY'
from PIL import Image
import numpy as np
import os

image_path = os.environ["IMAGE_PATH"]
out_path = os.environ["VISION_OUT"]

img = Image.open(image_path).convert("RGB")
img = img.resize((1024, 1024), Image.BICUBIC)
arr = np.asarray(img, dtype=np.uint8)
arr.tofile(out_path)
PY

# Step 2: Run vision encoder (NPU)
echo "Running vision encoder on NPU..."
adb shell "mkdir -p /data/local/tmp/vlm_npu/inputs/vision"
adb push pixel_values.raw /data/local/tmp/vlm_npu/inputs/vision/

adb shell "export LD_LIBRARY_PATH=/data/local/tmp/vlm_npu && \
    cd /data/local/tmp/vlm_npu && \
    ./qnn-net-run \
        --backend libQnnHtp.so \
        --retrieve_context ${VISION_CTX} \
        --input_list inputs/vision/pixel_values.raw \
        --output_dir outputs/vision"

# Step 3: Tokenize prompt
echo "Tokenizing prompt..."
python <<'PY'
from transformers import AutoTokenizer
import numpy as np
import os
import json

model_dir = os.environ["MODEL_DIR"]
prompt = os.environ["PROMPT"]
out_path = os.environ["TOKEN_OUT"]

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
messages = [{"role": "user", "content": prompt}]
token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)

input_ids = np.full((1, 512), tokenizer.pad_token_id, dtype=np.int32)
input_ids[0, :len(token_ids)] = np.array(token_ids, dtype=np.int32)

meta = {"prompt": prompt, "token_ids": token_ids.tolist()}
with open(os.environ["META_OUT"], "w") as f:
    json.dump(meta, f)

input_ids.tofile(out_path)
PY

# Step 4: Combine embeddings (for input to decoder)
echo "Combining embeddings for NPU decoder..."
# Note: This step may need to be done on NPU if projection is part of decoder

# Step 5: Run decoder on NPU
echo "Running full decoder on NPU..."
adb shell "mkdir -p /data/local/tmp/vlm_npu/inputs/decoder"
adb push combined_embeddings.raw /data/local/tmp/vlm_npu/inputs/decoder/
adb push ${DECODER_CTX} /data/local/tmp/vlm_npu/
adb push ${LUT_BIN} /data/local/tmp/vlm_npu/
adb push ${TOKENIZER_JSON} /data/local/tmp/vlm_npu/
adb push /path/to/genie-all-npu-config.json /data/local/tmp/vlm_npu/

adb shell "export LD_LIBRARY_PATH=/data/local/tmp/vlm_npu && \
    cd /data/local/tmp/vlm_npu && \
    ./genie-t2t-run \
        --config genie-all-npu-config.json \
        --embedding_file combined_embeddings.raw \
        --embedding_table ${LUT_BIN} \
        --embedding_query_output_type text"

echo "VLM inference complete!"
```

---

## Key Differences: E2E vs All-NPU

| Aspect | E2E (Hybrid) | All-NPU (Target) |
|---------|------------------|-------------------|
| **Vision Encoder** | NPU (QnnHtp) | NPU (QnnHtp) |
| **Text Embedder** | NPU (QnnHtp) | NPU (QnnHtp) |
| **Projection Layer** | CPU (part of QnnGenAiTransformer) | NPU (part of decoder binary) |
| **Text Decoder** | CPU (QnnGenAiTransformer) | NPU (QnnHtp) |
| **KV Cache** | CPU memory | NPU VTCM |
| **Latency** | Medium (NPU vision, CPU decode) | Low (all NPU) |
| **Memory** | Higher (CPU allocations) | Lower (all in NPU) |

---

## Troubleshooting

### 1. Decoder Context Binary Too Large

**Symptom**: Context binary > 1GB, fails to load

**Solution**:
- Split decoder into multiple segments (AR-1, AR-32, AR-128, AR-full)
- Use graph switching for different sequence lengths

```bash
# Split compilation
qnn-context-binary-generator \
    --model decoder_part1.onnx \
    --variant ar32 \
    --binary_file decoder_ar32.bin

qnn-context-binary-generator \
    --model decoder_part2.onnx \
    --variant ar128 \
    --binary_file decoder_ar128.bin
```

### 2. Projection Quantization Loss

**Symptom**: Degraded quality after INT8 quantization

**Solution**:
- Use higher precision for projection (FP16 or W4A16)
- Apply calibration dataset with representative images
- Consider keeping projection in FP16 while rest is INT8

### 3. Memory Overflow on NPU

**Symptom**: OOM or "context exceeded" errors

**Solution**:
- Reduce context size (4096 → 2048)
- Enable KV cache sliding window
- Adjust accumulator size in config

```json
{
  "text-generator": {
    "accumulator-size": 200000000,  // Reduced from 400MB
    "context": {
      "size": 2048  // Reduced context
    }
  }
}
```

---

## Reference Files

| File | Location | Purpose |
|------|-----------|---------|
| `export_decoder.py` | `fastvlm_npu_export_project/` | Export decoder with projection |
| `nsp-model.cpp` | `Genie/src/qualla/engines/qnn-htp/` | NPU model execution |
| `TextGenerator.cpp` | `Genie/src/pipeline/` | Text generation node |
| `Accumulator.cpp` | `Genie/src/pipeline/` | Embedding accumulation |
| `nsp-kvmanager.cpp` | `Genie/src/qualla/engines/qnn-htp/` | KV cache management |

---

## Next Steps

1. **Export full decoder**: Modify `export_decoder.py` to include projection layer
2. **Compile to HTP**: Use `qnn-context-binary-generator` with `--input_embeddings`
3. **Update GENIE config**: Change backend to `QnnHtp` for text generator
4. **Test**: Run vision + prompt through full NPU pipeline
5. **Profile**: Compare latency with E2E hybrid approach

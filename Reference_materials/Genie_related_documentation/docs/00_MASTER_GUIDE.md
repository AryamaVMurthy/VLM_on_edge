# FastVLM on Qualcomm HTP NPU - Master Guide

## Quick Navigation
1. [00_MASTER_GUIDE.md](./00_MASTER_GUIDE.md) - This file, overall architecture
2. [01_GENIE_OVERVIEW.md](./01_GENIE_OVERVIEW.md) - GENIE SDK basics
3. [02_GENIE_PIPELINE_API.md](./02_GENIE_PIPELINE_API.md) - Pipeline and Node APIs
4. [03_GENIE_DIALOG_API.md](./03_GENIE_DIALOG_API.md) - Dialog API for text generation
5. [04_IMAGE_ENCODER.md](./04_IMAGE_ENCODER.md) - Vision encoder implementation
6. [05_TEXT_GENERATOR.md](./05_TEXT_GENERATOR.md) - Text generation and decoding
7. [06_KV_CACHE.md](./06_KV_CACHE.md) - KV Cache management on NPU
8. [07_HTP_BACKEND.md](./07_HTP_BACKEND.md) - HTP/NPU backend configuration
9. [08_FASTVLM_E2E_FLOW.md](./08_FASTVLM_E2E_FLOW.md) - End-to-end FastVLM workflow
10. [09_KEY_CODE_LOCATIONS.md](./09_KEY_CODE_LOCATIONS.md) - File paths reference

---

## Architecture Overview

### What is GENIE?

**Qualcomm GENIE (Generative AI Next-generation Intelligent Engine)** is a software library that simplifies deployment of GenAI pipelines on Qualcomm AI Runtime (QAIRT). It provides:

- High-level **Dialog API** for simple text-to-text chat
- Low-level **Pipeline API** for composable multimodal workflows
- Automatic **KV Cache management** for transformers
- Direct integration with **QNN HTP backend** (Hexagon Tensor Processor)

### FastVLM Architecture on NPU

```
+------------------+     +-----------------------+     +------------------+
|   IMAGE INPUT    | --> |   Vision Encoder      | --> |                  |
|   (1024x1024)    |     |   (SigLIP/CLIP INT8)  |     |   EMBEDDING      |
+------------------+     +-----------------------+     |   CONCATENATION  |
                                                       |                  |
+------------------+     +-----------------------+     |   [vision_emb]   |
|   TEXT PROMPT    | --> |   Text Embedder       | --> |   [text_emb]     |
|   "Describe..."  |     |   (LUT Lookup)        |     |                  |
+------------------+     +-----------------------+     +--------+---------+
                                                                |
                                                                v
                                                       +------------------+
                                                       |   LLM DECODER    |
                                                       |   (Qwen2/Llama)  |
                                                       |   via GENIE      |
                                                       +--------+---------+
                                                                |
                                                                v
                                                       +------------------+
                                                       |   TEXT OUTPUT    |
                                                       |   "A cat sitting"|
                                                       +------------------+
```

### Execution Backends

| Backend | Description | Use Case |
|---------|-------------|----------|
| `QnnHtp` | Hexagon Tensor Processor (NPU) | Production inference |
| `QnnGpu` | Adreno GPU | Fallback/testing |
| `QnnGenAiTransformer` | CPU-based transformer | Development/debugging |

### Key Data Flow for VLM

1. **Image Preprocessing**: Resize to model size (e.g., 1024x1024), normalize
2. **Vision Encoding**: Run through SigLIP/CLIP encoder on HTP
3. **Text Tokenization**: Apply chat template, convert to token IDs
4. **Embedding Lookup**: Get text embeddings from LUT
5. **Concatenation**: Combine `[vision_embeddings, text_embeddings]`
6. **Autoregressive Decoding**: Feed to LLM, generate tokens one-by-one
7. **KV Cache Update**: After each token, update key/value cache for next step

---

## Key Components

### 1. GENIE APIs

| API | Purpose | When to Use |
|-----|---------|-------------|
| `GenieDialog` | High-level chat interface | Simple text-to-text or embedding-to-text |
| `GeniePipeline` | Composable node graph | Multimodal (image + text) |
| `GenieNode` | Individual pipeline components | ImageEncoder, TextGenerator, TextEncoder |
| `GenieSampler` | Token sampling | Temperature, top-k, top-p |
| `GenieTokenizer` | Tokenization/detokenization | Text encoding/decoding |

### 2. Node Types for VLM

```c
// From GenieNode.h
GENIE_NODE_TEXT_GENERATOR_TEXT_INPUT       = 0,   // Text string input
GENIE_NODE_TEXT_GENERATOR_EMBEDDING_INPUT  = 1,   // Embedding tensor input
GENIE_NODE_TEXT_GENERATOR_TEXT_OUTPUT      = 2,   // Generated text output
GENIE_NODE_IMAGE_ENCODER_IMAGE_INPUT       = 200, // Image pixels input
GENIE_NODE_IMAGE_ENCODER_EMBEDDING_OUTPUT  = 201, // Vision embeddings output
```

### 3. KV Cache Modes

```cpp
// From nsp-kvmanager.hpp
enum KVManagerMode {
    POINTER_SHIFT,  // Most efficient: modify tensor pointers (no data copy)
    SHIFT_CONCAT,   // Shift buffer, concatenate new tokens
    SMART_MASK      // Use attention mask to handle active region
};
```

---

## Configuration JSON Structure

### VLM Text Generator Config (glm-4v example)

```json
{
  "text-generator": {
    "version": 1,
    "type": "basic",
    "accumulator-size": 400000000,  // Buffer for vision embeddings
    
    "embedding": {
      "version": 1,
      "type": "lut",              // Lookup table for token embeddings
      "lut-path": "embedding_int8_lut.bin",
      "size": 2048,               // Embedding dimension
      "datatype": "ufixed8"       // 8-bit unsigned quantized
    },
    
    "context": {
      "version": 1,
      "size": 4096,               // Max context length
      "n-vocab": 59264,
      "bos-token": -1,
      "eos-token": [59246, 59253, 59255]
    },
    
    "engine": {
      "version": 1,
      "n-threads": 3,
      "backend": {
        "type": "QnnHtp",
        "QnnHtp": {
          "kv-dim": 128,          // KV cache embedding dimension
          "enable-graph-switching": true
        }
      },
      "model": {
        "type": "binary",
        "binary": {
          "ctx-bins": ["model_part1.bin", "model_part2.bin"]
        }
      }
    }
  }
}
```

---

## Critical Code Paths

### Vision Encoder Flow
```
ImageEncoder.cpp::setImageInputData()
    -> m_encoder->encode(inputs, m_data)
    -> m_pipeline->m_accumulator->append(vision_embeddings)
```

### Text Generation Flow
```
TextGenerator.cpp::execute()
    -> m_pipeline->m_accumulator->getData()  // Get combined embeddings
    -> m_generator->embeddingQuery(embData)  // Run LLM with embeddings
    -> m_textOutputCallback(response)        // Stream output
```

### KV Cache Update Flow
```
nsp-kvmanager.cpp::dispatchUpdate()
    -> runKVUpdateJob()        // Parallel update across threads
    -> updateKey()/updateValue()
    -> registerPointerOffset() // For POINTER_SHIFT mode
```

---

## Build & Deploy Commands

### 1. Compile Vision Encoder
```bash
qnn-context-binary-generator \
  --model vision_encoder.onnx \
  --backend libQnnHtp.so \
  --htp_arch v79 \
  --binary_file vision.bin
```

### 2. Push to Device
```bash
adb push vision.bin /data/local/tmp/vlm/
adb push llm.bin /data/local/tmp/vlm/
adb push tokenizer.json /data/local/tmp/vlm/
adb push config.json /data/local/tmp/vlm/
```

### 3. Run Inference
```bash
export LD_LIBRARY_PATH=/data/local/tmp/vlm
./genie-t2t-run --config config.json \
    --embedding_file combined_embeddings.raw \
    --embedding_query_output_type text
```

---

## Next Steps

1. Read [01_GENIE_OVERVIEW.md](./01_GENIE_OVERVIEW.md) for GENIE fundamentals
2. Study the [06_KV_CACHE.md](./06_KV_CACHE.md) for understanding autoregressive generation
3. Review [08_FASTVLM_E2E_FLOW.md](./08_FASTVLM_E2E_FLOW.md) for the complete workflow

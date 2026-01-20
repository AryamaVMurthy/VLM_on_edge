# FastVLM on NPU - Architecture Overview

## Complete Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                      HOST (Development Machine)                    │
├──────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │ 1. Model Export (Python)                                 │    │
│  │    - FastVLM model from HuggingFace                     │    │
│  │    - Export vision encoder → ONNX                           │    │
│  │    - Export projection layer → ONNX                         │    │
│  │    - Export text decoder → ONNX                             │    │
│  │    - Combine into fastvlm_full_model.onnx                 │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                     │                                             │
│                     ▼                                             │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │ 2. QNN Compilation (Host Tools)                           │    │
│  │    - qnn-context-binary-generator                          │    │
│  │    - --backend libQnnHtp.so                               │    │
│  │    - --htp_arch v79                                       │    │
│  │    - Output: Context binaries (.bin)                        │    │
│  └──────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              │ adb push / flash
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                 DEVICE (Snapdragon 8 Elite)                       │
├──────────────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │ Stage 1: Vision Encoding (NPU)                          │    │
│  │                                                          │    │
│  │  Input:  pixel_values (1024x1024x3, uint8)            │    │
│  │  Context: fastvlm_vision_encoder_int8_htp.bin          │    │
│  │  Backend: libQnnHtp.so                                   │    │
│  │  Tool: qnn-net-run                                       │    │
│  │                                                          │    │
│  │  Output:  vision_embeddings [1, N_vision, 1024]          │    │
│  │  Latency: ~10-50ms                                      │    │
│  │                                                          │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                     │                                             │
│                     ▼                                             │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │ Stage 2: Text Embedding (NPU)                           │    │
│  │                                                          │    │
│  │  Input:  input_ids [1, N_text, int32]                   │    │
│  │  Context: fastvlm_embed_tokens_fp16_htp.bin           │    │
│  │  Backend: libQnnHtp.so                                   │    │
│  │  Tool: qnn-net-run                                       │    │
│  │  LUT: embedding_int8_lut.bin                              │    │
│  │                                                          │    │
│  │  Output:  text_embeddings [1, N_text, 896]               │    │
│  │  Latency: ~5-20ms                                       │    │
│  │                                                          │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                     │                                             │
│                     ▼                                             │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │ Stage 3: Embedding Concatenation                         │    │
│  │                                                          │    │
│  │  Combined: [vision_emb | text_emb]                      │    │
│  │  Shape: [1, N_vision + N_text, 896]                     │    │
│  │  Order: CRITICAL - vision tokens MUST come first!               │    │
│  │                                                          │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                     │                                             │
│                     ▼                                             │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │ Stage 4: LLM Decoding (NPU)                             │    │
│  │                                                          │    │
│  │  Input:  combined_embeddings [1, total_tokens, 896]       │    │
│  │  Context: fastvlm_full_decoder_htp.bin                    │    │
│  │  Includes: Projection layer + Qwen2 decoder                  │    │
│  │  Backend: QnnHtp (NPU via GENIE)                       │    │
│  │  Tool: genie-t2t-run                                      │    │
│  │  KV Cache: Managed by GENIE, POINTER_SHIFT mode              │    │
│  │  VTCM: On-NPU SRAM (~8-16MB)                           │    │
│  │  Variants: AR-1, AR-32, AR-128 for efficiency         │    │
│  │                                                          │    │
│  │  Autoregressive Generation:                                 │    │
│  │    Token 1 → Token 2 → Token 3 → ... → <eos>             │    │
│  │    Each step: ~50-100ms (first), ~10-30ms (subsequent)     │    │
│  │                                                          │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                     │                                             │
│                     ▼                                             │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │ Output: Generated Text                                   │    │
│  │                                                          │    │
│  │  "A cat sitting on a colorful mat in a sunny room..."          │    │
│  │                                                          │    │
│  └──────────────────────────────────────────────────────────────┘    │
│                                                               │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Component Breakdown

### 1. Vision Encoder (SigLIP/CLIP)

| Property | Value |
|---------|--------|
| Input Resolution | 1024x1024 pixels |
| Input Channels | 3 (RGB) |
| Input Type | uint8 (INT8 quantized) |
| Output Dim | 1024 (vision embedding dimension) |
| Output Tokens | ~128-256 (depends on patch size) |
| Context Binary | `fastvlm_vision_encoder_int8_htp.bin` |
| Runtime Tool | `qnn-net-run --backend libQnnHtp.so` |

### 2. Projection Layer

| Property | Value |
|---------|--------|
| Input Dim | 1024 (from vision encoder) |
| Output Dim | 896 (LLM embedding dimension) |
| Type | MLP or Linear projection |
| In Compiled Graph | Part of decoder binary |
| Purpose | Align vision features to LLM embedding space |

### 3. Text Embedder (LUT Lookup)

| Property | Value |
|---------|--------|
| Input | Token IDs (int32) |
| Output Dim | 896 (LLM embedding dimension) |
| LUT File | `embedding_int8_lut.bin` |
| Lookup Method | Direct index into embedding table |
| Runtime Tool | `qnn-net-run --backend libQnnHtp.so` |

### 4. LLM Decoder (Qwen2-based)

| Property | Value |
|---------|--------|
| Architecture | Qwen2 (transformer decoder) |
| Layers | 24 |
| Attention Heads | 14 |
| KV Heads | 2 (grouped query attention) |
| Embedding Dim | 896 |
| Context Length | 512-4096 |
| Context Binary | `fastvlm_full_decoder_htp.bin` |
| Runtime Tool | `genie-t2t-run` |
| Backend | `QnnHtp` (NPU) |

---

## Memory Usage

### VTCM (On-NPU SRAM) Allocation

```
Total VTCM: ~12MB (Snapdragon 8 Elite)
├── Model weights (decoder): ~8MB
├── KV Cache (full context): ~3MB
└── Scratch space: ~1MB
```

### System RAM (DDR) Usage

```
Total: ~600MB for all binaries and buffers
├── Vision encoder binary: ~100MB
├── Text embedder binary: ~50MB
├── LLM decoder binary: ~400MB (may be split into segments)
├── LUT (embedding table): ~50MB
└── Runtime buffers: ~10MB
```

---

## Performance Targets

| Metric | Target | Notes |
|---------|--------|--------|
| Vision Encoding | < 50ms | 1024x1024 image, INT8 |
| Text Embedding | < 20ms | Tokenization + LUT lookup |
| First Token (TTFB) | < 100ms | Includes prefill latency |
| Token Generation | > 10 tokens/s | Sustained generation rate |
| Total Latency | < 5s | For 50-token response |

---

## Key Design Decisions

### 1. Vision Tokens First in Sequence

**Critical**: Vision embeddings must come before text embeddings in the combined sequence.

```
Correct:  [vision_token_0, ..., vision_token_N, text_token_0, ..., text_token_M]
Wrong:    [text_token_0, ..., text_token_M, vision_token_0, ..., vision_token_N]
```

### 2. Projection Layer Compiled with Decoder

The projection layer that maps 1024-dim vision embeddings to 896-dim LLM space is compiled **into the decoder binary**. This ensures:
- Single inference pass (vision → projection → attention)
- Better quantization (full graph aware)
- Faster execution (no separate projection step)

### 3. KV Cache in VTCM

KV cache is placed in VTCM for ultra-fast access during generation:
- Key cache: `past_key_layer*_in` tensors
- Value cache: `past_value_layer*_out` tensors
- Updates: POINTER_SHIFT mode (no memcpy, just pointer offset changes)

### 4. Variant System for Different Sequence Lengths

Multiple compiled graph variants for efficiency:
- **AR-1**: For autoregressive decode (1 new token)
- **AR-32**: For medium prefill (32 tokens)
- **AR-128**: For large prefill (128 tokens)
- **BERT**: For full context processing

GENIE automatically switches variants based on `n_past` (number of tokens in cache).

---

## Data Types and Quantization

### Vision Encoder
- **Input**: uint8 (INT8, no normalization needed on-device)
- **Weights**: INT8 quantized
- **Output**: FP32 or FP16 (depending on configuration)

### Text Embedder
- **Input**: int32 (token IDs)
- **LUT**: INT8 or FP16 quantized embedding table
- **Output**: FP32 (matches LLM input)

### LLM Decoder
- **Input**: FP32 embeddings
- **Weights**: W4A16 (4-bit weights, 16-bit activations) or INT8
- **KV Cache**: FP16 (for accuracy) or INT8 (for memory)
- **Output**: int32 (logits, sampled to token ID)

---

## GENIE's Role

GENIE orchestrates the entire flow:

1. **Pipeline Creation**: Connects ImageEncoder and TextGenerator nodes
2. **Accumulation**: Buffers vision + text embeddings
3. **Backend Management**: Switches between QNN backends (HTP NPU)
4. **KV Cache Automation**: Updates K/V tensors after each generated token
5. **Streaming Output**: Fires callbacks as tokens are generated

**GENIE does NOT execute the model directly** - it manages the QNN runtime which talks to the HTP.

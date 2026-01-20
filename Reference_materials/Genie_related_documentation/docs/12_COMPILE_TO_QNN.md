# Compile FastVLM to QNN Context Binaries

## Overview

After exporting FastVLM to ONNX, compile to **QNN context binaries** that run on HTP NPU.

## QNN Compilation Workflow

```
ONNX Model (Export)
    │
    ▼ qnn-onnx-converter
QNN Intermediate (converted.json)
    │
    ▼ qnn-context-binary-generator
Context Binary (.bin)
    │
    ▼ [Device]
Inference on HTP
```

## qnn-onnx-converter

Convert ONNX to QNN format and identify any unsupported operations.

```bash
qnn-onnx-converter \
    --input_network fastvlm_decoder_with_projection.onnx \
    --output_path converted_qnn.json \
    --out_node decoder_output

# Common flags:
--backend QNN_HTOP_LIB        # Target HTP backend
--op_package_version <version>  # QNN version
--verbose                     # Show detailed conversion info
```

**Output**: `converted_qnn.json` - Graph in QNN IR.

Check for errors in output:
```
✓ Conversion successful
⚠ Operator "Gather" converted to "GatherElements"
✗ Operator "Complex" not supported - ERROR
```

## qnn-context-binary-generator

Generate context binaries from QNN IR or directly from ONNX.

### Basic Compilation

```bash
qnn-context-binary-generator \
    --model fastvlm_decoder_with_projection.onnx \
    --backend libQnnHtp.so \
    --htp_arch v79 \
    --binary_file fastvlm_decoder_htp.bin
```

### Vision Encoder Compilation

```bash
qnn-context-binary-generator \
    --model fastvlm_vision_encoder.onnx \
    --backend libQnnHtp.so \
    --htp_arch v79 \
    --binary_file fastvlm_vision_encoder_int8_htp.bin \
    
    # Quantization
    --quantize_model \
    --input_file calibration_images.raw \
    
    # Output precision
    --output_datatype float16
```

### Text Embedder Compilation

```bash
qnn-context-binary-generator \
    --model fastvlm_token_embedder.onnx \
    --backend libQnnHtp.so \
    --htp_arch v79 \
    --binary_file fastvlm_embed_tokens_fp16_htp.bin \
    --output_datatype float16
```

### Compilation with Variants

Generate multiple graph variants for different sequence lengths:

```bash
# Single token decode (autoregressive)
qnn-context-binary-generator \
    --model fastvlm_decoder.onnx \
    --backend libQnnHtp.so \
    --variant ar1 \
    --binary_file decoder_ar1.bin

# 32-token prefill
qnn-context-binary-generator \
    --model fastvlm_decoder.onnx \
    --backend libQnnHtp.so \
    --variant ar32 \
    --binary_file decoder_ar32.bin

# 128-token prefill
qnn-context-binary-generator \
    --model fastvlm_decoder.onnx \
    --backend libQnnHtp.so \
    --variant ar128 \
    --binary_file decoder_ar128.bin

# Full context (BERT-style)
qnn-context-binary-generator \
    --model fastvlm_decoder.onnx \
    --backend libQnnHtp.so \
    --variant bert \
    --binary_file decoder_bert.bin
```

### Quantization

Quantize weights to reduce memory footprint:

```bash
# Calibration dataset (representative inputs)
python generate_calibration_data.py \
    --model-dir /path/to/fastvlm \
    --num-samples 100 \
    --output calibration_images.raw

# Compile with quantization
qnn-context-binary-generator \
    --model fastvlm_decoder.onnx \
    --backend libQnnHtp.so \
    --quantize_model \
    --input_file calibration_images.raw \
    --quantization_type int8 \
    --quantization_encoding symmetric \
    --binary_file fastvlm_decoder_int8_htp.bin
```

### VTCM Configuration

For Snapdragon 8 Elite (V79), configure VTCM:

```bash
qnn-context-binary-generator \
    --model fastvlm_decoder.onnx \
    --backend libQnnHtp.so \
    --htp_arch v79 \
    
    # VTCM settings
    --vtcm_size 1048576 \
    --vtcm_window_enable \
    --vtcm_align 128 \
    
    # Performance
    --performance_profile high_performance
```

### Graph Optimization

Enable optimizations for better performance:

```bash
qnn-context-binary-generator \
    --model fastvlm_decoder.onnx \
    --backend libQnnHtp.so \
    
    # Optimization level
    --optimize_level 3
    
    # HTP-specific optimizations
    --enable_htp_float16_optimizations \
    --enable_htp_weight_quantization
```

## Key Compilation Flags

### Required Flags

| Flag | Description |
|-------|-------------|
| `--model` | Input ONNX model path |
| `--backend` | Target backend (libQnnHtp.so) |
| `--htp_arch` | HTP architecture (v68, v73, v75, v79) |
| `--binary_file` | Output context binary path |

### Input Type Flags

| Flag | Use Case |
|-------|----------|
| `--input_tokens` | Standard LLM with token IDs |
| `--input_embeddings` | VLM with pre-computed embeddings |
| `--input_tokens_and_embeddings` | Multimodal (both inputs) |

### Variant Flags

| Flag | Sequence | Use Case |
|-------|-----------|----------|
| `--variant ar1` | 1 token | Autoregressive decode |
| `--variant ar32` | 32 tokens | Small prefill |
| `--variant ar128` | 128 tokens | Medium prefill |
| `--variant bert` | Full context | Initial prompt |

### Quantization Flags

| Flag | Description |
|-------|-------------|
| `--quantize_model` | Enable weight quantization |
| `--input_file` | Calibration data for quantization |
| `--quantization_type` | int8, uint8, float16, etc. |
| `--quantization_encoding` | symmetric or asymmetric |

### Memory Flags

| Flag | Description |
|-------|-------------|
| `--spill-fill-bufsize` | Spill/fill buffer size (bytes) |
| `--vtcm_size` | VTCM allocation size (bytes) |
| `--vtcm_window_enable` | Enable VTCM window (V79+) |
| `--use-mmap` | Memory-map model files |

## Output Artifacts

After compilation, you'll have:

```
fastvlm_qnn_artifacts/
├── fastvlm_vision_encoder_int8_htp.bin      # Vision encoder (~100MB)
├── fastvlm_embed_tokens_fp16_htp.bin        # Text embedder (~50MB)
├── fastvlm_decoder_ar1_htp.bin              # AR-1 variant (~100MB)
├── fastvlm_decoder_ar32_htp.bin             # AR-32 variant (~200MB)
├── fastvlm_decoder_ar128_htp.bin            # AR-128 variant (~300MB)
├── fastvlm_decoder_bert_htp.bin            # BERT variant (~400MB)
├── embedding_int8_lut.bin                    # LUT (~50MB)
└── tokenizer.json                             # Tokenizer
```

## Verification

### Inspect Context Binary

```bash
# Get graph info
qnn-context-binary-utility \
    --context_binary fastvlm_decoder_ar1_htp.bin \
    --json_file decoder_info.json

# Output includes:
# - Graph metadata
# - Input tensors
# - Output tensors
# - KV cache tensors
# - Quantization parameters
# - Memory requirements
```

### Check Tensor Shapes

```bash
# Parse JSON
cat decoder_info.json | jq '.info.graphs[0].graphOutputs[0]'
```

Verify:
- Input dimensions match model architecture
- Output logits dimension matches vocabulary
- KV cache dimensions are correct

## Performance Tuning

### Reduce Binary Size

If decoder binary too large (>1GB):

```bash
# 1. Split into multiple variants
--variant ar1 --variant ar32 --variant ar128

# 2. Use higher quantization
--quantization_type int8

# 3. Reduce context size
--max_seq_len 2048

# 4. Enable compression
--compress_weights
```

### Improve Latency

```bash
# 1. Enable all HTP optimizations
--enable_htp_float16_optimizations \
--enable_htp_weight_quantization \
--optimize_level 3

# 2. Configure VTCM properly
--vtcm_size 1048576 \
--vtcm_window_enable

# 3. Use proper variant at runtime
# GENIE automatically switches AR-n variants
```

## Troubleshooting

### "Graph Load Failed"

**Cause**: Architecture mismatch
```bash
# Check architecture
cat decoder_info.json | jq '.info.architecture'

# Solution: Recompile with correct --htp_arch
qnn-context-binary-generator \
    --model decoder.onnx \
    --backend libQnnHtp.so \
    --htp_arch v79  # Ensure this matches device
```

### "Out of Memory on NPU"

**Cause**: VTCM budget exceeded

```bash
# Solution: Reduce VTCM allocation
--vtcm_size 524288  # Half from 1MB

# Or reduce context size
# Edit ONNX or compile with smaller sequence
```

### "Quantization Error"

**Cause**: Calibration data not representative

```bash
# Solution: Generate better calibration data
python generate_calibration.py \
    --model-dir /path/to/fastvlm \
    --num-samples 1000 \
    --dataset coco_val2017  # Use real images
```

## Complete Compilation Script

```bash
#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="/path/to/fastvlm"
OUTPUT_DIR="./fastvlm_qnn_artifacts"
mkdir -p "${OUTPUT_DIR}"

# 1. Convert to QNN
echo "Converting to QNN format..."
qnn-onnx-converter \
    --input_network "${MODEL_DIR}/fastvlm_decoder_with_projection.onnx" \
    --output_path "${OUTPUT_DIR}/converted.json" \
    --backend QNN_HTOP_LIB

# 2. Compile vision encoder (INT8)
echo "Compiling vision encoder (INT8)..."
qnn-context-binary-generator \
    --model "${MODEL_DIR}/fastvlm_vision_encoder.onnx" \
    --backend libQnnHtp.so \
    --htp_arch v79 \
    --binary_file "${OUTPUT_DIR}/fastvlm_vision_encoder_int8_htp.bin" \
    --quantize_model \
    --input_file "${MODEL_DIR}/calibration_images.raw" \
    --quantization_type int8

# 3. Compile text embedder (FP16)
echo "Compiling text embedder (FP16)..."
qnn-context-binary-generator \
    --model "${MODEL_DIR}/fastvlm_token_embedder.onnx" \
    --backend libQnnHtp.so \
    --htp_arch v79 \
    --binary_file "${OUTPUT_DIR}/fastvlm_embed_tokens_fp16_htp.bin" \
    --output_datatype float16

# 4. Compile decoder variants
echo "Compiling decoder variants..."

for variant in ar1 ar32 ar128 bert; do
    echo "  Compiling ${variant} variant..."
    qnn-context-binary-generator \
        --model "${MODEL_DIR}/fastvlm_decoder_with_projection.onnx" \
        --backend libQnnHtp.so \
        --htp_arch v79 \
        --variant ${variant} \
        --binary_file "${OUTPUT_DIR}/fastvlm_decoder_${variant}_htp.bin" \
        --vtcm_size 1048576 \
        --vtcm_window_enable \
        --optimize_level 3
done

# 5. Copy LUT and tokenizer
echo "Copying LUT and tokenizer..."
cp "${MODEL_DIR}/embedding_int8_lut.bin" "${OUTPUT_DIR}/"
cp "${MODEL_DIR}/tokenizer.json" "${OUTPUT_DIR}/"

echo "Compilation complete! Artifacts in ${OUTPUT_DIR}"
```

## Next Steps

1. **Verify binaries**: Use `qnn-context-binary-utility` to inspect
2. **Configure GENIE**: See [13_GENIE_CONFIGURATION.md](./13_GENIE_CONFIGURATION.md)
3. **Deploy to device**: Push binaries and test

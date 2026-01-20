# Troubleshooting Guide

## Common Issues and Solutions

## Model Export Issues

### Issue: ONNX Export Fails with "Dynamic Dimension Error"

**Symptom**:
```
RuntimeError: Exporting the model failed. Could not export dynamic shape
```

**Cause**: Vision encoder has dynamic image size but ONNX expects fixed.

**Solution**:
```python
# Fix in export script
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    dynamic_axes={
        "pixel_values": {0: "batch"}  # Allow dynamic batch only
    }
)
```

### Issue: Projection Dimension Mismatch

**Symptom**:
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (256x1024 and 256x896)
```

**Cause**: Projection layer doesn't match LLM embedding dimension.

**Solution**:
```python
# Verify model architecture
assert model.visual_projection.out_features == model.config.hidden_size
print(f"Vision output: {model.visual_projection.out_features}")
print(f"LLM input: {model.config.hidden_size}")

# If mismatch, add projection layer
if model.visual_projection.out_features != model.config.hidden_size:
    model.visual_projection = nn.Linear(model.visual_projection.out_features, 
                                      model.config.hidden_size)
```

### Issue: Unsupported ONNX Operator

**Symptom**:
```
QNN Error: Operator 'FlashAttention' not supported
```

**Cause**: Custom operators not in QNN operator set.

**Solution**:
```python
# Replace with QNN-supported ops
# Bad: custom Flash Attention
x = flash_attention(q, k, v)

# Good: standard attention
x = scaled_dot_product_attention(q, k, v)
```

## QNN Compilation Issues

### Issue: "Graph Load Failed"

**Symptom**:
```
ERROR: Failed to load context binary
```

**Cause**: Architecture mismatch between compilation and device.

**Solution**:
```bash
# Check compiled architecture
qnn-context-binary-utility --context_binary decoder.bin | jq '.info.architecture'

# Recompile for correct architecture
qnn-context-binary-generator \
    --model decoder.onnx \
    --backend libQnnHtp.so \
    --htp_arch v79  # Ensure matches device
```

### Issue: "VTCM Budget Exceeded"

**Symptom**:
```
ERROR: VTCM allocation failed, budget too small
```

**Cause**: Requested VTCM larger than available on NPU.

**Solution**:
```bash
# Reduce VTCM allocation
qnn-context-binary-generator \
    --vtcm_size 524288  # 512KB instead of 1MB

# Or reduce model size
# Split into multiple binaries
--variant ar1 --variant ar32
```

### Issue: "Out of Memory on NPU During Inference"

**Symptom**:
```
ERROR: OOM during graph execution
```

**Cause**: Model + KV cache + activations exceed NPU memory.

**Solutions**:

1. Reduce context size:
```json
{
  "context": {"size": 2048}  // Half from 4096
}
```

2. Enable sliding window:
```json
{
  "longcontext": {
    "type": "sliding-window",
    "sliding-window": {"window-size": 4096}
  }
}
```

3. Use INT8 quantization:
```bash
qnn-context-binary-generator \
    --quantization_type int8
```

### Issue: Quantization Loss

**Symptom**: Generated text quality degrades after quantization.

**Cause**: Poor calibration data or aggressive quantization.

**Solutions**:

1. Use representative calibration data:
```bash
python generate_calibration.py \
    --dataset coco_val2017 \
    --num-samples 1000 \
    --output calibration.raw
```

2. Use higher precision for critical layers:
```bash
# Keep projection in FP16
--output_datatype float16
```

3. Adjust quantization parameters:
```bash
--quantization_encoding symmetric
--per_channel_quantization
```

## Runtime Issues

### Issue: "Embedding Size Mismatch"

**Symptom**:
```
ERROR: Embedding size 524288 doesn't match accumulator size 400000000
```

**Cause**: Combined embeddings exceed configured accumulator buffer.

**Solution**:
```json
{
  "text-generator": {
    "accumulator-size": 600000000  // Increase to fit
  }
}
```

### Issue: KV Cache Not Updating

**Symptom**: Every generation step recomputes from scratch (slow).

**Cause**: KV cache mode not configured properly.

**Solution**:
```json
{
  "QnnHtp": {
    "kv-update-method": "pointer-shift"  // Ensure this is set
  }
}
```

### Issue: "Vision Tokens Not Recognized"

**Symptom**: Text generator produces garbage after vision embeddings.

**Cause**: Vision embeddings concatenated after text (wrong order).

**Solution**:
```python
# CRITICAL: Vision tokens MUST come first
# Wrong:
combined = torch.cat([text_emb, vision_emb], dim=1)

# Correct:
combined = torch.cat([vision_emb, text_emb], dim=1)
```

### Issue: Streaming Not Working

**Symptom**: No output until generation completes.

**Cause**: Not using streaming callbacks or wrong sentence codes.

**Solution**:
```cpp
// Use proper callback signature
void callback(const char* text, 
             GenieDialog_SentenceCode_t code,
             const void* userData) {
    // Handle streaming codes
    switch (code) {
        case GENIE_DIALOG_SENTENCE_BEGIN:
            printf("[START] %s", text);
            break;
        case GENIE_DIALOG_SENTENCE_CONTINUE:
            printf("[MORE] %s", text);
            break;
        case GENIE_DIALOG_SENTENCE_END:
            printf("[DONE] %s\n", text);
            break;
    }
}

// Request streaming
GenieDialog_query(dialog, prompt,
    GENIE_DIALOG_SENTENCE_BEGIN,  // Request streaming
    callback, userData);
```

## Device Issues

### Issue: "Library Load Failed"

**Symptom**:
```
ERROR: Failed to load libQnnHtp.so
```

**Cause**: Wrong library architecture or missing dependencies.

**Solution**:
```bash
# Push correct architecture
adb push libQnnHtp.so /data/local/tmp/vlm/
adb push libQnnHtpV79Skel.so /data/local/tmp/vlm/

# Set LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/data/local/tmp/vlm:$LD_LIBRARY_PATH
```

### Issue: "Context Binary Not Found"

**Symptom**:
```
ERROR: Cannot open context binary: No such file
```

**Cause**: Wrong path or file not pushed.

**Solution**:
```bash
# Verify files on device
adb shell "ls -lh /data/local/tmp/vlm/*.bin"

# Push if missing
adb push fastvlm_decoder_htp.bin /data/local/tmp/vlm/
```

### Issue: "Segmentation Fault"

**Symptom**: App crashes immediately.

**Cause**: Null pointer, uninitialized data, or API misuse.

**Solution**:
```cpp
// Check all pointers before use
if (dialogHandle == nullptr) {
    printf("ERROR: Dialog not created\n");
    return -1;
}

// Validate data before passing
if (embeddings == nullptr || embeddingsSize == 0) {
    printf("ERROR: Invalid embeddings\n");
    return -1;
}
```

## Performance Issues

### Issue: Slow First Token Latency

**Symptom**: TTFB > 500ms.

**Causes**:
1. Prefill processing entire sequence
2. KV cache not warmed
3. No graph switching

**Solutions**:

1. Enable graph switching:
```json
{
  "QnnHtp": {"enable-graph-switching": true}
}
```

2. Use prefill variant:
```bash
# Compile AR-32 variant for prefill
--variant ar32
```

3. Optimize quantization:
```bash
--quantize_model \
--quantization_type int8
```

### Issue: Slow Token Generation Rate

**Symptom**: < 5 tokens/s.

**Causes**:
1. Inefficient KV update (shift-concat instead of pointer-shift)
2. No VTCM usage
3. CPU fallback

**Solutions**:

1. Use pointer-shift:
```json
{
  "QnnHtp": {"kv-update-method": "pointer-shift"}
}
```

2. Enable VTCM:
```bash
--vtcm_size 1048576 \
--vtcm_window_enable
```

3. Verify NPU execution:
```bash
# Check backend
adb shell "cat /proc/hvx/summary"

# Ensure QnnHtp is being used
# (not QnnGenAiTransformer which is CPU)
```

## Debugging

### Enable GENIE Logging

```bash
export GENIE_LOG_LEVEL=DEBUG
./genie-t2t-run --config config.json
```

### Enable QNN Logging

```bash
export QNN_LOG_LEVEL=3
./genie-t2t-run --config config.json
```

### Enable Profiling

```bash
./genie-t2t-run \
    --config config.json \
    --profile profile.json

# Analyze profile
cat profile.json | jq '.components[0].events'
```

### Inspect Context Binary

```bash
qnn-context-binary-utility \
    --context_binary decoder.bin \
    --json_file info.json

# Check tensor shapes
cat info.json | jq '.info.graphs[0].graphInputs'
cat info.json | | jq '.info.graphs[0].graphOutputs'

# Check memory requirements
cat info.json | jq '.info.graphs[0].memoryRequirements'
```

## Verification Checklist

Before deploying, verify:

- [ ] Model exported to ONNX successfully
- [ ] ONNX has correct input/output shapes
- [ ] QNN compilation completed without errors
- [ ] Context binary loads on device
- [ ] LUT file format correct (FP32 or INT8)
- [ ] Tokenizer JSON valid
- [ ] GENIE configuration validates
- [ ] Accumulator size sufficient for embeddings
- [ ] KV cache dimensions match model
- [ ] Vision tokens precede text tokens in combined sequence
- [ ] HTP backend selected (not CPU)
- [ ] All binaries pushed to device
- [ ] LD_LIBRARY_PATH set correctly

## Common Error Messages and Meanings

| Error | Meaning | Solution |
|-------|-----------|----------|
| `GENIE_STATUS_ERROR_JSON_SCHEMA` | Invalid config JSON | Check JSON syntax |
| `GENIE_STATUS_ERROR_INVALID_ARGUMENT` | Invalid API arguments | Verify function parameters |
| `GENIE_STATUS_ERROR_MEM_ALLOC` | Memory allocation failed | Increase memory or reduce model size |
| `GENIE_STATUS_ERROR_NOT_SUPPORTED` | Feature not supported | Check backend/version compatibility |
| `GENIE_STATUS_ERROR_GENERAL` | Generic error | Enable logging and check logs |
| `ContextLimitException` | Context length exceeded | Reduce prompt length or increase context size |

## Getting Help

1. **Check documentation**: See [00_ARCHITECTURE_OVERVIEW.md](./00_ARCHITECTURE_OVERVIEW.md)
2. **Review logs**: `adb logcat -d QNN:*` for QNN errors
3. **Check GENIE logs**: Look for `GENIE_LOG_LEVEL` output
4. **Verify binaries**: Use `qnn-context-binary-utility` to inspect
5. **Test on different devices**: Try on Snapdragon 8 Gen 2 vs 8 Elite

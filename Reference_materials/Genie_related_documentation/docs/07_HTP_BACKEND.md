# HTP Backend - Complete Reference

## Overview

HTP (Hexagon Tensor Processor) is Qualcomm's **NPU** (Neural Processing Unit). It accelerates matrix operations for deep learning models.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│            Snapdragon 8 Elite SoC                    │
├─────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────┐    │
│  │  Application Processor (CPU)               │    │
│  │  - ARM Cortex-X4 Prime                  │    │
│  │  - Host OS, Android                    │    │
│  └───────────────────────────────────────────────┘    │
│          │                                            │
│          ▼                                            │
│  ┌───────────────────────────────────────────────┐    │
│  │  Hexagon NPU (HTP)                        │    │
│  │  ┌───────────────────────────────────────┐    │    │
│  │  │  HVX (Vector) Units             │    │
│  │  │  - 128-bit SIMD                  │    │
│  │  │  - MAC operations                │    │
│  │  └───────────────────────────────────────┘    │    │
│  │  ┌───────────────────────────────────────┐    │    │
│  │  │  Tensor Accelerators              │    │
│  │  │  - Matrix multiply            │    │    │
│  │  │  - Activation functions       │    │    │
│  │  └───────────────────────────────────────┘    │    │
│  │                                                  │    │
│  │  ┌───────────────────────────────────────┐    │    │
│  │  │  VTCM (On-NPU SRAM)           │    │
│  │  │  - ~8-16MB ultra-fast memory   │    │    │
│  │  │  - Multiple models share this     │    │    │
│  │  └───────────────────────────────────────┘    │    │
│  └───────────────────────────────────────────────┘    │
│          │                                            │
│          ▼                                            │
│  ┌───────────────────────────────────────────────┐    │
│  │  System RAM (DDR)                      │    │
│  │  - ~8-16GB total                     │    │
│  │  - Slower than VTCM                   │    │
│  └───────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

## HTP Versions

| Version | Chip Generation | Architecture |
|---------|----------------|-------------|
| V68 | Snapdragon 8 Gen 1 | HVX v68 |
| V73 | Snapdragon 8 Gen 2 | HVX v73 |
| V75 | Snapdragon 8 Gen 3 | HVX v75 |
| V79 | Snapdragon 8 Elite | HVX v79, VTCM Window |

**Snapdragon 8 Elite uses V79** - includes VTCM Window feature.

## QNN HTP Backend Integration

### Backend Loading

```cpp
// From qualla/engines/qnn-htp/nsp-base-model.cpp

// 1. Load HTP library
std::shared_ptr<DynLib> _backendLib = std::make_shared<DynLib>("libQnnHtp.so");

// 2. Get QNN interface
QnnInterface_t* interface = _backendLib->getSymbol<QnnInterface_t*>(
    "QnnInterface_getProviders");

// 3. Create HTP backend
interface->backendCreate(
    &_backendHandle,          // QnnBackend_Handle_t
    &logHandle,
    &backendConfig            // QnnHtpBackend_CustomConfig_t
);
```

### Context Binary Loading

```cpp
// Load compiled model binary
interface->contextCreateFromBinary(
    _backendHandle,
    &logHandle,
    &context,
    modelBinary,      // fastvlm_full_decoder_htp.bin
    &contextConfig    // QnnHtpContext_CustomConfig_t
);
```

### VTCM Configuration

```cpp
// Configure VTCM window (V79+ feature)
QnnHtpGraph_Config_t graphConfig;

graphConfig.vtcmmConfig.enableVtcmm = true;
graphConfig.vtcmmConfig.vtcmmBudget = 0x100000;  // ~1MB

interface->graphConfig(
    _graphHandle,
    &graphConfig
);
```

## QNN HTP Configuration Options

### Backend Configuration

```json
{
  "QnnHtp": {
    "version": 1,
    
    // Memory management
    "spill-fill-bufsize": 0,      // Spill/fill buffer size (bytes)
    "use-mmap": true,               // Memory-map model files
    "mmap-budget": 0,               // mmap budget (bytes)
    
    // Execution
    "poll": false,                   // Polling mode vs blocking
    
    // CPU affinity
    "cpu-mask": "0xe0",           // CPU cores for DSP
    
    // KV cache
    "kv-dim": 128,                  // KV head dimension
    "kv-update-method": "pointer-shift",  // or "shift-concat", "smart-mask"
    
    // Performance
    "enable-graph-switching": true,   // Enable AR-n variants
    "allow-async-init": true,       // Async context initialization
    
    // Memory alignment
    "data-alignment-size": 128,     // Memory alignment (bytes)
    
    // Advanced
    "rope-theta": 10000.0,           // RoPE theta
    "pos-id-dim": 64                   // Position ID dimension
    "skip-lora-validation": false     // Skip LoRA validation (dev)
    
    // VTCM (V79+)
    "shared-engine": false               // Share engine between graphs
    "graph-switching-lora-policy": "lazy"  // or "eager"
  }
}
```

### Context Configuration

```json
{
  "context": {
    "version": 1,
    "graph-name": "fastvlm_decoder",
    
    // Memory
    "graph-state-size": <size_in_bytes>,
    "mem-const-size": <size_in_bytes>,
    
    // VTCM
    "vtcm-size": <vtcm_allocation>,
    "vtcm-size-align": <alignment>,
    
    // Performance
    "performance-profile": "high_performance" | "balanced" | "power_saving",
    
    // Precision
    "precision": "float16" | "int8",
    "precision-default-type": "float16" | "int8"
  }
}
```

## Compilation Flags

### qnn-context-binary-generator

```bash
qnn-context-binary-generator \
    --model fastvlm_decoder.onnx \
    --backend libQnnHtp.so \
    --htp_arch v79 \
    --binary_file fastvlm_decoder_htp.bin \
    
    # Quantization
    --quantize_model \
    --input_file calibration_data.raw \
    
    # Graph optimization
    --optimize_level 3 \
    --enable_htp_float16_optimizations \
    
    # VTCM
    --vtcm_size 1048576 \
    --vtcm_window_enable \
    
    # Variants
    --variant ar1 \
    --variant ar32 \
    --variant ar128 \
    --variant bert
```

### Model Input Types

```bash
# Token input (standard LLM)
--input_tokens

# Embedding input (VLM)
--input_embeddings

# Both (multi-modal)
--input_tokens_and_embeddings
```

## Performance Tuning

### Graph Switching

For models that need different graph layouts at different sequence lengths:

```json
{
  "QnnHtp": {
    "enable-graph-switching": true
  }
}
```

GENIE automatically switches between:
- AR-1: Single token decode
- AR-32: 32-token prefill
- AR-128: 128-token prefill
- BERT: Full context

### KV Update Methods

| Method | Memory Movement | Latency | Use Case |
|---------|---------------|----------|-----------|
| pointer-shift | None (pointer arithmetic) | Fastest | Production V79+ |
| shift-concat | memcpy of ~KB | Medium | Older hardware |
| smart-mask | None (mask update) | Low | Fixed sequence length |

### LoRA Support

```json
{
  "QnnHtp": {
    "graph-switching-lora-policy": "lazy" | "eager"
  },
  "lora": {
    "version": 1,
    "adapters": [
      {
        "name": "adapter1",
        "alphas": ["tensor_alpha"],
        "bin-sections": ["section1.bin", "section2.bin"]
      }
    ]
  }
}
```

## Memory Layout

### System RAM (DDR)

```
Model Binary (mmap'd):
├── Graph metadata
├── Weights (INT8/W4A16 quantized)
└── Variants (AR-1, AR-32, AR-128, BERT)

Runtime Allocations:
├── KV Cache (full context)
│   ├── Key cache [layers, n_heads, seq_len, kv_dim]
│   └── Value cache [layers, n_heads, seq_len, kv_dim]
├── Input tensors
├── Output tensors
└── Scratch buffers
```

### VTCM (On-NPU SRAM)

```
VTCM Partition (V79+):
├── Active graph weights (~8MB)
├── Active KV cache (~2-4MB)
└── Scratch space (~1MB)

VTCM Window Feature:
- Allows multiple models to allocate disjoint regions
- Prevents fragmentation
- Faster context switches
```

## Profiling

### Enable Profile

```bash
./genie-t2t-run \
    --config vlm_config.json \
    --profile profile.json
```

### Profile Metrics

```json
{
  "components": [
    {
      "events": [
        {
          "type": "GenieDialog_query",
          "time-to-first-token": {"value": 120, "unit": "ms"},
          "prompt-processing-rate": {"value": 850, "unit": "tokens/s"},
          "token-generation-rate": {"value": 25, "unit": "tokens/s"},
          "num-prompt-tokens": {"value": 50},
          "num-generated-tokens": {"value": 30},
          "token-generation-time": {"value": 1200, "unit": "ms"}
        }
      ]
    }
  ]
}
```

## Troubleshooting HTP Issues

### "Graph Load Failed"

**Causes**:
- Wrong architecture (compiled for V68, running on V79)
- VTCM budget exceeded
- Missing required variant

**Solutions**:
```bash
# Check architecture
qnn-context-binary-utility --context_binary model.bin

# Reduce VTCM allocation
--vtcm_size 524288  # Smaller budget

# Ensure all variants present
--variant ar1 --variant ar32 --variant ar128 --variant bert
```

### "KV Cache Overflow"

**Causes**:
- Context length too large
- KV cache not configured

**Solutions**:
```json
{
  "context": {"size": 2048},  // Reduce from 4096
  "QnnHtp": {"kv-update-method": "smart-mask"}  // Use mask instead of buffer
}
```

### "Out of Memory on NPU"

**Causes**:
- Model too large for VTCM
- Multiple active graphs

**Solutions**:
```bash
# Reduce context size
--max_seq_len 2048

# Enable spill/fill
--spill_fill_buf_size 1048576

# Use quantized weights
--quantize_model --input_file calibration.raw
```

## Key Files

| File | Purpose |
|------|---------|
| `libQnnHtp.so` | HTP backend library |
| `libQnnHtpV79Skel.so` | V79 DSP skeleton |
| `libQnnHtpPrepare.so` | Graph preparation |
| `qnn-context-binary-generator` | Compilation tool |
| `qnn-context-binary-utility` | Inspection tool |
| `qnn-net-run` | Inference runner (low-level) |
| `genie-t2t-run` | GENIE runner (uses HTP backend) |

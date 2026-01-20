# KV Cache Management on NPU

The KV (Key-Value) cache is critical for efficient autoregressive transformer inference. This document explains how GENIE manages KV cache on the Qualcomm HTP NPU.

## Source Files

| File | Purpose |
|------|---------|
| `nsp-kvmanager.cpp` | Core KV cache logic for HTP |
| `nsp-kvmanager.hpp` | KV cache structures and modes |
| `nsp-model.cpp` | Model execution with KV cache |

**Primary Location:**
```
/home/aryamavmurthy/work/QIDK/VLM_on_edge/VLM_run_experiments_QIDK/fastvlm_yashas/2.42.0.251225/examples/Genie/Genie/src/qualla/engines/qnn-htp/
```

## What is KV Cache?

In transformer attention:
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

During autoregressive generation:
- **Without cache**: Recompute K and V for all previous tokens every step
- **With cache**: Store K and V from previous steps, only compute for new token

**Memory savings**: O(n²) → O(n) per step

## KV Cache Tensor Naming Convention

GENIE parses tensor names to identify cache tensors:
```
past_{key,value}_{layer_idx}[_h{head_idx}]_{in,out}
```

**Examples:**
```
past_key_0_in       # Key cache for layer 0, input
past_value_5_out    # Value cache for layer 5, output
past_key_12_h3_in   # Key cache for layer 12, head 3, input
```

## KVCache Data Structure

```cpp
// From nsp-kvmanager.hpp
struct KVCache {
    bool is_key;           // true for key, false for value
    char* buffer;          // Persistent cache buffer (input to model)
    char* output_buffer;   // Output buffer (new K/V from model)
    int32_t n_heads;       // Number of attention heads
};
```

## KV Manager Modes

GENIE supports three update modes:

### 1. POINTER_SHIFT (Most Efficient)
- **Mechanism**: Modify tensor pointers instead of copying data
- **Memory movement**: None (just pointer arithmetic)
- **Best for**: Production inference on NPU
- **How it works**: The QNN API allows updating tensor pointers at runtime

```cpp
// From nsp-kvmanager.cpp
void NewNSPKVManager::registerPointerOffset() {
    _register_pointer_fn(variant, ptr_offset * _bw);
}
```

### 2. SHIFT_CONCAT
- **Mechanism**: Physically shift buffer, append new tokens
- **Memory movement**: `memmove` + `memcpy`
- **Best for**: When pointer shifting not available

```cpp
// Shift existing cache left, append new tokens at end
std::memmove(dst, dst + copy_size, n_iter * iter_size - copy_size);
```

### 3. SMART_MASK
- **Mechanism**: Fixed buffer with attention mask
- **Memory movement**: None (mask defines active region)
- **Best for**: Fixed sequence length scenarios

## Variant System (AR-n)

To handle different sequence lengths efficiently, GENIE uses "variants":

| Variant | Use Case | Cache Dimension |
|---------|----------|-----------------|
| AR-1 | Single token decode | ctx_size - 1 |
| AR-32 | Prefill 32 tokens | ctx_size - 32 |
| AR-128 | Prefill 128 tokens | ctx_size - 128 |
| BERT (ctx_size) | Full context | ctx_size |

**Switching variants** (e.g., from prefill to decode):
```cpp
bool NewNSPKVManager::switchKeyVariant(KVCache cache, int32_t m, int32_t n, int32_t offset) {
    // Realign cache data from AR-m format to AR-n format
    const size_t in_cache_dim  = (m == _n_ctx) ? _n_ctx : _n_ctx - m;
    const size_t out_cache_dim = _n_ctx - n;
    // ... memory movement logic ...
}
```

## Update Flow

### Step 1: Dispatch Update
```cpp
void NewNSPKVManager::dispatchUpdate(int32_t n_past, int32_t variant, 
                                      const std::vector<bool>& selected) {
    // Determine update mode
    if (_req_state.n_past == 0) {
        _req_mode = CLEAR_CACHE;
    } else if (_req_state.n_past == _cur_state.n_past) {
        _req_mode = SET_VARIANT;  // Just switching variant
    } else if (_req_state.n_past < _cur_state.n_past) {
        _req_mode = UPDATE_OUTPUT;  // Rewind (e.g., for speculative decoding)
    } else {
        _req_mode = UPDATE_AND_SET;  // Normal forward
    }
    
    // Launch parallel updates
    _threadpool->enqueue(_update_jobs);
}
```

### Step 2: Update Key Cache
```cpp
bool NewNSPKVManager::updateKey(KVCache cache, int32_t variant, 
                                 int32_t n_past, int32_t n_update, ...) {
    char* dst = cache.buffer;         // Input buffer (persistent)
    char* src = cache.output_buffer;  // Output buffer (from model)
    
    // Calculate positions
    const int32_t iter_size = (_n_ctx - variant) * _bw;
    const int32_t copy_size = abs(n_update) * _bw;
    
    // Concatenate new keys into cache
    for (int32_t i = 0; i < n_iter; i++) {
        std::memcpy(write_ptr, read_ptr, copy_size);
        write_ptr += iter_size;
        read_ptr += out_size;
    }
}
```

### Step 3: Register Pointer (POINTER_SHIFT mode)
```cpp
bool NewNSPKVManager::registerPointerOffset() {
    // Tell QNN API the new pointer offset
    _register_pointer_fn(variant, ptr_offset * _bw);
}
```

## Sliding Window Cache

For long contexts, GENIE supports sliding window attention:

```cpp
// From Engine.cpp/Dialog.cpp
static void validateSlidingWindowConfig(const qualla::json& config) {
    // Required fields
    // - version: 1
    // - window-size: e.g., 4096
}
```

**Configuration:**
```json
{
  "engine": {
    "longcontext": {
      "type": "sliding-window",
      "sliding-window": {
        "version": 1,
        "window-size": 4096
      }
    }
  }
}
```

**Implementation:**
```cpp
// When n_update < 0, shrink the cache (sliding window)
if (n_update < 0) {
    if (_mode == SHIFT_CONCAT) {
        std::memmove(dst + copy_size, dst, n_iter * iter_size - copy_size);
        std::memset(dst, _pad_value, copy_size);
    }
}
```

## Memory Layout

### Key Cache (Left-padded for POINTER_SHIFT)
```
+--------+--------+--------+--------+
| pad    | pad    | key_0  | key_1  |  <- AR-2 variant
+--------+--------+--------+--------+
          ^
          ptr_offset points here

After adding key_2:
+--------+--------+--------+--------+
| pad    | key_0  | key_1  | key_2  |  <- Still AR-2
+--------+--------+--------+--------+
   ^
   ptr_offset shifts left
```

### Value Cache (Block layout per head)
```
+---------------------------+---------------------------+
| Head 0: [val_0, val_1, ...] | Head 1: [val_0, val_1, ...] |
+---------------------------+---------------------------+
```

## VTCM (Vector Tightly Coupled Memory)

For Snapdragon 8 Elite (V79+), GENIE leverages **VTCM windows**:

- **VTCM**: High-speed on-NPU memory (typically 8-16MB)
- **VTCM Window**: Hardware feature allowing multiple AI models to share VTCM
- **KV Cache placement**: Often placed in VTCM for fast access

```
/home/aryamavmurthy/work/QIDK/qcom_ai_stack/docs/HAP_compute_res.md
```

## Threading for KV Updates

Updates are parallelized across CPU threads:

```cpp
// From constructor
for (int idx = 0; idx < n_threads; idx++) {
    _update_jobs.push_back([this, idx] { this->runKVUpdateJob(idx); });
}

// Each thread handles a subset of layers
void NewNSPKVManager::runKVUpdateJob(int thread_idx) {
    int job_count = 1 + ((getNumKVTensors() - 1) / n_threads);
    int end_idx = job_count * (thread_idx + 1);
    
    for (int idx = job_count * thread_idx; idx < end_idx; idx++) {
        KVCache& cache = _kv_cache[idx];
        if (cache.is_key)
            updateKey(cache, ...);
        else
            updateValue(cache, ...);
    }
}
```

## Cache Save/Restore

For resumable sessions:

```cpp
bool NewNSPKVManager::dumpCache(std::ofstream* fs, bool is_key, 
                                 int32_t n_valid, int32_t n_heads) {
    // Write valid portion of cache to file
    for (KVCache& cache : _kv_cache) {
        if (cache.is_key != is_key) continue;
        fs->write(data, copy_size);
    }
}

bool NewNSPKVManager::loadCache(std::ifstream* fs, bool is_key,
                                 int32_t n_valid, int32_t variant, int32_t n_heads) {
    // Read cache from file, pad appropriately
    clearBuffer(cache);
    fs->read(data, copy_size);
}
```

## Quantization for KV Cache

```cpp
// From constructor
if (rt->dtype == QNN_DATATYPE_FLOAT_16)
    _pad_value = 0;  // FP16: pad with zeros
else
    // INT8: pad with quantization offset
    _pad_value = static_cast<uint8_t>(-rt->quantParam[0].offset);
```

**Common configurations:**
- FP16 KV cache: Better accuracy, 2x memory
- INT8 KV cache: Smaller footprint, slight accuracy loss
- W4A16: 4-bit weights, 16-bit activations (for model weights, not KV)

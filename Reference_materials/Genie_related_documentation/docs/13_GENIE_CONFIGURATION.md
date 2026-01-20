# GENIE Configuration Reference

## Overview

Complete JSON configuration reference for GENIE VLM deployment on HTP NPU.

## Configuration Hierarchy

```
Configuration
├── text-generator / image-encoder / text-encoder
│   ├── embedding
│   ├── context
│   ├── tokenizer
│   ├── sampler
│   └── engine
│       ├── backend
│       │   └── QnnHtp / QnnGpu / QnnGenAiTransformer
│       └── model
│           └── binary / library
```

## Text Generator Configuration

### Basic Structure

```json
{
  "text-generator": {
    "version": 1,
    "type": "basic",
    "accumulator-size": 400000000
  }
}
```

### Embedding Configuration

```json
{
  "embedding": {
    "version": 1,
    "type": "lut",
    "lut-path": "embedding_int8_lut.bin",
    "size": 896,
    "datatype": "float32",
    "quant-param": {
      "scale": 1.0,
      "offset": 0
    }
  }
}
```

| Field | Values | Description |
|-------|---------|-------------|
| `type` | `"lut"`, `"callback"` | LUT = lookup table, Callback = custom function |
| `size` | Embedding dimension | LLM embedding dimension (e.g., 896) |
| `datatype` | `"float32"`, `"ufixed8"`, `"sfixed16"` | Data type of embeddings |
| `lut-path` | Path to LUT.bin | Token embedding table file |
| `quant-param` | Scale/offset | Quantization parameters |

### Context Configuration

```json
{
  "context": {
    "version": 1,
    "size": 4096,
    "n-vocab": 151936,
    "bos-token": 151644,
    "eos-token": [151645],
    "pad-token": 151643,
    "n-embd": 896
  }
}
```

| Field | Values | Description |
|-------|---------|-------------|
| `size` | Sequence length | Maximum context (tokens) |
| `n-vocab` | Vocabulary size | Tokenizer vocabulary size |
| `bos-token` | Token ID or -1 | Begin-of-sequence token |
| `eos-token` | Token ID or array | End-of-sequence token (can be multiple) |
| `pad-token` | Token ID | Padding token |
| `n-embd` | Embedding dimension | Alternative to `embedding.size` |

### Tokenizer Configuration

```json
{
  "tokenizer": {
    "version": 1,
    "path": "tokenizer.json"
  }
}
```

### Sampler Configuration

```json
{
  "sampler": {
    "version": 1,
    "type": "basic",
    "seed": 42,
    "temp": 0.8,
    "top-k": 40,
    "top-p": 0.95,
    "greedy": false
  }
}
```

| Field | Range | Description |
|-------|--------|-------------|
| `type` | `"basic"` | Sampling algorithm type |
| `seed` | Any integer | Random seed for reproducibility |
| `temp` | 0.0 to 1.0 | Temperature (0.0 = greedy) |
| `top-k` | 1 to vocab_size | Top-k sampling |
| `top-p` | 0.0 to 1.0 | Nucleus sampling |
| `greedy` | boolean | Override temperature for deterministic output |

### Engine Configuration

```json
{
  "engine": {
    "version": 1,
    "n-threads": 6,
    "backend": {
      "version": 1,
      "type": "QnnHtp",
      "extensions": "htp_backend_ext_config.json",
      "QnnHtp": {
        "version": 1,
        "spill-fill-bufsize": 0,
        "use-mmap": true,
        "mmap-budget": 0,
        "poll": false,
        "cpu-mask": "0xe0",
        "kv-dim": 128,
        "kv-update-method": "pointer-shift",
        "data-alignment-size": 128,
        "rope-theta": 10000.0,
        "pos-id-dim": 64,
        "enable-graph-switching": true,
        "allow-async-init": true,
        "shared-engine": false
      }
    },
    "model": {
      "version": 1,
      "type": "binary",
      "binary": {
        "version": 1,
        "ctx-bins": [
          "fastvlm_decoder_part1.bin",
          "fastvlm_decoder_part2.bin"
        ],
        "lora": {
          "version": 1,
          "alpha-tensor-name": "lora_alpha",
          "adapters": [
            {
              "version": 1,
              "name": "adapter1",
              "alphas": ["tensor_alpha"],
              "bin-sections": ["section1.bin"]
            }
          ]
        }
      },
      "positional-encoding": {
        "type": "rope",
        "rope-dim": 64,
        "rope-theta": 10000.0,
        "rope-scaling": {
          "version": 1,
          "rope-type": "llama3",
          "factor": 8.0
        }
      }
    },
    "longcontext": {
      "version": 1,
      "type": "sliding-window",
      "sliding-window": {
        "version": 1,
        "window-size": 4096
      }
    }
  }
}
```

## Image Encoder Configuration

```json
{
  "image-encoder": {
    "version": 1,
    "embedding": {
      "version": 1,
      "type": "image-encoder",
      "engine": {
        "backend": {
          "type": "QnnHtp",
          "QnnHtp": {
            "pooled-output": false,
            "disable-kv-cache": true
          }
        },
        "model": {
          "vision-param": {
            "height": 1024,
            "width": 1024
          }
        }
      }
    }
  }
}
```

## Backend Types

### QnnHtp (NPU - Recommended for Production)

```json
{
  "type": "QnnHtp",
  "QnnHtp": {
    // Memory
    "spill-fill-bufsize": 0,
    "use-mmap": true,
    "mmap-budget": 0,
    "data-alignment-size": 128,
    
    // Execution
    "poll": false,
    "cpu-mask": "0xe0",
    
    // KV Cache
    "kv-dim": 128,
    "kv-update-method": "pointer-shift" | "shift-concat" | "smart-mask",
    
    // Performance
    "enable-graph-switching": true,
    "allow-async-init": true,
    
    // Advanced
    "rope-theta": 10000.0,
    "pos-id-dim": 64,
    "skip-lora-validation": false,
    "shared-engine": false,
    "graph-switching-lora-policy": "lazy" | "eager"
  }
}
```

### QnnGpu (GPU - Fallback)

```json
{
  "type": "QnnGpu"
}
```

### QnnGenAiTransformer (CPU - Reference)

```json
{
  "type": "QnnGenAiTransformer",
  "QnnGenAiTransformer": {
    "version": 1,
    "model-input": "embeddings" | "tokens",
    "use-mmap": true,
    "n-layer": 24,
    "n-embd": 896,
    "n-heads": 14,
    "n-kv-heads": 2
  }
}
```

## Model Types

### Binary Model (Compiled Context Binary)

```json
{
  "type": "binary",
  "binary": {
    "version": 1,
    "ctx-bins": [
      "fastvlm_decoder_part1.bin",
      "fastvlm_decoder_part2.bin"
    ],
    "lora": {
      "version": 1,
      "alpha-tensor-name": "lora_alpha",
      "adapters": [...]
    }
  }
}
```

### Library Model (Dynamic Library)

```json
{
  "type": "library",
  "library": {
    "version": 1,
    "model-bin": "fastvlm_decoder.so"
  }
}
```

## Long Context Support

### Sliding Window

```json
{
  "longcontext": {
    "version": 1,
    "type": "sliding-window",
    "sliding-window": {
      "version": 1,
      "window-size": 4096
    }
  }
}
```

### KeyDiff (Advanced)

```json
{
  "longcontext": {
    "version": 1,
    "type": "keydiff",
    "keydiff": {
      "version": 1,
      "scoring-network": "keydiff_net.bin",
      "update-frequency": 32,
      "anchor-alpha": 0.5
    }
  }
}
```

## Positional Encoding

### RoPE (Rotary Position Embedding)

```json
{
  "positional-encoding": {
    "type": "rope",
    "rope-dim": 64,
    "rope-theta": 10000.0,
    "rope-scaling": {
      "version": 1,
      "rope-type": "llama3" | "default" | "longrope" | "qwen2vl-mrope",
      "factor": 8.0,
      "short-factor": [1.0, 1.0, 1.0],
      "long-factor": [1.0, 1.0, 1.0],
      "original-max-position-embeddings": 8192
    }
  }
}
```

### Absolute (No Positional Encoding)

```json
{
  "positional-encoding": {
    "type": "absolute"
  }
}
```

## VLM-Specific Configuration

### Cross-Attention Model

For models like Qwen2-VL that use cross-attention:

```json
{
  "text-generator": {
    "version": 1,
    "type": "basic",
    "engine": {
      "model": {
        "cross-attention": true
      }
    }
  }
}
```

### MRoPE (Multi-dimensional RoPE)

For Qwen2-VL with 3D positional encoding:

```json
{
  "text-generator": {
    "engine": {
      "model": {
        "positional-encoding": {
          "type": "rope",
          "rope-scaling": {
            "rope-type": "qwen2vl-mrope",
            "mrope-section": ["horizontal", "vertical", "temporal"],
            "factor": 1.0
          }
        }
      }
    }
  }
}
```

## Complete VLM Configuration Example

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
      "eos-token": [151645],
      "pad-token": 151643
    },
    "tokenizer": {
      "version": 1,
      "path": "tokenizer.json"
    },
    "sampler": {
      "version": 1,
      "type": "basic",
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
        "type": "QnnHtp",
        "QnnHtp": {
          "version": 1,
          "spill-fill-bufsize": 0,
          "use-mmap": true,
          "mmap-budget": 0,
          "poll": false,
          "cpu-mask": "0xe0",
          "kv-dim": 128,
          "kv-update-method": "pointer-shift",
          "data-alignment-size": 128,
          "enable-graph-switching": true,
          "allow-async-init": true
        }
      },
      "model": {
        "version": 1,
        "type": "binary",
        "binary": {
          "version": 1,
          "ctx-bins": [
            "fastvlm_decoder_part1.bin",
            "fastvlm_decoder_part2.bin"
          ]
        },
        "positional-encoding": {
          "type": "rope",
          "rope-dim": 64,
          "rope-theta": 10000.0
        }
      }
    }
  },
  "image-encoder": {
    "version": 1,
    "embedding": {
      "version": 1,
      "type": "image-encoder",
      "engine": {
        "backend": {
          "type": "QnnHtp",
          "QnnHtp": {
            "pooled-output": false,
            "disable-kv-cache": true
          }
        },
        "model": {
          "vision-param": {
            "height": 1024,
            "width": 1024
          }
        }
      }
    }
  }
}
```

## Configuration Validation

GENIE validates all configurations and throws errors if:

1. **Missing required fields**: `"Missing context field: size"`
2. **Invalid values**: `"Invalid QnnHtp config: unsupported type: invalid"`
3. **Conflicting options**: `"Specify one config from pos-id-dim and positional-encoding"`
4. **Type mismatches**: `"Embedding datatype must match quantization"`
5. **Backend compatibility**: `"Backend config for incorrect backend type: QnnHtp"`

## Performance Tuning

### Reduce Latency

```json
{
  "QnnHtp": {
    "enable-graph-switching": true,
    "kv-update-method": "pointer-shift"
  },
  "sampler": {
    "top-k": 40,
    "top-p": 0.95
  }
}
```

### Reduce Memory

```json
{
  "text-generator": {
    "accumulator-size": 200000000,  // Reduced from 400MB
    "context": {
      "size": 2048  // Reduced from 4096
    }
  },
  "QnnHtp": {
    "vtcm-size": 524288  // Smaller VTCM
  }
}
```

### Increase Accuracy

```json
{
  "QnnHtp": {
    "kv-update-method": "smart-mask"  // Better than pointer-shift for some models
  },
  "embedding": {
    "datatype": "float32"  // Higher precision than int8
  }
}
```

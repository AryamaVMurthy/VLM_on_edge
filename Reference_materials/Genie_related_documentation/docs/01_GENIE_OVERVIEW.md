# GENIE SDK Overview

## What is GENIE?

**Qualcomm GENIE (Generative AI Next-generation Intelligent Engine)** is Qualcomm's runtime library for deploying generative AI models on edge devices. It's part of the Qualcomm AI Runtime (QAIRT) stack.

## Source Location
```
/home/aryamavmurthy/work/QIDK/qcom_ai_stack/GENIE_README.txt
```

## Key Capabilities

1. **Transformer Model Execution**: Optimized for autoregressive LLMs
2. **Multi-Backend Support**: HTP (NPU), GPU, CPU
3. **KV Cache Management**: Automatic handling of key-value caches
4. **Quantization Support**: INT8, W4A16, FP16 execution
5. **Streaming Output**: Token-by-token text generation
6. **LoRA Support**: Dynamic adapter loading

## Directory Structure

### Header Files (Public API)
```
/home/aryamavmurthy/work/QIDK/qcom_ai_stack/include/Genie/
├── GenieCommon.h        # Common types and status codes
├── GenieDialog.h        # High-level chat API
├── GenieEngine.h        # Low-level engine API
├── GenieLog.h           # Logging API
├── GenieNode.h          # Pipeline node definitions
├── GeniePipeline.h      # Composable pipeline API
├── GenieProfile.h       # Profiling API
├── GenieSampler.h       # Token sampling API
└── GenieTokenizer.h     # Tokenization API
```

### Source Implementation
```
/home/aryamavmurthy/work/QIDK/qcom_ai_stack/examples/Genie/Genie/src/
├── Dialog.cpp           # GenieDialog implementation
├── Engine.cpp           # Engine management
├── GeniePipeline.cpp    # Pipeline C API wrapper
├── GenieDialog.cpp      # Dialog C API wrapper
├── Sampler.cpp          # Sampling logic
├── Tokenizer.cpp        # Tokenization
├── pipeline/
│   ├── Pipeline.cpp     # Internal pipeline class
│   ├── Node.cpp         # Base node class
│   ├── TextGenerator.cpp # Text generation node
│   ├── ImageEncoder.cpp # Vision encoder node
│   └── Accumulator.cpp  # Embedding accumulator
└── qualla/
    ├── engines/qnn-htp/ # HTP backend
    │   ├── nsp-kvmanager.cpp  # KV cache management
    │   ├── nsp-model.cpp      # Model execution
    │   └── nsp-graph.cpp      # Graph management
    └── encoders/
        └── image-encoders/
            └── imageEncoder.cpp
```

### Example Configurations
```
/home/aryamavmurthy/work/QIDK/qcom_ai_stack/examples/Genie/configs/
├── llama2-7b/           # Llama 2 configurations
├── llama3-8b/           # Llama 3 configurations
├── glm-4v/              # Vision-Language model (GLM-4V)
│   ├── glm-4v.json      # Main config
│   ├── siglip.json      # Vision encoder config
│   └── text-encoder.json
├── llava-e2t/           # LLaVA embedding-to-text
├── htp_backend_ext_config.json  # HTP backend extensions
└── sampler.json         # Sampler configuration
```

### Binary Tools
```
/home/aryamavmurthy/work/QIDK/qcom_ai_stack/bin/
├── genie-t2t-run        # Text-to-text runner
├── genie-app            # Application runner with scripts
└── genie-t2e-run        # Text-to-embedding runner
```

## Two Main APIs

### 1. GenieDialog API (High-Level)

For simple text-in/text-out or embedding-in/text-out use cases.

```c
// Create dialog from JSON config
GenieDialogConfig_Handle_t configHandle;
GenieDialogConfig_createFromJson(configJson, &configHandle);

GenieDialog_Handle_t dialogHandle;
GenieDialog_create(configHandle, &dialogHandle);

// Execute query
GenieDialog_query(dialogHandle, "Hello!", 
    GENIE_DIALOG_SENTENCE_COMPLETE,
    myCallback, userData);

// For VLM: embedding query (vision embeddings -> text)
GenieDialog_embeddingQuery(dialogHandle, 
    embeddings, embeddingsSize,
    GENIE_DIALOG_SENTENCE_COMPLETE,
    tokenToEmbeddingCallback,
    responseCallback, userData);
```

### 2. GeniePipeline API (Low-Level)

For composable multimodal pipelines.

```c
// Create pipeline
GeniePipelineConfig_Handle_t pipeConfig;
GeniePipelineConfig_createFromJson(pipelineJson, &pipeConfig);

GeniePipeline_Handle_t pipeline;
GeniePipeline_create(pipeConfig, &pipeline);

// Create nodes
GenieNode_Handle_t imageEncoder, textGenerator;
GenieNodeConfig_createFromJson(encoderJson, &encoderConfig);
GenieNode_create(encoderConfig, &imageEncoder);

// Add nodes to pipeline
GeniePipeline_addNode(pipeline, imageEncoder);
GeniePipeline_addNode(pipeline, textGenerator);

// Connect nodes
GeniePipeline_connect(pipeline, 
    imageEncoder, GENIE_NODE_IMAGE_ENCODER_EMBEDDING_OUTPUT,
    textGenerator, GENIE_NODE_TEXT_GENERATOR_EMBEDDING_INPUT);

// Set input data
GenieNode_setData(imageEncoder, 
    GENIE_NODE_IMAGE_ENCODER_IMAGE_INPUT,
    imagePixels, imageSize, NULL);

// Execute
GeniePipeline_execute(pipeline, userData);
```

## Backend Types

| Backend Type | Library | Description |
|--------------|---------|-------------|
| `QnnHtp` | libQnnHtp.so | Hexagon NPU (Production) |
| `QnnGpu` | libQnnGpu.so | Adreno GPU |
| `QnnGenAiTransformer` | libQnnGenAiTransformer.so | CPU Reference |

## Key Configuration Fields

```json
{
  "dialog": {
    "context": {
      "size": 4096,           // Max sequence length
      "n-vocab": 32000,       // Vocabulary size
      "bos-token": 1,         // Begin-of-sequence token
      "eos-token": 2          // End-of-sequence token
    },
    "embedding": {
      "size": 4096,           // Embedding dimension
      "type": "lut",          // "lut" (lookup table) or "callback"
      "lut-path": "lut.bin"   // Path to embedding table
    },
    "engine": {
      "backend": {
        "type": "QnnHtp",
        "QnnHtp": {
          "kv-dim": 128,      // KV cache head dimension
          "use-mmap": true,   // Memory-map model files
          "poll": false       // Polling mode for inference
        }
      }
    }
  }
}
```

## Common Usage Patterns

### Pattern 1: Basic LLM Chat
```bash
./genie-t2t-run --config llama3-8b-htp.json
```

### Pattern 2: VLM with Embedding Input
```bash
./genie-t2t-run --config vlm_config.json \
    --embedding_file combined_embeddings.raw \
    --embedding_query_output_type text
```

### Pattern 3: Programmatic Pipeline
See `ImageEncoder.cpp` and `TextGenerator.cpp` for how nodes are connected.

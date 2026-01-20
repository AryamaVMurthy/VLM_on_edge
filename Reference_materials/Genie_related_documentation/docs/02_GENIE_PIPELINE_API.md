# GENIE Pipeline API - Complete Reference

## Overview

GENIE Pipeline API provides **node-based, composable workflow** for multimodal AI. Use this when:
- You need to connect multiple components (e.g., vision encoder + text generator)
- You want to build custom pipelines
- You're deploying VLM (Vision + Language models)

**Do NOT use with Dialog API** - they're separate.

## API Functions

### Pipeline Creation

```cpp
// Create configuration from JSON
GenieStatus_t GeniePipelineConfig_createFromJson(
    const char* jsonString,      // JSON configuration string
    GeniePipelineConfig_Handle_t* handle  // Output: config handle
);

// Create pipeline from configuration
GenieStatus_t GeniePipeline_create(
    GeniePipelineConfig_Handle_t configHandle,
    GeniePipeline_Handle_t* pipelineHandle
);

// Destroy pipeline
GenieStatus_t GeniePipeline_destroy(
    GeniePipeline_Handle_t pipelineHandle
);
```

### Node Management

```cpp
// Create node from configuration
GenieStatus_t GenieNode_create(
    const char* jsonString,      // Node configuration JSON
    GenieNode_Handle_t* nodeHandle
);

// Add node to pipeline
GenieStatus_t GeniePipeline_addNode(
    GeniePipeline_Handle_t pipelineHandle,
    GenieNode_Handle_t nodeHandle
);

// Connect nodes: Producer output → Consumer input
GenieStatus_t GeniePipeline_connect(
    GeniePipeline_Handle_t pipelineHandle,
    GenieNode_Handle_t producerNode,
    GenieNode_IOName_t producerIOName,    // Output port of producer
    GenieNode_Handle_t consumerNode,
    GenieNode_IOName_t consumerIOName     // Input port of consumer
);
```

### Node I/O

```cpp
// Set input data for node
GenieStatus_t GenieNode_setData(
    GenieNode_Handle_t nodeHandle,
    GenieNode_IOName_t ioName,    // Which input port
    const void* data,               // Data pointer
    const uint32_t size,             // Data size in bytes
    GenieNode_DataType_t dataType,
    void* reserved                  // Always NULL
);

// Execute pipeline
GenieStatus_t GeniePipeline_execute(
    GeniePipeline_Handle_t pipelineHandle,
    void* userData                 // Passed to callbacks
);
```

## Node Types

### Image Encoder

```cpp
// I/O Names
GENIE_NODE_IMAGE_ENCODER_IMAGE_INPUT             = 200  // Raw pixels (e.g., 1024x1024x3)
GENIE_NODE_IMAGE_ENCODER_EMBEDDING_OUTPUT        = 201  // Vision embeddings output
GENIE_NODE_IMAGE_ENCODER_IMAGE_POS_SIN           = 202  // RoPE sin (Qwen2-VL)
GENIE_NODE_IMAGE_ENCODER_IMAGE_POS_COS           = 203  // RoPE cos (Qwen2-VL)
GENIE_NODE_IMAGE_ENCODER_IMAGE_FULL_ATTN_MASK    = 204  // Full attention mask
GENIE_NODE_IMAGE_ENCODER_IMAGE_WINDOW_ATTN_MASK  = 205  // Window attention mask
GENIE_NODE_IMAGE_ENCODER_PRETILE_EMBEDDING_INPUT  = 206  // Llama3.2-11B
GENIE_NODE_IMAGE_ENCODER_POSTTILE_EMBEDDING_INPUT = 207  // Llama3.2-11B
GENIE_NODE_IMAGE_ENCODER_GATED_POS_EMBEDDING_INPUT = 208  // Llama3.2-11B
```

### Text Generator

```cpp
// I/O Names
GENIE_NODE_TEXT_GENERATOR_TEXT_INPUT        = 0    // Text string input
GENIE_NODE_TEXT_GENERATOR_EMBEDDING_INPUT   = 1    // Embedding tensor input
GENIE_NODE_TEXT_GENERATOR_TEXT_OUTPUT       = 2    // Generated text output
```

### Text Encoder

```cpp
// I/O Names
GENIE_NODE_TEXT_ENCODER_TEXT_INPUT         = 100  // Text string input
GENIE_NODE_TEXT_ENCODER_EMBEDDING_OUTPUT  = 101  // Text embeddings output
```

## Complete VLM Pipeline Example

```cpp
#include <Genie/GeniePipeline.h>
#include <Genie/GenieNode.h>

// 1. Create pipeline configuration
GeniePipelineConfig_Handle_t pipeConfig;
const char* pipeJson = R"({
  "nodes": [
    {"type": "image-encoder", "config": "vision_encoder.json"},
    {"type": "text-generator", "config": "llm_decoder.json"}
  ]
})";
GeniePipelineConfig_createFromJson(pipeJson, &pipeConfig);

// 2. Create pipeline
GeniePipeline_Handle_t pipeline;
GeniePipeline_create(pipeConfig, &pipeline);

// 3. Create nodes
GenieNode_Handle_t imageEncoder, textGenerator;

const char* visionConfig = R"({
  "embedding": {
    "type": "image-encoder",
    "engine": {
      "backend": {"type": "QnnHtp"},
      "model": {"vision-param": {"height": 1024, "width": 1024}}
    }
  }
})";

const char* generatorConfig = R"({
  "text-generator": {
    "version": 1,
    "embedding": {"size": 896, "datatype": "float32"},
    "context": {"size": 4096, "n-vocab": 151936},
    "tokenizer": {"path": "tokenizer.json"},
    "engine": {
      "backend": {"type": "QnnHtp"},
      "model": {"type": "binary", "binary": {"ctx-bins": ["decoder.bin"]}}
    }
  }
})";

GenieNode_create(visionConfig, &imageEncoder);
GenieNode_create(generatorConfig, &textGenerator);

// 4. Add nodes to pipeline
GeniePipeline_addNode(pipeline, imageEncoder);
GeniePipeline_addNode(pipeline, textGenerator);

// 5. Connect nodes
GeniePipeline_connect(pipeline,
    imageEncoder, GENIE_NODE_IMAGE_ENCODER_EMBEDDING_OUTPUT,
    textGenerator, GENIE_NODE_TEXT_GENERATOR_EMBEDDING_INPUT);

// 6. Set input data
uint8_t imagePixels[1024*1024*3];  // Load your image
GenieNode_setData(imageEncoder,
    GENIE_NODE_IMAGE_ENCODER_IMAGE_INPUT,
    imagePixels, sizeof(imagePixels), GENIE_NODE_DATA_TYPE_RAW, NULL);

// 7. Execute pipeline
GeniePipeline_execute(pipeline, userData);

// Cleanup
GeniePipeline_destroy(pipeline);
GenieNode_destroy(imageEncoder);
GenieNode_destroy(textGenerator);
```

## Callback Functions

```cpp
// Text output callback signature
typedef void (*GenieNode_TextOutput_Callback_t)(
    const char* text,               // Generated text chunk
    GenieNode_TextOutput_SentenceCode_t code,  // BEGIN, CONTINUE, END
    const void* userData
);

// Set callback
GenieStatus_t GenieNode_setCallback(
    GenieNode_Handle_t nodeHandle,
    GenieNode_IOName_t ioName,
    void* callback,                // Function pointer
    GenieNode_CallbackType_t type
);
```

## Pipeline Execution Flow

```
1. GeniePipeline_execute() called
   │
   ├─► ImageEncoder node executes
   │     - setImageInputData() receives image
   │     - encode() runs on NPU
   │     - Output: vision embeddings
   │
   ├─► Vision embeddings flow to TextGenerator via connection
   │
   ├─► TextGenerator node executes
   │     - Receives vision embeddings + text embeddings from Accumulator
   │     - embeddingQuery() runs autoregressive generation
   │     - KV cache updates automatically
   │     - Text output callback fires for each token
   │
   └─► Pipeline completes
```

## Node Configuration Schema

```json
{
  "version": 1,
  "type": "image-encoder" | "text-generator" | "text-encoder",
  
  "embedding": {
    "version": 1,
    "type": "lut" | "image-encoder" | "callback",
    "size": <embedding_dim>,
    "datatype": "float32" | "ufixed8" | "sfixed16",
    "lut-path": "<path_to_lut.bin>"
  },
  
  "engine": {
    "version": 1,
    "backend": {
      "type": "QnnHtp" | "QnnGpu" | "QnnGenAiTransformer"
    },
    "model": {
      "type": "binary" | "library",
      "binary": {
        "ctx-bins": ["<binary_file>", ...]
      },
      "library": {
        "model-bin": "<library_file>"
      }
    }
  }
}
```

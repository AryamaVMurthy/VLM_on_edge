# GENIE Dialog API - Complete Reference

## Overview

GENIE Dialog API provides **high-level, simple** interface for text generation. Use this when:
- You just need text-in/text-out (chat)
- You don't need custom node connections
- You want minimal code

**Do NOT use with Pipeline API** - they're separate.

## API Functions

### Dialog Creation

```cpp
// Create configuration from JSON
GenieStatus_t GenieDialogConfig_createFromJson(
    const char* jsonString,
    GenieDialogConfig_Handle_t* handle
);

// Create dialog
GenieStatus_t GenieDialog_create(
    GenieDialogConfig_Handle_t configHandle,
    GenieDialog_Handle_t* dialogHandle
);

// Destroy dialog
GenieStatus_t GenieDialog_destroy(
    GenieDialog_Handle_t dialogHandle
);
```

### Text Query (Standard)

```cpp
// Query with text string
GenieStatus_t GenieDialog_query(
    GenieDialog_Handle_t dialogHandle,
    const char* queryStr,                // Your prompt
    GenieDialog_SentenceCode_t sentenceCode,  // Requested output type
    GenieDialog_QueryCallback_t callback,  // Response callback
    const void* userData
);
```

### Embedding Query (VLM)

```cpp
// Query with pre-computed embeddings
GenieStatus_t GenieDialog_embeddingQuery(
    GenieDialog_Handle_t dialogHandle,
    const void* embeddings,               // Combined [vision | text] embeddings
    const uint32_t embeddingsSize,
    GenieDialog_SentenceCode_t sentenceCode,
    GenieDialog_TokenToEmbeddingCallback_t t2eCallback,  // Optional
    GenieDialog_QueryCallback_t callback,
    const void* userData
);
```

### Token Query

```cpp
// Query with token IDs
GenieStatus_t GenieDialog_tokenQuery(
    GenieDialog_Handle_t dialogHandle,
    const uint32_t* inputTokens,
    const uint32_t numTokens,
    GenieDialog_SentenceCode_t sentenceCode,
    GenieDialog_TokenQueryCallback_t callback,
    const void* userData
);
```

## Sentence Codes

```cpp
typedef enum {
    GENIE_DIALOG_SENTENCE_COMPLETE = 0,    // Full response at once
    GENIE_DIALOG_SENTENCE_BEGIN = 1,      // First chunk (streaming start)
    GENIE_DIALOG_SENTENCE_CONTINUE = 2,   // Middle chunks
    GENIE_DIALOG_SENTENCE_END = 3,        // Last chunk (streaming end)
    GENIE_DIALOG_SENTENCE_ABORT = 4,       // Aborted (error)
    GENIE_DIALOG_SENTENCE_REWIND = 5,      // KV cache rewind (speculative)
    GENIE_DIALOG_SENTENCE_RESUME = 6        // Resume after pause
} GenieDialog_SentenceCode_t;
```

## Callback Signatures

### Query Callback (Text Output)

```cpp
typedef void (*GenieDialog_QueryCallback_t)(
    const char* response,              // Generated text chunk (null-terminated)
    GenieDialog_SentenceCode_t sentenceCode,  // BEGIN, CONTINUE, END
    const void* userData
);

// Example:
void myCallback(const char* response, GenieDialog_SentenceCode_t code, const void* userData) {
    if (code == GENIE_DIALOG_SENTENCE_BEGIN) {
        printf("[START] %s", response);
    } else if (code == GENIE_DIALOG_SENTENCE_CONTINUE) {
        printf("[MORE] %s", response);
    } else if (code == GENIE_DIALOG_SENTENCE_END) {
        printf("[DONE] %s\n", response);
    }
}
```

### Token-to-Embedding Callback (Optional)

```cpp
typedef void (*GenieDialog_TokenToEmbeddingCallback_t)(
    const uint32_t token,              // Token ID
    const float* embedding,            // Embedding vector
    const uint32_t embeddingSize,      // Embedding dimension
    const void* userData
);
```

### Token Query Callback

```cpp
typedef void (*GenieDialog_TokenQueryCallback_t)(
    const uint32_t token,              // Generated token ID
    GenieDialog_SentenceCode_t sentenceCode,
    const void* userData
);
```

## Complete Dialog Example

### Simple Text Chat

```cpp
#include <Genie/GenieDialog.h>

// 1. Create configuration
GenieDialogConfig_Handle_t configHandle;
const char* dialogJson = R"({
  "dialog": {
    "version": 1,
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
      "seed": 42,
      "temp": 0.8,
      "top-k": 40,
      "top-p": 0.95
    },
    "engine": {
      "version": 1,
      "backend": {
        "type": "QnnHtp",
        "QnnHtp": {
          "kv-dim": 128,
          "use-mmap": true,
          "poll": false
        }
      },
      "model": {
        "version": 1,
        "type": "binary",
        "binary": {
          "version": 1,
          "ctx-bins": ["fastvlm_full_decoder_htp.bin"]
        }
      }
    }
  }
})";

GenieDialogConfig_createFromJson(dialogJson, &configHandle);

// 2. Create dialog
GenieDialog_Handle_t dialogHandle;
GenieDialog_create(configHandle, &dialogHandle);

// 3. Query with text
const char* query = "What is the capital of France?";
GenieDialog_query(dialogHandle, query,
    GENIE_DIALOG_SENTENCE_COMPLETE,
    myCallback, userData);

// Cleanup
GenieDialog_destroy(dialogHandle);
GenieDialogConfig_destroy(configHandle);
```

### VLM with Embedding Input

```cpp
#include <Genie/GenieDialog.h>

// Assume you have combined embeddings ready
float* combinedEmbeddings;  // [vision_emb | text_emb]
uint32_t embeddingsSize;    // Total bytes

// Query with embeddings
GenieDialog_embeddingQuery(dialogHandle,
    combinedEmbeddings,
    embeddingsSize,
    GENIE_DIALOG_SENTENCE_COMPLETE,
    NULL,  // No token-to-embedding callback
    myCallback, userData);
```

## Command Line Usage

### genie-t2t-run Binary

```bash
# Simple chat with config file
./genie-t2t-run --config llama3-8b-htp.json

# With custom prompt
./genie-t2t-run --config llama3-8b-htp.json --prompt "Hello"

# With embeddings (VLM)
./genie-t2t-run \
    --config vlm_config.json \
    --embedding_file combined_embeddings.raw \
    --embedding_table embedding_int8_lut.bin \
    --embedding_query_output_type text

# With profile
./genie-t2t-run \
    --config llama3-8b-htp.json \
    --profile profile.json
```

## Configuration Schema

```json
{
  "dialog": {
    "version": 1,
    "type": "basic",
    
    // Embedding configuration
    "embedding": {
      "version": 1,
      "type": "lut" | "callback",
      "size": <embedding_dim>,
      "datatype": "float32" | "ufixed8" | "sfixed16",
      "lut-path": "<path>"
    },
    
    // Context configuration
    "context": {
      "version": 1,
      "size": <max_sequence_length>,
      "n-vocab": <vocabulary_size>,
      "bos-token": <begin_token>,
      "eos-token": <end_token>,  // Can be array
      "pad-token": <pad_token>,
      "n-embd": <embedding_dim>  // Alternative to embedding.size
    },
    
    // Tokenizer
    "tokenizer": {
      "version": 1,
      "path": "<tokenizer.json>"
    },
    
    // Sampling configuration
    "sampler": {
      "version": 1,
      "type": "basic",
      "seed": 42,
      "temp": 0.0 to 1.0,      // 0.0 = greedy
      "top-k": 1 to vocab_size,
      "top-p": 0.0 to 1.0,
      "greedy": false
    },
    
    // Engine configuration
    "engine": {
      "version": 1,
      "n-threads": <num_cpu_threads>,
      "backend": {
        "type": "QnnHtp" | "QnnGpu" | "QnnGenAiTransformer"
      },
      "model": {
        "type": "binary" | "library"
      }
    }
  }
}
```

## Streaming Output Format

GENIE streams output as text chunks:

```
[BEGIN]:The
[CONTINUE]:quick
[CONTINUE]:brown
[CONTINUE]:fox
[END]:jumps over the lazy dog.
```

Your callback receives these chunks and can:
- Display progressively
- Concatenate for final result
- Detect [END] to know generation complete

## Differences: Dialog vs Pipeline

| Aspect | Dialog API | Pipeline API |
|---------|-------------|--------------|
| **Abstraction** | High-level (single call) | Low-level (nodes + connections) |
| **Use Case** | Simple chat | Multimodal, custom workflows |
| **VLM Support** | embeddingQuery() only | ImageEncoder + TextGenerator nodes |
| **Complexity** | Minimal | High (you manage graph) |
| **Flexibility** | Low | High (arbitrary nodes) |

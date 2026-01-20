# Text Generator & Decoding Implementation

## Source Files

| File | Purpose |
|------|---------|
| `TextGenerator.hpp` | GENIE API for text generation node |
| `TextGenerator.cpp` | GENIE pipeline node wrapper |
| `Dialog.hpp` | Public GENIE dialog API |
| `Dialog.cpp` | Dialog implementation with validation |

**Locations:**
```
Public: /home/aryamavmurthy/work/QIDK/qcom_ai_stack/include/Genie/
       /home/aryamavmurthy/work/QIDK/qcom_ai_stack/examples/Genie/Genie/src/pipeline/

Internal: /home/aryamavmurthy/work/QIDK/qcom_ai_stack/examples/Genie/Genie/src/qualla/
```

## TextGenerator Class

```cpp
// From Genie/src/pipeline/TextGenerator.cpp

class TextGenerator : public Node {
public:
    TextGenerator(qualla::json config,
                 std::shared_ptr<ProfileStat> profileStat,
                 std::shared_ptr<genie::log::Logger> logger);
    
    // Bind to pipeline
    int32_t bindPipeline(Pipeline& pipeline);
    
    // Set text input
    int32_t setTextInputData(GenieNode_IOName_t nodeIOName,
                           const char* txt,
                           std::shared_ptr<ProfileStat> profileStat);
    
    // Set embedding input
    int32_t setEmbeddingInputData(GenieNode_IOName_t nodeIOName,
                                const void* embedding,
                                size_t embeddingSize,
                                std::shared_ptr<ProfileStat> profileStat);
    
    // Set output callback
    int32_t setTextOutputCallback(GenieNode_IOName_t nodeIOName,
                                 GenieNode_TextOutput_Callback_t callback);
    
    // Execute generation
    int32_t execute(void* userData, std::shared_ptr<ProfileStat> profileStat);
    
    // State management
    void reset();
    int32_t save(const std::string& name);
    int32_t restore(const std::string& name);
    
    // LoRA support
    int32_t applyLora(std::string loraAdapterName,
                     std::string engine,
                     std::shared_ptr<ProfileStat> profileStat);
    
private:
    std::shared_ptr<genie::Dialog> m_generator;  // Underlying GENIE dialog
    Pipeline* m_pipeline;                    // Parent pipeline
    std::string m_queryString;               // Accumulated text
    bool m_crossAttention;                  // VLM flag
    bool m_usingMRope;                       // Qwen2-VL flag
    size_t m_accumulatorSize;
};
```

## Generation Modes

```cpp
// From TextGenerator.cpp::execute()

int32_t TextGenerator::execute(...) {
    bool useEmbedding = false;
    const void* embData = nullptr;
    uint32_t embSize = 0;
    
    // Get accumulator data (combined vision + text embeddings)
    const void* accData = m_pipeline->m_accumulator->getData();
    const uint32_t accSize = m_pipeline->m_accumulator->getDataSize();
    
    // Choose generation mode
    if (!m_crossAttention) {
        // Standard VLM: Embeddings go to LLM input
        if (accSize > 0) {
            useEmbedding = true;
            embData = accData;
            embSize = accSize;
        }
    } else {
        // Cross-attention VLM: Different flow
        useEmbedding = false;
        m_generator->setCrossAttentionHiddenStates(accData, accSize);
    }
    
    // Handle MRoPE (Multi-dimensional RoPE)
    if (useEmbedding && m_usingMRope) {
        m_generator->setVisionParam(m_pipeline->m_accumulator->getVisionParam());
    }
    
    // Execute query
    if (useEmbedding) {
        // VLM mode: embeddings -> text
        m_generator->embeddingQuery(embData, embSize,
                                 GENIE_NODE_SENTENCE_COMPLETE,
                                 m_textOutputCallback,
                                 userData, profileStat);
    } else {
        // Text-only mode: text -> text
        m_generator->query(m_queryString.c_str(),
                          GENIE_NODE_SENTENCE_COMPLETE,
                          m_textOutputCallback,
                          userData, profileStat);
    }
    
    // Clear buffers
    m_pipeline->m_accumulator->flush();
    m_queryString.clear();
    
    return GENIE_STATUS_SUCCESS;
}
```

## GENIE Dialog API

```c
// From GenieDialog.h - High-level API

// Query types
GenieStatus_t GenieDialog_query(
    GenieDialog_Handle_t dialogHandle,
    const char* queryStr,
    GenieDialog_SentenceCode_t sentenceCode,
    GenieDialog_QueryCallback_t callback,
    const void* userData);

// For VLM with pre-computed embeddings
GenieStatus_t GenieDialog_embeddingQuery(
    GenieDialog_Handle_t dialogHandle,
    const void* embeddings,           // Combined [vision | text]
    const uint32_t embeddingsSize,
    GenieDialog_SentenceCode_t sentenceCode,
    GenieDialog_TokenToEmbeddingCallback_t t2eCallback,
    GenieDialog_QueryCallback_t callback,
    const void* userData);

// Token-to-token query
GenieStatus_t GenieDialog_tokenQuery(
    GenieDialog_Handle_t dialogHandle,
    const uint32_t* inputTokens,
    const uint32_t numTokens,
    GenieDialog_SentenceCode_t sentenceCode,
    GenieDialog_TokenQueryCallback_t callback,
    const void* userData);
```

## Dialog Configuration

```json
{
  "dialog": {
    "version": 1,
    "type": "basic",
    "accumulator-size": 400000000,      // Buffer for embeddings (bytes)
    
    // Embedding input configuration
    "embedding": {
      "version": 1,
      "type": "lut",                    // "lut" or "callback"
      "lut-path": "embedding_int8_lut.bin",
      "size": 896,                      // Embedding dimension
      "datatype": "float32",            // "float32", "ufixed8", etc.
      "quant-param": {
        "scale": 1.0,
        "offset": 0
      }
    },
    
    // Context configuration
    "context": {
      "version": 1,
      "size": 4096,                    // Max sequence length
      "n-vocab": 151936,                // Vocabulary size
      "bos-token": 151644,
      "eos-token": [151645],
      "pad-token": 151643
    },
    
    // Tokenizer
    "tokenizer": {
      "version": 1,
      "path": "tokenizer.json"
    },
    
    // Sampling configuration
    "sampler": {
      "version": 1,
      "type": "basic",
      "seed": 42,
      "temp": 0.8,                    // Temperature (0.0 = greedy)
      "top-k": 40,                    // Top-k sampling
      "top-p": 0.95,                   // Nucleus sampling
      "greedy": false
    },
    
    // Engine configuration
    "engine": {
      "version": 1,
      "n-threads": 6,                 // Number of CPU threads
      "backend": {
        "version": 1,
        "type": "QnnHtp",            // "QnnHtp", "QnnGpu", "QnnGenAiTransformer"
        "QnnHtp": {
          "kv-dim": 128,              // KV head dimension
          "enable-graph-switching": true
        },
        "QnnGenAiTransformer": {
          "model-input": "embeddings"    // "tokens" or "embeddings"
        }
      },
      "model": {
        "version": 1,
        "type": "binary",             // "binary" or "library"
        "binary": {
          "version": 1,
          "ctx-bins": [
            "model_part1.bin",
            "model_part2.bin"
          ]
        }
      },
      "longcontext": {
        "version": 1,
        "type": "sliding-window",    // "sliding-window" or "keydiff"
        "sliding-window": {
          "version": 1,
          "window-size": 4096
        }
      }
    }
  }
}
```

## Sentence Codes (Streaming)

```c
// From GenieDialog.h
typedef enum {
    GENIE_DIALOG_SENTENCE_COMPLETE = 0,  // Full response
    GENIE_DIALOG_SENTENCE_BEGIN = 1,    // First chunk
    GENIE_DIALOG_SENTENCE_CONTINUE = 2, // Middle chunks
    GENIE_DIALOG_SENTENCE_END = 3,      // Last chunk
    GENIE_DIALOG_SENTENCE_ABORT = 4,     // Aborted
    GENIE_DIALOG_SENTENCE_REWIND = 5,     // KV cache rewind (speculative)
    GENIE_DIALOG_SENTENCE_RESUME = 6      // Resumed after pause
} GenieDialog_SentenceCode_t;
```

**Callback signature:**
```c
typedef void (*GenieDialog_QueryCallback_t)(
    const char* response,              // Null-terminated string
    GenieDialog_SentenceCode_t sentenceCode,
    const void* userData
);
```

## Cross-Attention Support

For VLMs with cross-attention (e.g., Qwen2-VL):

```cpp
// From TextGenerator.cpp

if (m_crossAttention) {
    // Set cross-attention hidden states
    if (!m_generator->setCrossAttentionHiddenStates(
            m_pipeline->m_accumulator->getData(),
            m_pipeline->m_accumulator->getDataSize())) {
        throw Exception(GENIE_STATUS_ERROR_GENERAL,
                        "setCrossAttentionHiddenStates fail");
    }
    
    // Query with text (embeddings handled separately)
    m_generator->query(m_queryString.c_str(),
                      GENIE_NODE_SENTENCE_COMPLETE,
                      m_textOutputCallback,
                      userData,
                      profileStat);
}
```

## MRoPE (Multi-dimensional RoPE)

For models like Qwen2-VL with 3D positional encoding:

```cpp
// From Dialog.cpp - Configuration validation

if (posEncodingConfig.contains("rope-scaling") &&
    posEncodingConfig["rope-scaling"]["rope-type"] == "qwen2vl-mrope") {
    // Enable MRoPE
    m_usingMRope = true;
}
```

**MRoPE Configuration:**
```json
{
  "positional-encoding": {
    "type": "rope",
    "rope-scaling": {
      "version": 1,
      "rope-type": "qwen2vl-mrope",
      "mrope-section": [           // 3 dimensions
        "horizontal",
        "vertical",
        "temporal"
      ],
      "factor": 1.0
    }
  }
}
```

## Output Flow

```
Combined Embeddings (from Accumulator)
    │ Shape: [1, total_tokens, embedding_dim]
    │
    ▼
TextGenerator::execute()
    │
    ├─────────────┐
    │             │
    │             ▼
    │     Check Mode
    │             │
    │  ┌──────────┴────────┐
    │  │                    │
    │  │                    ▼
    │  │            useEmbedding
    │  │              (VLM mode)
    │  │                    │
    │  │                    ▼
    │  │         Set MRoPE?   │
    │  │                    │
    │  │           ┌─────────┴─────────┐
    │  │           │                 │
    │  │           ▼                 ▼
    │  │    embeddingQuery()    query()  (text-only)
    │  │                    │          │
    │  │           ┌──────────────┴───────┐
    │  │           │                   │
    │  │           ▼                   ▼
    │  │  LLM Decoding   LLM Decoding
    │  │  │                   │
    │  │  KV Cache      No KV Cache
    │  │  │                   │
    │  │  │                   │
    │  │  └───────────┬─────────┘
    │  │              │
    │  │              ▼
    │  │    Text Output (streaming)
    │  │              │
    │  │  [BEGIN]: token1 [BEGIN]: token2 ...
    │  │              │
    │  │              │
    │  │              ▼
    │  │    Callback: "Generated text"
```

## Key Differences: GenieDialog vs TextGenerator Node

| Aspect | GenieDialog | TextGenerator Node |
|---------|--------------|---------------------|
| **Abstraction** | High-level | Low-level pipeline component |
| **Use Case** | Simple T2T | Composable multimodal pipeline |
| **Embeddings** | Token-to-embedding lookup | Accepts raw embeddings |
| **Pipeline** | Not required | Requires Pipeline binding |
| **Flexibility** | Limited | High (custom nodes) |

## Memory Management

Accumulator buffer size must accommodate:
```
max_vision_tokens * embedding_dim + max_text_tokens * embedding_dim

Example for FastVLM:
  - Vision: 128 tokens * 1024 dim = 131,072 floats = 524KB
  - Text: 512 tokens * 896 dim = 458,752 floats = 1.8MB
  - Total: ~2.4MB
  - With padding (4096 tokens): ~14.7MB
```

Configuration:
```json
{
  "text-generator": {
    "accumulator-size": 400000000  // 400MB buffer
  }
}
```

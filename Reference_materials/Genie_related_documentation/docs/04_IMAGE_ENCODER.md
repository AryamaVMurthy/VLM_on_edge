# Image Encoder Implementation (Vision on NPU)

## Source Files

| File | Purpose |
|------|---------|
| `ImageEncoder.hpp` | Public GENIE API for image encoder node |
| `ImageEncoder.cpp` | GENIE pipeline node wrapper |
| `imageEncoder.hpp` | Internal qualla implementation |
| `imageEncoder.cpp` | Internal qualla encoder |

**Locations:**
```
Public: /home/aryamavmurthy/work/QIDK/qcom_ai_stack/include/Genie/
       /home/aryamavmurthy/work/QIDK/qcom_ai_stack/examples/Genie/Genie/src/pipeline/Internal:
       
Internal: /home/aryamavmurthy/work/QIDK/qcom_ai_stack/examples/Genie/Genie/src/qualla/encoders/image-encoders/
```

## GENIE Node Types for VLM

```c
// From GenieNode.h - Image encoder I/O definitions
GENIE_NODE_IMAGE_ENCODER_IMAGE_INPUT            = 200, // Raw pixels (e.g., 1024x1024x3)
GENIE_NODE_IMAGE_ENCODER_EMBEDDING_OUTPUT       = 201, // Vision embeddings output
GENIE_NODE_IMAGE_ENCODER_IMAGE_POS_SIN          = 202, // RoPE sin positions
GENIE_NODE_IMAGE_ENCODER_IMAGE_POS_COS          = 203, // RoPE cos positions
GENIE_NODE_IMAGE_ENCODER_IMAGE_FULL_ATTN_MASK   = 204, // Full attention mask
GENIE_NODE_IMAGE_ENCODER_IMAGE_WINDOW_ATTN_MASK = 205, // Window attention mask
// Llama3.2-11B specific:
GENIE_NODE_IMAGE_ENCODER_PRETILE_EMBEDDING_INPUT   = 206,
GENIE_NODE_IMAGE_ENCODER_POSTTILE_EMBEDDING_INPUT  = 207,
GENIE_NODE_IMAGE_ENCODER_GATED_POS_EMBEDDING_INPUT = 208,
```

## ImageEncoder Class (GENIE Pipeline Node)

```cpp
// From Genie/src/pipeline/ImageEncoder.cpp

class ImageEncoder : public Node {
public:
    ImageEncoder(qualla::json config,
                  std::shared_ptr<ProfileStat> profileStat,
                  std::shared_ptr<genie::log::Logger> logger);
    
    // Set input data for image encoder
    int32_t setImageInputData(GenieNode_IOName_t nodeIOName,
                              const void* imageData,
                              size_t imageSize,
                              std::shared_ptr<ProfileStat> profileStat);
    
    // Set callback for embedding output
    int32_t setEmbeddingOutputCallback(GenieNode_IOName_t nodeIOName,
                                    GenieNode_EmbeddingOutputCallback_t callback);
    
    // Execute encoding
    int32_t execute(void* userData, std::shared_ptr<ProfileStat> profileStat);
    
    // LoRA support
    int32_t applyLora(std::string loraAdapterName,
                     std::string engine,
                     std::shared_ptr<ProfileStat> profileStat);
    
private:
    std::shared_ptr<genie::Embedding> m_encoder;  // Underlying encoder
    std::unordered_map<std::string,
                        GenieNode_IOName_t> m_inputIOMap;
    
    // Vision parameters from config
    uint32_t m_height;
    uint32_t m_width;
};
```

## Input Processing Flow

```cpp
// From ImageEncoder.cpp::setImageInputData()

int32_t ImageEncoder::setImageInputData(...) {
    // 1. Store input in map
    const uint8_t* dataPtr = static_cast<const uint8_t*>(imageData);
    m_input[inputName] = std::vector<uint8_t>(dataPtr, 
                                              dataPtr + imageSize);
    
    // 2. When all inputs received, encode
    if (m_input.size() == m_inputIOMap.size()) {
        // 3. Call encoder engine
        status = m_encoder->encode(m_input, m_data, nullptr);
        
        // 4. Handle embedding output
        if (status == GENIE_STATUS_SUCCESS) {
            // Get output dimensions
            std::vector<uint32_t> dimensions;
            m_encoder->getOutputDimensions(dimensions);
            uint32_t embeddingSize = dimensions.back();  // Last dimension
            
            // 5. Set vision parameters for accumulator
            uint32_t visionPos = m_pipeline->m_accumulator->getTokenNum();
            m_pipeline->m_accumulator->setVisionParam(visionPos, 1, 
                                                       m_height, m_width);
            
            // 6. Append to accumulator
            m_pipeline->m_accumulator->append(m_data.data(), 
                                                 outputDataType,
                                                 outputScale,
                                                 outputOffset,
                                                 numElements,
                                                 embeddingTokenNum);
        }
        
        // 7. Clear input buffer
        m_input.clear();
    }
    
    return status;
}
```

## Configuration

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
            "pooled-output": false,           // Keep full embeddings
            "disable-kv-cache": true        // No KV cache for encoder
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

## qualla::ImageEncoder (Internal)

```cpp
// From qualla/encoders/image-encoders/imageEncoder.cpp

ImageEncoder::ImageEncoder(std::shared_ptr<Env> env, 
                           const qualla::json& json)
    : Encoder(env, "ImageEncoder", json) {
    
    // Create dummy context (required by engine)
    std::unique_ptr<Context> ctx = 
        Context::create(_env, _type, json, "context", {});
    
    // Create HTP engine
    const qualla::json& eng_conf = json["engine"];
    _engine = Engine::create(*ctx, eng_conf);
    
    // Get output dimensions
    _engine->getTensorDimensions(LayerType::OUTPUT, _output_dimensions);
    
    // Check engine supports embeddings
    using FF = Engine::Feature::Flags;
    if (!_engine->supports(FF::OUTPUT_EMBEDDINGS))
        throw std::runtime_error("engine must output embeddings");
    
    // Get performance profile
    _engine->getPerfProfile(m_defaultPerfProfile);
}

bool ImageEncoder::process(
    const std::unordered_map<std::string, std::vector<uint8_t>>& inputs,
    std::vector<uint8_t>& outputs) {
    
    State::clear();
    
    // Run inference
    size_t n = _engine->process(inputs, outputs);
    
    if (!n) {
        State::error("engine image encoder failed");
        return false;
    }
    
    return true;
}

bool ImageEncoder::encode(
    const std::unordered_map<std::string, std::vector<uint8_t>>& inputs,
    std::vector<uint8_t>& image_features) {
    return process(inputs, image_features);
}
```

## Supported Models

| Vision Model | Encoder Type | Location |
|-------------|--------------|----------|
| SigLIP | CLIP-based | `glm-4v/siglip.json` |
| CLIP | ViT/ResNet | Generic examples |
| ConvNext | Custom | Custom exports |

## Image Preprocessing

Typical preprocessing steps before encoding:

```python
# 1. Resize to model size
img = Image.open(image_path).resize((1024, 1024))

# 2. Convert to RGB
img = img.convert("RGB")

# 3. Normalize (depends on model)
# For INT8 quantized:
arr = np.asarray(img, dtype=np.uint8)

# For FP16:
arr = (np.asarray(img, dtype=np.float32) / 255.0).transpose(2, 0, 1)  # HWC -> CHW
arr = np.expand_dims(arr, axis=0)  # Add batch

# Save as raw
arr.tofile("pixel_values.raw")
```

## Vision Parameter Management

The encoder passes vision parameters to the accumulator for the LLM:

```cpp
// From Accumulator (internal)
void Accumulator::setVisionParam(uint32_t position, 
                               uint32_t batch,
                               uint32_t height, 
                               uint32_t width) {
    m_visionPos = position;
    m_hasVision = true;
    m_visionHeight = height;
    m_visionWidth = width;
}
```

This ensures the LLM knows:
1. Where vision embeddings start in the sequence
2. The spatial dimensions (for RoPE or position encoding)

## Output Flow

```
Image Input (1024x1024x3)
    │
    ▼
ImageEncoder (NPU - qnn-net-run)
    │ Input: pixel_values
    │
    ▼
Vision Embeddings [1, num_tokens, 1024]
    │
    ▼
Accumulator Buffer
    │ Stores: vision_emb + text_emb
    │
    ▼
Text Generator (LLM)
    │ Input: Combined embeddings
```

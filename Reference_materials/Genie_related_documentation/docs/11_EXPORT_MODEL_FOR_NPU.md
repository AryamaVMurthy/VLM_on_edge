# Export FastVLM for NPU

## Overview

Exporting FastVLM involves creating ONNX models that can be compiled to QNN context binaries for HTP execution.

## Model Architecture

```
FastVLM (Apple)
│
├── Vision Encoder (SigLIP/CLIP)
│   Input:  Image [B, 3, 1024, 1024]
│   Output:  Vision embeddings [1, N_vision, 1024]
│
├── Projection Layer (Connector)
│   Input:  Vision embeddings [1, N_vision, 1024]
│   Output:  Projected embeddings [1, N_vision, 896]
│
└── Text Decoder (Qwen2-based)
    Input:  [Projected vision, Text embeddings] [1, N_total, 896]
    Output:  Logits [1, N_vocab]
```

## Export Script Structure

```python
#!/usr/bin/env python3
"""
FastVLM Export Script for NPU
Exports vision encoder, projection, and decoder to ONNX
"""
import torch
from fastvlm import FastVLMForConditionalGeneration

def export_model(model_dir, output_dir, image_size=1024):
    # 1. Load FastVLM
    print(f"Loading FastVLM from {model_dir}")
    model = FastVLMForConditionalGeneration.from_pretrained(model_dir)
    model.eval()
    
    # 2. Export Vision Encoder separately
    print("Exporting vision encoder...")
    export_vision_encoder(model, output_dir, image_size)
    
    # 3. Export with Projection (CRITICAL for all-NPU)
    print("Exporting decoder with projection...")
    export_decoder_with_projection(model, output_dir, image_size)
    
    # 4. Export Token Embedder
    print("Exporting text embedder...")
    export_text_embedder(model, output_dir)
    
    # 5. Extract LUT
    print("Extracting embedding LUT...")
    extract_lut(model, output_dir)
    
    print(f"Export complete! Artifacts in {output_dir}")

def export_vision_encoder(model, output_dir, image_size):
    """Export vision encoder to ONNX"""
    # Create dummy inputs
    pixel_values = torch.randn(1, 3, image_size, image_size)
    
    # Export
    torch.onnx.export(
        model.vision_model,
        (pixel_values,),
        f"{output_dir}/fastvlm_vision_encoder.onnx",
        input_names=["pixel_values"],
        output_names=["image_features"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"}
        },
        opset_version=17
    )

def export_decoder_with_projection(model, output_dir, image_size):
    """Export decoder WITH projection layer for NPU execution"""
    
    # Create combined model class
    class FastVLMDecoderWithProjection(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.vision_projection = model.visual_projection
            self.language_model = model.language_model
            
        def forward(self, vision_features, input_ids):
            # Project vision to LLM space
            projected_vision = self.vision_projection(vision_features)
            
            # Get text embeddings
            text_embeddings = self.language_model.model.embed_tokens(input_ids)
            
            # Concatenate: vision FIRST (critical!)
            combined = torch.cat([projected_vision, text_embeddings], dim=1)
            
            # Forward through decoder
            outputs = self.language_model(
                inputs_embeds=combined,
                use_cache=False
            )
            return outputs
    
    # Create wrapper model
    decoder_model = FastVLMDecoderWithProjection(model)
    decoder_model.eval()
    
    # Create dummy inputs
    batch_size = 1
    num_vision_tokens = 256  # Depends on model
    num_text_tokens = 512
    vision_features = torch.randn(batch_size, num_vision_tokens, model.config.vision_config.hidden_size)
    input_ids = torch.randint(0, model.config.vocab_size, (batch_size, num_text_tokens))
    
    # Export to ONNX
    torch.onnx.export(
        decoder_model,
        (vision_features, input_ids),
        f"{output_dir}/fastvlm_decoder_with_projection.onnx",
        input_names=["vision_features", "input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "vision_features": {0: "batch_size", 1: "num_vision_tokens"},
            "input_ids": {0: "batch_size", 1: "num_text_tokens"}
        },
        opset_version=17
    )

def export_text_embedder(model, output_dir):
    """Export token embedder (LUT component)"""
    
    class TokenEmbedder(torch.nn.Module):
        def __init__(self, embedding):
            super().__init__()
            self.embedding = embedding
            
        def forward(self, input_ids):
            return self.embedding(input_ids)
    
    embedder = TokenEmbedder(model.get_input_embeddings())
    embedder.eval()
    
    input_ids = torch.randint(0, model.config.vocab_size, (1, 512))
    
    torch.onnx.export(
        embedder,
        input_ids,
        f"{output_dir}/fastvlm_token_embedder.onnx",
        input_names=["input_ids"],
        output_names=["embeddings"],
        dynamic_axes={"input_ids": {0: "batch_size", 1: "seq_len"}},
        opset_version=17
    )

def extract_lut(model, output_dir):
    """Extract embedding table as LUT.bin"""
    embeddings = model.get_input_embeddings().weight.detach().cpu().numpy()
    
    # Save as raw binary
    embeddings.tofile(f"{output_dir}/embedding_float32_lut.bin")
    
    # Also save quantized version
    scale = 1.0 / embeddings.abs().max()
    embeddings_int8 = (embeddings * scale).astype(np.int8)
    embeddings_int8.tofile(f"{output_dir}/embedding_int8_lut.bin")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=1024)
    args = parser.parse_args()
    
    export_model(args.model_dir, args.output_dir, args.image_size)
```

## Critical Export Decisions

### 1. Projection Layer Must Be Compiled With Decoder

For all-NPU execution, the projection layer **cannot be separate** - it must be part of the decoder graph:

```
WRONG (would require CPU):
  Vision Encoder (NPU) → Projection (CPU) → Decoder (NPU)

CORRECT (all NPU):
  Vision Encoder (NPU) → [Projection + Decoder] (NPU, single graph)
```

### 2. Vision Embeddings Must Be Input to Decoder

The exported decoder must accept:
```python
def forward(self, vision_features, input_ids):
    # vision_features: [batch, num_vision_tokens, 1024]
    # input_ids: [batch, num_text_tokens]
    
    # Both go into combined sequence
    combined = self.concatenate(vision_features, text_embeddings)
    ...
```

### 3. Dynamic Axes for Flexibility

```python
torch.onnx.export(
    model,
    (vision_features, input_ids),
    "output.onnx",
    dynamic_axes={
        "vision_features": {
            0: "batch_size",        # Dynamic batch
            1: "num_vision_tokens"  # Dynamic vision tokens
        },
        "input_ids": {
            0: "batch_size",        # Dynamic batch
            1: "num_text_tokens"   # Dynamic text tokens
        }
    }
)
```

This allows the same binary to handle:
- Different batch sizes (1, 2, 4, etc.)
- Different image resolutions (different vision token counts)
- Different prompt lengths

### 4. Opset Version

Use **Opset 17** for QNN compatibility:
- QNN best supports ONNX Opset 14-17
- Opset 17 provides operators used by modern VLMs
- Avoid custom ops that aren't supported

## Verification

After export, verify the ONNX files:

```bash
# Check structure
pip install onnx
python -c "
import onnx
model = onnx.load('fastvlm_decoder_with_projection.onnx')
print(f'Inputs: {[i.name for i in model.graph.input]}')
print(f'Outputs: {[o.name for o in model.graph.output]}')
print(f'Nodes: {len(model.graph.node)}')
"

# Check for unsupported ops
qnn-onnx-converter \
    --input_network fastvlm_decoder_with_projection.onnx \
    --output_path converted.json

# If errors, fix model (e.g., remove unsupported ops)
```

## Artifacts After Export

```
fastvlm_npu_export/
├── fastvlm_vision_encoder.onnx          # Vision encoder
├── fastvlm_decoder_with_projection.onnx  # Decoder WITH projection
├── fastvlm_token_embedder.onnx          # Token embedder
├── embedding_float32_lut.bin            # FP32 LUT
├── embedding_int8_lut.bin              # INT8 LUT
└── tokenizer.json                      # From original model
```

## Common Issues and Solutions

### Issue: Dynamic Dimension Mismatch

**Problem**: Vision tokens count varies with image size, but ONNX expects fixed dimension.

**Solution**: Ensure dynamic axes are properly set:

```python
dynamic_axes={
    "vision_features": {1: "num_vision_tokens"}  # Dynamic
}
```

### Issue: Operator Not Supported

**Problem**: `Gather`, `Scatter`, or custom ops not in QNN.

**Solution**: Replace with QNN-supported ops:
```python
# Replace non-supported op
output = torch.gather(input, indices)  # Not supported
# → Use index_select (supported)
output = input.index_select(dim, indices)
```

### Issue: Projection Dimension Mismatch

**Problem**: Vision projection outputs wrong dimension for LLM.

**Solution**: Verify projection output:
```python
assert model.visual_projection.out_features == model.config.hidden_size
```

## Next Steps

1. **Compile to QNN**: See [12_COMPILE_TO_QNN.md](./12_COMPILE_TO_QNN.md)
2. **Verify binaries**: Use `qnn-context-binary-utility` to inspect
3. **Test on device**: Push to device and run inference

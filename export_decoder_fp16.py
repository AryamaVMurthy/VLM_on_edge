#!/usr/bin/env python3
"""
FP16 Version: Export decoder in single ONNX file (no external data).

Includes a fixed-shape KV-cache interface and a static 4D attention mask
to avoid dynamic indexing ops that are not supported in QNN.
"""
import os
import sys
import torch
import argparse
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.cache_utils import DynamicCache

def log(msg):
    print(msg)
    sys.stdout.flush()

def export_full_model_fp16(model_dir, output_dir, cache_len=1023, output_name="fastvlm_full_fp16.onnx"):
    log("Loading full model (FP16)...")
    
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    
    config = Qwen2Config.from_pretrained(model_dir)
    config.model_type = "qwen2"
    
    # FP16 Load
    model = Qwen2ForCausalLM.from_pretrained(
        model_dir,
        config=config,
        torch_dtype=torch.float16,
        trust_remote_code=False,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
    model.eval()
    
    num_layers = config.num_hidden_layers
    log(f"✓ Model loaded: {num_layers} layers")
    
    # Wrapper
    class DecoderFixedKV(torch.nn.Module):
        def __init__(self, model, cache_len, num_layers):
            super().__init__()
            self.model = model
            self.cache_len = cache_len
            self.num_layers = num_layers
        
        def forward(self, inputs_embeds, attention_mask, position_ids, *past_key_values):
            # Create DynamicCache
            # Input KV cache:
            #   key:   (batch, num_heads, head_dim, seq_len)  -> transpose to PyTorch
            #   value: (batch, num_heads, seq_len, head_dim)  -> already PyTorch
            pkv = DynamicCache()
            for i in range(0, len(past_key_values), 2):
                layer_idx = i // 2
                k = past_key_values[i].transpose(2, 3)
                v = past_key_values[i + 1]
                pkv.update(k, v, layer_idx)
            
            # Mask surgery: build a static 4D additive mask to avoid dynamic gather ops.
            # This bypasses _prepare_decoder_attention_mask which triggers GatherND.
            # Input shape: (batch, 1, 1, seq_len)
            dtype = inputs_embeds.dtype
            min_dtype = -65504.0 if dtype == torch.float16 else -3.4e38
            
            mask_4d = attention_mask.to(dtype)
            # 1.0 -> 0.0 (keep), 0.0 -> -min (discard)
            mask_4d = (1.0 - mask_4d) * min_dtype
            
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=mask_4d,  # Pass 4D mask directly
                position_ids=position_ids,
                past_key_values=pkv,
                use_cache=True,
                return_dict=True
            )
            
            # Extract present KV cache and truncate/reshape.
            out = (outputs.logits,)
            n_process = inputs_embeds.shape[1]
            for i in range(self.num_layers):
                k, v = outputs.past_key_values[i]
                # Slice to only new tokens (AR-N)
                k = k[:, :, -n_process:, :]
                v = v[:, :, -n_process:, :]
                # Output format:
                #   key:   (batch, num_heads, head_dim, seq_len)
                #   value: (batch, num_heads, seq_len, head_dim)
                k = k.transpose(2, 3)
                out += (k, v)
            return out
    
    decoder = DecoderFixedKV(model, cache_len, num_layers)
    
    # Prepare FP16 inputs
    hidden_size = config.hidden_size
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = hidden_size // config.num_attention_heads
    
    decode_len = 1
    inputs_embeds = torch.randn(1, decode_len, hidden_size, dtype=torch.float16) # FP16
    attention_mask = torch.ones(1, 1, 1, cache_len + decode_len, dtype=torch.int64)
    position_ids = torch.arange(0, decode_len, dtype=torch.int64).unsqueeze(0)
    
    past = []
    for _ in range(num_layers):
        # Genie expects KV cache in (batch, num_heads, head_dim, seq_len) format
        # Standard PyTorch is (batch, num_heads, seq_len, head_dim)
        # So we create transposed tensors
        past.extend([
            torch.zeros(1, num_kv_heads, head_dim, cache_len, dtype=torch.float16),
            torch.zeros(1, num_kv_heads, cache_len, head_dim, dtype=torch.float16)
        ])
    
    input_names = ["inputs_embeds", "attention_mask", "position_ids"]
    for i in range(num_layers):
        input_names.extend([f"past_key_values.{i}.key", f"past_key_values.{i}.value"])
    
    output_names = ["logits"]
    for i in range(num_layers):
        output_names.extend([f"present.{i}.key", f"present.{i}.value"])
    
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, output_name)
    
    log("Exporting to ONNX (FP16)...")
    try:
        with torch.no_grad():
            torch.onnx.export(
                decoder,
                (inputs_embeds, attention_mask, position_ids, *past),
                onnx_path,
                input_names=input_names,
                output_names=output_names,
                opset_version=18,
                do_constant_folding=True,
                export_params=True  # Will be <2GB, so single file
            )
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        log(f"✓ ONNX exported: {size_mb:.1f} MB")
        return True
    except Exception as e:
        log(f"✗ Export failed: {e}")
        import traceback
        log(traceback.format_exc())
        return False

def parse_args():
    parser = argparse.ArgumentParser(description="Export FastVLM decoder to FP16 ONNX.")
    parser.add_argument("--model-dir", required=True, help="Path to model checkpoint directory.")
    parser.add_argument("--output-dir", default=".", help="Directory to write ONNX.")
    parser.add_argument("--cache-len", type=int, default=1023, help="KV cache length.")
    parser.add_argument("--output-name", default="fastvlm_full_fp16.onnx", help="ONNX filename.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ok = export_full_model_fp16(
        args.model_dir,
        args.output_dir,
        cache_len=args.cache_len,
        output_name=args.output_name,
    )
    sys.exit(0 if ok else 1)

#!/usr/bin/env python3
import os

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import argparse
import torch
from transformers import AutoModelForCausalLM


def build_decoder_fixed_kv(model, cache_len: int):
    class DecoderFixedKV(torch.nn.Module):
        def __init__(self, model, cache_len: int):
            super().__init__()
            self.model = model
            self.cache_len = cache_len

        def forward(self, inputs_embeds, attention_mask, position_ids, *past_key_values):
            pkv = []
            for i in range(0, len(past_key_values), 2):
                pkv.append((past_key_values[i], past_key_values[i + 1]))
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=tuple(pkv),
                use_cache=True,
                return_dict=True,
            )
            out = (outputs.logits,)
            for k, v in outputs.past_key_values:
                k = k[:, :, -self.cache_len :, :]
                v = v[:, :, -self.cache_len :, :]
                out += (k, v)
            return out

    return DecoderFixedKV(model, cache_len)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache-len", type=int, default=512)
    parser.add_argument("--decode-len", type=int, default=1)
    parser.add_argument("--opset", type=int, default=14)
    parser.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    args = parser.parse_args()

    dtype = torch.float32 if args.dtype == "fp32" else torch.float16

    torch.set_grad_enabled(False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    decoder = build_decoder_fixed_kv(model, args.cache_len)
    decoder.eval()

    inputs_embeds = torch.randn((1, args.decode_len, model.config.hidden_size), dtype=dtype)
    attention_mask = torch.ones((1, args.cache_len + args.decode_len), dtype=torch.int64)
    position_ids = torch.arange(0, args.decode_len, dtype=torch.int64).unsqueeze(0)

    past = []
    for _ in range(num_layers):
        past_k = torch.zeros((1, num_kv_heads, args.cache_len, head_dim), dtype=dtype)
        past_v = torch.zeros((1, num_kv_heads, args.cache_len, head_dim), dtype=dtype)
        past.extend([past_k, past_v])

    input_names = ["inputs_embeds", "attention_mask", "position_ids"]
    for i in range(num_layers):
        input_names.append(f"past_key_values.{i}.key")
        input_names.append(f"past_key_values.{i}.value")

    output_names = ["logits"]
    for i in range(num_layers):
        output_names.append(f"present.{i}.key")
        output_names.append(f"present.{i}.value")

    torch.onnx.export(
        decoder,
        (inputs_embeds, attention_mask, position_ids, *past),
        args.out,
        input_names=input_names,
        output_names=output_names,
        opset_version=args.opset,
        do_constant_folding=False,
    )
    print(f"Exported decoder to {args.out}")


if __name__ == "__main__":
    main()

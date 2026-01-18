#!/usr/bin/env python3
import os

# Prevent Transformers from importing TensorFlow (installed but unused here).
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import argparse
import torch
from transformers import AutoModelForCausalLM
import time


def _ensure_vision_loaded(model):
    vision = model.get_model().get_vision_tower()
    if hasattr(vision, "is_loaded") and not vision.is_loaded:
        vision.load_model()
    return vision


def build_vision_encoder(model):
    class VisionEncoderWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.vision_tower = _ensure_vision_loaded(model)
            self.mm_projector = model.get_model().mm_projector

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            feats = self.vision_tower(pixel_values)
            feats = self.mm_projector(feats)
            return feats

    return VisionEncoderWrapper(model)


def build_embed_tokens(model):
    class EmbedTokensWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.embed_tokens = model.get_model().embed_tokens

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return self.embed_tokens(input_ids)

    return EmbedTokensWrapper(model)


def build_decoder_prefill(model):
    class DecoderPrefillWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(
            self,
            inputs_embeds: torch.Tensor,
            attention_mask: torch.Tensor,
            position_ids: torch.Tensor,
        ):
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
                return_dict=True,
            )
            out = (outputs.logits,)
            for k, v in outputs.past_key_values:
                out += (k, v)
            return out

    return DecoderPrefillWrapper(model)


def build_decoder_decode(model):
    class DecoderDecodeWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(
            self,
            inputs_embeds: torch.Tensor,
            attention_mask: torch.Tensor,
            position_ids: torch.Tensor,
            *past_key_values,
        ):
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
                out += (k, v)
            return out

    return DecoderDecodeWrapper(model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--opset", type=int, default=14)
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument(
        "--only",
        choices=["all", "vision", "embed", "prefill", "decode"],
        default="all",
    )
    parser.add_argument(
        "--no-constant-folding",
        action="store_true",
        help="Disable constant folding to speed up export.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=0,
        help="Set torch.set_num_threads (0 = leave default).",
    )
    parser.add_argument(
        "--torch-inter-threads",
        type=int,
        default=0,
        help="Set torch.set_num_interop_threads (0 = leave default).",
    )
    parser.add_argument(
        "--verbose-onnx",
        action="store_true",
        help="Enable verbose torch.onnx export logging.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    if args.torch_inter_threads > 0:
        torch.set_num_interop_threads(args.torch_inter_threads)
    torch.set_grad_enabled(False)
    print("Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded.", flush=True)

    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    if args.only in ("all", "vision"):
        print(
            f"Exporting vision encoder: image_size={args.image_size}, dtype={args.dtype}",
            flush=True,
        )
        vision = build_vision_encoder(model)
        vision.eval()
        pixel_values = torch.randn(
            1, 3, args.image_size, args.image_size, dtype=dtype
        )
        vision_path = os.path.join(
            args.out_dir,
            "vision_encoder_fp16.onnx" if args.dtype == "fp16" else "vision_encoder_fp32.onnx",
        )
        t0 = time.time()
        torch.onnx.export(
            vision,
            (pixel_values,),
            vision_path,
            input_names=["pixel_values"],
            output_names=["image_features"],
            opset_version=args.opset,
            do_constant_folding=not args.no_constant_folding,
            verbose=args.verbose_onnx,
        )
        print(f"Vision exported in {time.time() - t0:.1f}s: {vision_path}", flush=True)

    if args.only in ("all", "embed"):
        print(f"Exporting embed tokens: seq_len={args.seq_len}", flush=True)
        embed = build_embed_tokens(model)
        embed.eval()
        input_ids = torch.zeros((1, args.seq_len), dtype=torch.int64)
        embed_path = os.path.join(
            args.out_dir,
            "embed_tokens_fp16.onnx" if args.dtype == "fp16" else "embed_tokens_fp32.onnx",
        )
        t0 = time.time()
        torch.onnx.export(
            embed,
            (input_ids,),
            embed_path,
            input_names=["input_ids"],
            output_names=["inputs_embeds"],
            opset_version=args.opset,
            do_constant_folding=not args.no_constant_folding,
            verbose=args.verbose_onnx,
        )
        print(f"Embed exported in {time.time() - t0:.1f}s: {embed_path}", flush=True)

    if args.only in ("all", "prefill"):
        print(f"Exporting decoder prefill: seq_len={args.seq_len}", flush=True)
        inputs_embeds = torch.randn(
            (1, args.seq_len, model.config.hidden_size), dtype=dtype
        )
        attention_mask = torch.ones((1, args.seq_len), dtype=torch.int64)
        position_ids = torch.arange(0, args.seq_len, dtype=torch.int64).unsqueeze(0)

        prefill = build_decoder_prefill(model)
        prefill.eval()
        prefill_path = os.path.join(
            args.out_dir,
            "decoder_prefill_fp16.onnx" if args.dtype == "fp16" else "decoder_prefill_fp32.onnx",
        )

        prefill_output_names = ["logits"]
        for i in range(num_layers):
            prefill_output_names.append(f"present.{i}.key")
            prefill_output_names.append(f"present.{i}.value")

        t0 = time.time()
        torch.onnx.export(
            prefill,
            (inputs_embeds, attention_mask, position_ids),
            prefill_path,
            input_names=["inputs_embeds", "attention_mask", "position_ids"],
            output_names=prefill_output_names,
            opset_version=args.opset,
            do_constant_folding=not args.no_constant_folding,
            verbose=args.verbose_onnx,
        )
        print(f"Prefill exported in {time.time() - t0:.1f}s: {prefill_path}", flush=True)

    if args.only in ("all", "decode"):
        print(f"Exporting decoder decode: past_len={args.seq_len}", flush=True)
        past_len = args.seq_len
        decode_inputs_embeds = torch.randn(
            (1, 1, model.config.hidden_size), dtype=dtype
        )
        decode_attention_mask = torch.ones((1, past_len + 1), dtype=torch.int64)
        decode_position_ids = torch.tensor([[past_len]], dtype=torch.int64)

        past = []
        for _ in range(num_layers):
            past_k = torch.zeros(
                (1, num_kv_heads, past_len, head_dim), dtype=dtype
            )
            past_v = torch.zeros(
                (1, num_kv_heads, past_len, head_dim), dtype=dtype
            )
            past.extend([past_k, past_v])

        decode = build_decoder_decode(model)
        decode.eval()
        decode_path = os.path.join(
            args.out_dir,
            "decoder_decode_fp16.onnx" if args.dtype == "fp16" else "decoder_decode_fp32.onnx",
        )

        decode_input_names = ["inputs_embeds", "attention_mask", "position_ids"]
        for i in range(num_layers):
            decode_input_names.append(f"past_key_values.{i}.key")
            decode_input_names.append(f"past_key_values.{i}.value")

        decode_output_names = ["logits"]
        for i in range(num_layers):
            decode_output_names.append(f"present.{i}.key")
            decode_output_names.append(f"present.{i}.value")

        t0 = time.time()
        torch.onnx.export(
            decode,
            (decode_inputs_embeds, decode_attention_mask, decode_position_ids, *past),
            decode_path,
            input_names=decode_input_names,
            output_names=decode_output_names,
            opset_version=args.opset,
            do_constant_folding=not args.no_constant_folding,
            verbose=args.verbose_onnx,
        )
        print(f"Decode exported in {time.time() - t0:.1f}s: {decode_path}", flush=True)


if __name__ == "__main__":
    main()

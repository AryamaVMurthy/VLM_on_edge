#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np


def load_meta(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def combine(
    vision_raw: str,
    text_raw: str,
    text_meta: str,
    out_dir: str,
    max_prompt_tokens: int,
    vision_tokens: int,
    vision_stride: int,
    add_bos: bool,
    lut_path: str,
    bos_token: int,
    order: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    vision = np.fromfile(vision_raw, dtype=np.float32).reshape(1, 256, 896)
    text = np.fromfile(text_raw, dtype=np.float32).reshape(1, 512, 896)

    meta = load_meta(text_meta)
    text_count = int(meta.get("token_count", 0))
    text_count = min(text_count, text.shape[1])
    image_token_pos = meta.get("image_token_pos", None)

    # Select vision tokens (subsample if stride > 1).
    vision_sel = vision[:, ::vision_stride, :]
    if vision_tokens == 0:
        vision_sel = vision[:, :0, :]
    elif vision_tokens > 0:
        vision_sel = vision_sel[:, :vision_tokens, :]

    # Trim text to fit remaining budget.
    remaining = max_prompt_tokens - vision_sel.shape[1]
    if remaining < 0:
        remaining = 0
    text_sel = text[:, : min(text_count, remaining), :]

    # Match decoder input expectations: float32 when using float32 LUTs, fp16 otherwise.
    output_dtype = np.float16 if lut_path.endswith("_fp16.bin") else np.float32
    if image_token_pos is not None:
        insert_at = int(image_token_pos)
        insert_at = max(0, min(insert_at, text_sel.shape[1]))
        text_pre = text_sel[:, :insert_at, :]
        text_post = text_sel[:, insert_at:, :]
        combined = np.concatenate([text_pre, vision_sel, text_post], axis=1).astype(output_dtype)
    elif order == "text-vision":
        combined = np.concatenate([text_sel, vision_sel], axis=1).astype(output_dtype)
    else:
        combined = np.concatenate([vision_sel, text_sel], axis=1).astype(output_dtype)

    if add_bos:
        if not os.path.isfile(lut_path):
            raise FileNotFoundError(f"LUT file not found: {lut_path}")
        # Heuristic: fp16 LUTs are named *_fp16.bin; otherwise assume fp32.
        lut_dtype = np.float16 if lut_path.endswith("_fp16.bin") else np.float32
        lut = np.fromfile(lut_path, dtype=lut_dtype)
        if lut.size % combined.shape[2] != 0:
            raise ValueError("LUT size is not divisible by embedding dim.")
        vocab = lut.size // combined.shape[2]
        if not (0 <= bos_token < vocab):
            raise ValueError(f"bos_token {bos_token} out of range (vocab={vocab}).")
        lut = lut.reshape(vocab, combined.shape[2])
        bos = lut[bos_token].astype(combined.dtype)[None, None, :]
        combined = np.concatenate([bos, combined], axis=1)

    combined_path = os.path.join(out_dir, "combined_embeddings.raw")
    combined.tofile(combined_path)

    # Split into per-token files for sequential prefill.
    token_dir = os.path.join(out_dir, "prefill_tokens")
    os.makedirs(token_dir, exist_ok=True)
    for name in os.listdir(token_dir):
        if name.startswith("token_") and name.endswith(".raw"):
            os.remove(os.path.join(token_dir, name))
    for idx in range(combined.shape[1]):
        token = combined[:, idx : idx + 1, :]
        token.tofile(os.path.join(token_dir, f"token_{idx:04d}.raw"))

    info = {
        "vision_tokens": int(vision_sel.shape[1]),
        "text_tokens": int(text_sel.shape[1]),
        "total_tokens": int(combined.shape[1]),
        "max_prompt_tokens": int(max_prompt_tokens),
        "vision_stride": int(vision_stride),
        "bos_added": bool(add_bos),
        "bos_token": int(bos_token) if add_bos else None,
        "output_dtype": "float16" if output_dtype == np.float16 else "float32",
        "image_token_pos": image_token_pos,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print(f"Wrote {combined_path}")
    print(f"Wrote {token_dir}/token_*.raw")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine vision + text embeddings.")
    parser.add_argument("--vision-raw", required=True, help="Vision output raw file.")
    parser.add_argument("--text-raw", required=True, help="Text output raw file.")
    parser.add_argument("--text-meta", required=True, help="Text meta.json path.")
    parser.add_argument("--out-dir", default="host_outputs/combined", help="Output directory.")
    parser.add_argument("--max-prompt-tokens", type=int, default=128, help="Max prompt tokens (KV cache).")
    parser.add_argument("--vision-tokens", type=int, default=64, help="Vision tokens to keep.")
    parser.add_argument("--vision-stride", type=int, default=4, help="Stride for vision token sampling.")
    parser.add_argument("--add-bos", action="store_true", help="Prepend BOS embedding from LUT.")
    parser.add_argument("--lut", default="embedding_fp16.bin", help="Embedding LUT file.")
    parser.add_argument("--bos-token", type=int, default=151643, help="BOS token id.")
    parser.add_argument(
        "--order",
        choices=["vision-text", "text-vision"],
        default="vision-text",
        help="Concatenation order for vision and text embeddings.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    combine(
        args.vision_raw,
        args.text_raw,
        args.text_meta,
        args.out_dir,
        args.max_prompt_tokens,
        args.vision_tokens,
        args.vision_stride,
        args.add_bos,
        args.lut,
        args.bos_token,
        args.order,
    )

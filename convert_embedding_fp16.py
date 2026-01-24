#!/usr/bin/env python3
import argparse
import os
import numpy as np


def convert(src_path: str, dst_path: str, vocab_size: int, emb_dim: int) -> None:
    count = vocab_size * emb_dim
    data = np.fromfile(src_path, dtype=np.float32, count=count)
    if data.size != count:
        raise ValueError(f"Expected {count} floats, got {data.size}")
    data = data.reshape((vocab_size, emb_dim)).astype(np.float16)
    data.tofile(dst_path)
    size_mb = os.path.getsize(dst_path) / (1024 * 1024)
    print(f"Wrote {dst_path} ({size_mb:.1f} MB)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert embedding.bin float32 -> float16")
    parser.add_argument("--src", default="embedding.bin", help="Source embedding.bin (float32)")
    parser.add_argument("--dst", default="embedding_fp16.bin", help="Destination embedding_fp16.bin")
    parser.add_argument("--vocab-size", type=int, default=151936, help="Vocabulary size")
    parser.add_argument("--emb-dim", type=int, default=896, help="Embedding dimension")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(args.src, args.dst, args.vocab_size, args.emb_dim)

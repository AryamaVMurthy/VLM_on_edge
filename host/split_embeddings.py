#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def parse_dtype(name: str) -> np.dtype:
    name = name.lower()
    if name in ("float16", "fp16"):
        return np.float16
    if name in ("float32", "fp32"):
        return np.float32
    raise ValueError(f"Unsupported dtype: {name}")


def split_embeddings(raw_path: Path, out_dir: Path, emb_dim: int, dtype: np.dtype, token_count: int | None):
    data = np.fromfile(raw_path, dtype=dtype)
    if data.size % emb_dim != 0:
        raise ValueError(f"Raw size {data.size} not divisible by emb_dim {emb_dim}")
    total_tokens = data.size // emb_dim
    if token_count is None:
        token_count = total_tokens
    if token_count > total_tokens:
        raise ValueError(f"token_count {token_count} exceeds available tokens {total_tokens}")

    out_dir.mkdir(parents=True, exist_ok=True)
    tokens = data.reshape(total_tokens, emb_dim)
    for idx in range(token_count):
        token_path = out_dir / f"token_{idx:04d}.raw"
        tokens[idx].astype(dtype, copy=False).tofile(token_path)

    meta = {
        "raw_path": str(raw_path),
        "embedding_dim": emb_dim,
        "dtype": str(dtype),
        "token_count": int(token_count),
    }
    (out_dir.parent / "meta.json").write_text(json.dumps(meta, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Split embedding raw file into per-token raw files.")
    parser.add_argument("--raw", required=True, help="Path to embedding raw file.")
    parser.add_argument("--out-dir", required=True, help="Output directory for token_*.raw files.")
    parser.add_argument("--emb-dim", type=int, default=896, help="Embedding dimension.")
    parser.add_argument("--dtype", default="float16", help="Datatype (float16/float32).")
    parser.add_argument("--token-count", type=int, default=None, help="Number of tokens to split.")
    args = parser.parse_args()

    split_embeddings(
        Path(args.raw),
        Path(args.out_dir),
        args.emb_dim,
        parse_dtype(args.dtype),
        args.token_count,
    )


if __name__ == "__main__":
    main()

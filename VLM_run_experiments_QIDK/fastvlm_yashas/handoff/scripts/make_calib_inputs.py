#!/usr/bin/env python3
import argparse
import os
import sys
from typing import List

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None


def _load_image(path: str, image_size: int) -> np.ndarray:
    if Image is None:
        raise RuntimeError("Pillow is required. Install it in qnn_env: pip install Pillow")
    img = Image.open(path).convert("RGB")
    img = img.resize((image_size, image_size), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32)
    arr = arr / 255.0
    # HWC -> NCHW
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    return arr


def _write_raw(arr: np.ndarray, path: str) -> None:
    arr.astype(np.float32).tofile(path)


def _expand_inputs(paths: List[str]) -> List[str]:
    out = []
    for p in paths:
        if os.path.isdir(p):
            for name in sorted(os.listdir(p)):
                if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    out.append(os.path.join(p, name))
        else:
            out.append(p)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="*", default=[])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--random", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    raw_paths = []
    if args.random:
        for i in range(args.max_samples):
            arr = np.random.rand(1, 3, args.image_size, args.image_size).astype(np.float32)
            raw_path = os.path.join(args.out_dir, f"sample_{i:03d}.raw")
            _write_raw(arr, raw_path)
            raw_paths.append(raw_path)
    else:
        if not args.images:
            print("No images provided. Use --images or --random.", file=sys.stderr)
            return 1
        image_paths = _expand_inputs(args.images)
        if not image_paths:
            print("No image files found.", file=sys.stderr)
            return 1
        for i, path in enumerate(image_paths[: args.max_samples]):
            arr = _load_image(path, args.image_size)
            raw_path = os.path.join(args.out_dir, f"sample_{i:03d}.raw")
            _write_raw(arr, raw_path)
            raw_paths.append(raw_path)

    input_list_path = os.path.join(args.out_dir, "input_list.txt")
    with open(input_list_path, "w", encoding="utf-8") as f:
        for p in raw_paths:
            f.write(os.path.abspath(p) + "\n")

    print(f"Wrote {len(raw_paths)} samples to {args.out_dir}")
    print(f"Input list: {input_list_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

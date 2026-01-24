#!/usr/bin/env python3
import argparse
import os
import json

import numpy as np
from PIL import Image


def prepare(
    image_path: str,
    out_dir: str,
    image_size: int,
    device_dir: str | None,
    fmt: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    img = Image.open(image_path).convert("RGB")
    img = img.resize((image_size, image_size), Image.BICUBIC)

    if fmt == "float32_nchw":
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
        arr = np.expand_dims(arr, axis=0)   # Add batch
        dtype_label = "float32"
    elif fmt == "float16_nchw":
        arr = np.asarray(img, dtype=np.float16) / 255.0
        arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
        arr = np.expand_dims(arr, axis=0)   # Add batch
        dtype_label = "float16"
    elif fmt == "float32_hwc":
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)  # Add batch
        dtype_label = "float32"
    elif fmt == "float16_hwc":
        arr = np.asarray(img, dtype=np.float16) / 255.0
        arr = np.expand_dims(arr, axis=0)  # Add batch
        dtype_label = "float16"
    elif fmt == "float32_hwc_nobatch":
        arr = np.asarray(img, dtype=np.float32) / 255.0
        dtype_label = "float32"
    elif fmt == "float16_hwc_nobatch":
        arr = np.asarray(img, dtype=np.float16) / 255.0
        dtype_label = "float16"
    elif fmt == "float32_hwc_flat":
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = arr.reshape(-1, 3)
        dtype_label = "float32"
    elif fmt == "float16_hwc_flat":
        arr = np.asarray(img, dtype=np.float16) / 255.0
        arr = arr.reshape(-1, 3)
        dtype_label = "float16"
    elif fmt == "uint8_hwc":
        arr = np.asarray(img, dtype=np.uint8)  # Keep HWC, uint8
        dtype_label = "uint8"
    elif fmt == "uint8_hwc_flat":
        arr = np.asarray(img, dtype=np.uint8)
        arr = arr.reshape(-1, 3)
        dtype_label = "uint8"
    elif fmt == "uint8_nchw":
        arr = np.asarray(img, dtype=np.uint8)
        arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
        arr = np.expand_dims(arr, axis=0)   # Add batch
        dtype_label = "uint8"
    elif fmt == "image_png":
        out_path = os.path.join(out_dir, "image.png")
        img.save(out_path, format="PNG")
        arr = None
        dtype_label = "png"
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    if fmt == "image_png":
        raw_path = os.path.join(out_dir, "image.png")
    else:
        raw_path = os.path.join(out_dir, "pixel_values.raw")
        arr.tofile(raw_path)

    input_list = os.path.join(out_dir, "input_list.txt")
    if device_dir:
        dev_path = f"{device_dir}/inputs/vision/pixel_values.raw"
    else:
        dev_path = "pixel_values.raw"
    with open(input_list, "w", encoding="utf-8") as f:
        f.write(f"pixel_values:={dev_path}\n")

    meta = {
        "image": image_path,
        "shape": list(arr.shape) if arr is not None else [image_size, image_size, 3],
        "dtype": dtype_label,
        "format": fmt,
        "device_dir": device_dir,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {raw_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare vision encoder input.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--out-dir", default="host_inputs/vision", help="Output directory.")
    parser.add_argument("--size", type=int, default=1024, help="Image size (square).")
    parser.add_argument("--device-dir", default=None, help="Device directory for absolute paths.")
    parser.add_argument(
        "--format",
        default="float32_nchw",
        choices=(
            "float32_nchw",
            "float16_nchw",
            "float32_hwc",
            "float16_hwc",
            "float32_hwc_nobatch",
            "float16_hwc_nobatch",
            "float32_hwc_flat",
            "float16_hwc_flat",
            "uint8_hwc",
            "uint8_hwc_flat",
            "uint8_nchw",
            "image_png",
        ),
        help="Output format for pixel_values.raw",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare(args.image, args.out_dir, args.size, args.device_dir, args.format)

#!/usr/bin/env python3
import argparse

import onnx
from onnx import TensorProto


def patch_attention_mask_dtype(in_path: str, out_path: str, dtype: int) -> None:
    model = onnx.load(in_path, load_external_data=False)
    found = False
    for v in model.graph.input:
        if v.name == "attention_mask":
            v.type.tensor_type.elem_type = dtype
            found = True
            break
    if not found:
        raise ValueError("attention_mask input not found")
    onnx.save(model, out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set attention_mask input dtype.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--dtype",
        default="int32",
        choices=["int32", "int64"],
        help="Target dtype for attention_mask.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dtype_map = {"int32": TensorProto.INT32, "int64": TensorProto.INT64}
    patch_attention_mask_dtype(args.input, args.output, dtype_map[args.dtype])

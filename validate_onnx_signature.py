#!/usr/bin/env python3
import argparse
import sys
from typing import Dict, List, Tuple

import onnx


ElemType = int
Shape = Tuple[int, ...]


def _get_shape(dims) -> Shape:
    return tuple(int(d.dim_value) for d in dims)


def _tensor_info(value_info) -> Tuple[Shape, ElemType]:
    tt = value_info.type.tensor_type
    return _get_shape(tt.shape.dim), int(tt.elem_type)


def build_expected_signature(cache_len: int, kv_style: str) -> Dict[str, Tuple[Shape, ElemType]]:
    # ElemType values: 10=float16, 7=int64 (ONNX TensorProto.DataType)
    expected: Dict[str, Tuple[Shape, ElemType]] = {
        "inputs_embeds": ((1, 1, 896), 10),
        "attention_mask": ((1, 1, 1, cache_len + 1), 7),
        "position_ids": ((1, 1), 7),
    }
    for i in range(24):
        if kv_style == "genie":
            expected[f"past_key_{i}_in"] = ((1, 2, 64, cache_len), 10)
            expected[f"past_value_{i}_in"] = ((1, 2, cache_len, 64), 10)
        else:
            expected[f"past_key_values.{i}.key"] = ((1, 2, 64, cache_len), 10)
            expected[f"past_key_values.{i}.value"] = ((1, 2, cache_len, 64), 10)
    return expected


def validate(onnx_path: str, cache_len: int, kv_style: str) -> int:
    model = onnx.load(onnx_path, load_external_data=False)
    expected = build_expected_signature(cache_len, kv_style)

    actual: Dict[str, Tuple[Shape, ElemType]] = {}
    for v in model.graph.input:
        actual[v.name] = _tensor_info(v)

    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))

    mismatched: List[str] = []
    for name, exp in expected.items():
        act = actual.get(name)
        if act is None:
            continue
        if act != exp:
            mismatched.append(f"{name}: expected={exp}, actual={act}")

    if missing or unexpected or mismatched:
        print("Signature validation FAILED")
        if missing:
            print("Missing inputs:")
            for name in missing:
                print(f"  - {name}")
        if unexpected:
            print("Unexpected inputs:")
            for name in unexpected:
                print(f"  - {name}")
        if mismatched:
            print("Mismatched inputs:")
            for line in mismatched:
                print(f"  - {line}")
        return 1

    print("Signature validation OK")
    print(f"Inputs validated: {len(expected)}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate FastVLM decoder ONNX signature.")
    parser.add_argument("--onnx", required=True, help="Path to ONNX file.")
    parser.add_argument("--cache-len", type=int, default=128, help="KV cache length.")
    parser.add_argument("--kv-style", choices=["dot", "genie"], default="dot", help="KV tensor naming style.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(validate(args.onnx, args.cache_len, args.kv_style))

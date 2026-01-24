#!/usr/bin/env python3
import argparse
from pathlib import Path

import onnx


def build_kv_map(num_layers: int) -> dict:
    mapping = {}
    for i in range(num_layers):
        mapping[f"past_key_values.{i}.key"] = f"past_key_{i}_in"
        mapping[f"past_key_values.{i}.value"] = f"past_value_{i}_in"
        mapping[f"present.{i}.key"] = f"past_key_{i}_out"
        mapping[f"present.{i}.value"] = f"past_value_{i}_out"
    return mapping


def rename_value_name(value, mapping: dict) -> None:
    if value.name in mapping:
        value.name = mapping[value.name]


def rename_model(model: onnx.ModelProto, mapping: dict) -> onnx.ModelProto:
    for v in model.graph.input:
        rename_value_name(v, mapping)
    for v in model.graph.output:
        rename_value_name(v, mapping)
    for v in model.graph.value_info:
        rename_value_name(v, mapping)
    for init in model.graph.initializer:
        if init.name in mapping:
            init.name = mapping[init.name]
    for node in model.graph.node:
        for idx, name in enumerate(node.input):
            if name in mapping:
                node.input[idx] = mapping[name]
        for idx, name in enumerate(node.output):
            if name in mapping:
                node.output[idx] = mapping[name]
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Rename KV tensor names for GENIE.")
    parser.add_argument("--input", required=True, help="Input ONNX path.")
    parser.add_argument("--output", required=True, help="Output ONNX path.")
    parser.add_argument("--layers", type=int, default=24, help="Number of layers.")
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)

    model = onnx.load(str(src), load_external_data=True)
    mapping = build_kv_map(args.layers)
    model = rename_model(model, mapping)
    onnx.save_model(model, str(dst))
    print(f"Wrote renamed ONNX: {dst}")


if __name__ == "__main__":
    main()

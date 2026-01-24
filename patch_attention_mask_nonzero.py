#!/usr/bin/env python3
import argparse
import onnx
from onnx import helper


def get_input_dtype(model, name: str) -> int:
    for inp in model.graph.input:
        if inp.name == name:
            return int(inp.type.tensor_type.elem_type)
    raise ValueError(f"Input {name} not found in model inputs.")


def patch_model(in_path: str, out_path: str) -> None:
    model = onnx.load(in_path, load_external_data=False)

    cast_idx = None
    cast_node = None
    for idx, node in enumerate(model.graph.node):
        if node.op_type == "Cast" and node.input and node.input[0] == "attention_mask":
            cast_idx = idx
            cast_node = node
            break
    if cast_node is None:
        raise ValueError("No Cast node found that takes attention_mask as input.")

    orig_out = cast_node.output[0]
    raw_out = f"{orig_out}_raw"
    cast_node.output[0] = raw_out

    mul_node = helper.make_node(
        "Mul",
        inputs=[raw_out, raw_out],
        outputs=[orig_out],
        name=f"{orig_out}_square",
    )

    # Insert new nodes right after the Cast.
    model.graph.node.insert(cast_idx + 1, mul_node)

    onnx.save(model, out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patch attention_mask to treat any non-zero as 1 (square)."
    )
    parser.add_argument("--input", required=True, help="Input ONNX path.")
    parser.add_argument("--output", required=True, help="Output ONNX path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    patch_model(args.input, args.output)

#!/usr/bin/env python3
import argparse
import os
from typing import List

import onnx
from onnx import TensorProto, helper, utils


def _replace_inputs(model: onnx.ModelProto, src: str, dst: str) -> None:
    for node in model.graph.node:
        for idx, name in enumerate(node.input):
            if name == src:
                node.input[idx] = dst


def _add_input(model: onnx.ModelProto, name: str, dtype: int, shape: List[int]) -> None:
    if any(inp.name == name for inp in model.graph.input):
        return
    model.graph.input.append(helper.make_tensor_value_info(name, dtype, shape))


def _add_initializer(model: onnx.ModelProto, name: str, dtype: int, dims: List[int], vals: List[int]) -> None:
    if any(init.name == name for init in model.graph.initializer):
        return
    tensor = helper.make_tensor(name, dtype, dims, vals)
    model.graph.initializer.append(tensor)


def _replace_inputs_except(model: onnx.ModelProto, src: str, dst: str, skip_names) -> None:
    for node in model.graph.node:
        if node.name in skip_names:
            continue
        for idx, name in enumerate(node.input):
            if name == src:
                node.input[idx] = dst


def patch_rope_inputs(
    in_path: str,
    out_path: str,
    rope_dim: int,
    sin_name: str,
    cos_name: str,
    dtype: int,
) -> None:
    model = onnx.load(in_path, load_external_data=False)

    sin_out = None
    cos_out = None
    for node in model.graph.node:
        if node.op_type == "Sin":
            sin_out = node.output[0]
        elif node.op_type == "Cos":
            cos_out = node.output[0]

    if not sin_out or not cos_out:
        raise ValueError("Unable to locate Sin/Cos outputs in the ONNX graph.")

    _replace_inputs(model, sin_out, sin_name)
    _replace_inputs(model, cos_out, cos_name)

    _add_input(model, sin_name, dtype, [1, 1, 1, rope_dim])
    _add_input(model, cos_name, dtype, [1, 1, 1, rope_dim])

    squeeze_axes = "rope_squeeze_axes"
    _add_initializer(model, squeeze_axes, TensorProto.INT64, [1], [2])
    squeeze_sin = f"{sin_name}_squeezed"
    squeeze_cos = f"{cos_name}_squeezed"
    squeeze_sin_node = helper.make_node(
        "Squeeze",
        inputs=[sin_name, squeeze_axes],
        outputs=[squeeze_sin],
        name="rope_squeeze_sin",
    )
    squeeze_cos_node = helper.make_node(
        "Squeeze",
        inputs=[cos_name, squeeze_axes],
        outputs=[squeeze_cos],
        name="rope_squeeze_cos",
    )

    model.graph.node.insert(0, squeeze_sin_node)
    model.graph.node.insert(1, squeeze_cos_node)

    skip_names = {squeeze_sin_node.name, squeeze_cos_node.name}
    _replace_inputs_except(model, sin_name, squeeze_sin, skip_names)
    _replace_inputs_except(model, cos_name, squeeze_cos, skip_names)

    tmp_path = f"{out_path}.tmp.onnx"
    onnx.save(model, tmp_path)

    input_names = [inp.name for inp in model.graph.input if inp.name != "position_ids"]
    output_names = [out.name for out in model.graph.output]
    utils.extract_model(tmp_path, out_path, input_names, output_names)

    os.remove(tmp_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replace rotary Sin/Cos with external inputs.")
    parser.add_argument("--input", required=True, help="Input ONNX path.")
    parser.add_argument("--output", required=True, help="Output ONNX path.")
    parser.add_argument("--rope-dim", type=int, default=64, help="RoPE dimension.")
    parser.add_argument("--sin-name", default="position_ids_sin", help="Input name for sin.")
    parser.add_argument("--cos-name", default="position_ids_cos", help="Input name for cos.")
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Datatype for sin/cos inputs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dtype = TensorProto.FLOAT16 if args.dtype == "float16" else TensorProto.FLOAT
    patch_rope_inputs(
        args.input,
        args.output,
        rope_dim=args.rope_dim,
        sin_name=args.sin_name,
        cos_name=args.cos_name,
        dtype=dtype,
    )

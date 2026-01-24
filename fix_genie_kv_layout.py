#!/usr/bin/env python3
import argparse
from typing import Dict, Tuple

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


def _shape_from_value_info(value_info) -> Tuple[int, ...]:
    return tuple(int(d.dim_value) for d in value_info.type.tensor_type.shape.dim)


def _set_shape(value_info, shape: Tuple[int, ...]) -> None:
    del value_info.type.tensor_type.shape.dim[:]
    for dim in shape:
        value_info.type.tensor_type.shape.dim.add().dim_value = int(dim)


def _make_const(name: str, values: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(values.astype(np.int64), name=name)


def _replace_node_inputs(nodes, old: str, new: str) -> None:
    for node in nodes:
        for i, inp in enumerate(node.input):
            if inp == old:
                node.input[i] = new


def _rename_node_outputs(nodes, mapping: Dict[str, str]) -> None:
    for node in nodes:
        for i, out in enumerate(node.output):
            if out in mapping:
                node.output[i] = mapping[out]


def fix_kv_layout(
    onnx_in: str, onnx_out: str, num_layers: int = 24, cache_len: int | None = None
) -> None:
    model = onnx.load(onnx_in, load_external_data=False)
    graph = model.graph

    # Infer cache_len from past_key_0_in if not provided.
    if cache_len is None:
        for v in graph.input:
            if v.name == "past_key_0_in":
                shape = _shape_from_value_info(v)
                cache_len = shape[-1]
                break
    if cache_len is None:
        raise ValueError("Unable to infer cache_len from past_key_0_in input.")

    # Update attention_mask to 4D [1, 1, 1, cache_len + 1] and bypass legacy unsqueeze.
    attn_unsqueeze_out = None
    for v in graph.input:
        if v.name == "attention_mask":
            _set_shape(v, (1, 1, 1, cache_len + 1))
            break
    for node in graph.node:
        if node.op_type == "Unsqueeze" and "attention_mask" in node.input:
            attn_unsqueeze_out = node.output[0]
            break
    if attn_unsqueeze_out:
        _replace_node_inputs(graph.node, attn_unsqueeze_out, "attention_mask")

    # Update past_value_*_in input shapes to [B, H, cache_len, head_dim].
    for v in graph.input:
        if v.name.startswith("past_value_") and v.name.endswith("_in"):
            shape = list(_shape_from_value_info(v))
            if len(shape) != 4:
                raise ValueError(f"Unexpected rank for {v.name}: {shape}")
            _set_shape(v, (shape[0], shape[1], cache_len, shape[3]))

    new_nodes = []

    # Keep original nodes (with updated inputs/outputs), skipping legacy attention_mask unsqueeze.
    for node in graph.node:
        if attn_unsqueeze_out and node.op_type == "Unsqueeze" and attn_unsqueeze_out in node.output:
            continue
        new_nodes.append(node)

    graph.ClearField("node")
    graph.node.extend(new_nodes)

    onnx.save(model, onnx_out)
    print(f"Wrote {onnx_out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fix KV layout for GENIE/QNN expectations.")
    parser.add_argument("--input", required=True, help="Input ONNX path.")
    parser.add_argument("--output", required=True, help="Output ONNX path.")
    parser.add_argument("--layers", type=int, default=24, help="Number of layers.")
    parser.add_argument("--cache-len", type=int, default=None, help="Cache length (optional).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fix_kv_layout(args.input, args.output, args.layers, args.cache_len)

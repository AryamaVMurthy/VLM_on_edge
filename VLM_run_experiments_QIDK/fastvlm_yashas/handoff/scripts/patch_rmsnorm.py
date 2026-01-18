#!/usr/bin/env python3
import argparse
import re
from typing import List, Optional, Tuple

import numpy as np
import onnx
from onnx import helper, TensorProto


def _get_attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            if a.type == onnx.AttributeProto.FLOAT:
                return a.f
            if a.type == onnx.AttributeProto.INT:
                return a.i
            if a.type == onnx.AttributeProto.INTS:
                return list(a.ints)
            if a.type == onnx.AttributeProto.FLOATS:
                return list(a.floats)
    return default


def _safe_name(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", s.strip())
    return s or "rmsnorm"


def _make_const(name: str, value: float) -> onnx.NodeProto:
    tensor = helper.make_tensor(
        name=f"{name}_value",
        data_type=TensorProto.FLOAT,
        dims=[1],
        vals=[float(value)],
    )
    return helper.make_node("Constant", [], [name], value=tensor, name=f"{name}_const")


def _rmsnorm_nodes(
    prefix: str,
    x: str,
    gamma: str,
    epsilon: float,
    axis: int,
    output: str,
    bias: Optional[str] = None,
    skip: Optional[str] = None,
    residual_output: Optional[str] = None,
) -> List[onnx.NodeProto]:
    nodes = []

    if skip is not None:
        sum_out = residual_output or f"{prefix}_sum"
        nodes.append(helper.make_node("Add", [x, skip], [sum_out], name=f"{prefix}_add"))
        x_in = sum_out
    else:
        x_in = x
        sum_out = None

    x_sq = f"{prefix}_x_sq"
    nodes.append(helper.make_node("Mul", [x_in, x_in], [x_sq], name=f"{prefix}_mul"))

    mean = f"{prefix}_mean"
    nodes.append(
        helper.make_node(
            "ReduceMean",
            [x_sq],
            [mean],
            name=f"{prefix}_reduce_mean",
            axes=[axis],
            keepdims=1,
        )
    )

    eps_const = f"{prefix}_eps"
    nodes.append(_make_const(eps_const, epsilon))

    var_eps = f"{prefix}_var_eps"
    nodes.append(helper.make_node("Add", [mean, eps_const], [var_eps], name=f"{prefix}_add_eps"))

    std = f"{prefix}_std"
    nodes.append(helper.make_node("Sqrt", [var_eps], [std], name=f"{prefix}_sqrt"))

    inv_std = f"{prefix}_inv_std"
    nodes.append(helper.make_node("Reciprocal", [std], [inv_std], name=f"{prefix}_reciprocal"))

    norm = f"{prefix}_norm"
    nodes.append(helper.make_node("Mul", [x_in, inv_std], [norm], name=f"{prefix}_mul_norm"))

    scaled = f"{prefix}_scaled"
    nodes.append(helper.make_node("Mul", [norm, gamma], [scaled], name=f"{prefix}_mul_scale"))

    if bias:
        nodes.append(helper.make_node("Add", [scaled, bias], [output], name=f"{prefix}_add_bias"))
    else:
        nodes.append(helper.make_node("Identity", [scaled], [output], name=f"{prefix}_out"))

    return nodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model = onnx.load(args.input)
    new_nodes: List[onnx.NodeProto] = []
    replaced = 0

    for idx, node in enumerate(model.graph.node):
        if node.op_type == "SimplifiedLayerNormalization" and node.domain == "":
            prefix = _safe_name(node.name) + f"_{idx}"
            epsilon = _get_attr(node, "epsilon", 1e-5)
            axis = int(_get_attr(node, "axis", -1))
            inputs = list(node.input)
            x = inputs[0]
            gamma = inputs[1]
            bias = inputs[2] if len(inputs) > 2 else None
            out = node.output[0]
            new_nodes.extend(
                _rmsnorm_nodes(
                    prefix=prefix,
                    x=x,
                    gamma=gamma,
                    epsilon=epsilon,
                    axis=axis,
                    output=out,
                    bias=bias,
                )
            )
            replaced += 1
            continue

        if node.op_type == "SkipSimplifiedLayerNormalization":
            prefix = _safe_name(node.name) + f"_{idx}"
            epsilon = _get_attr(node, "epsilon", 1e-5)
            axis = int(_get_attr(node, "axis", -1))
            inputs = list(node.input)
            x = inputs[0]
            skip = inputs[1]
            gamma = inputs[2]
            bias = inputs[3] if len(inputs) > 3 else None
            outputs = list(node.output)
            out = outputs[0]
            residual_out = outputs[3] if len(outputs) > 3 and outputs[3] else None
            new_nodes.extend(
                _rmsnorm_nodes(
                    prefix=prefix,
                    x=x,
                    gamma=gamma,
                    epsilon=epsilon,
                    axis=axis,
                    output=out,
                    bias=bias,
                    skip=skip,
                    residual_output=residual_out,
                )
            )
            replaced += 1
            continue

        new_nodes.append(node)

    model.graph.ClearField("node")
    model.graph.node.extend(new_nodes)

    onnx.save(model, args.output)
    print(f"Replaced {replaced} RMSNorm nodes. Saved to {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
from pathlib import Path

import onnx
from onnx import helper, TensorProto, compose


def build_wrapper(
    src_path: Path,
    out_path: Path,
    height: int,
    width: int,
    output_name: str | None,
    input_dtype: str,
    output_dtype: str,
) -> None:
    src = onnx.load(src_path)
    src_prefixed = compose.add_prefix(src, "src_")

    src_input_dtype = None
    if src.graph.input:
        src_input_dtype = src.graph.input[0].type.tensor_type.elem_type

    flat = height * width
    input_name = "pixel_values"
    reshaped = "pre_pixel_values_reshaped"

    if input_dtype == "uint8":
        inp_dtype = TensorProto.UINT8
    elif input_dtype == "float32":
        inp_dtype = TensorProto.FLOAT
    elif input_dtype == "float16":
        inp_dtype = TensorProto.FLOAT16
    else:
        raise ValueError(f"Unsupported input dtype: {input_dtype}")

    inp = helper.make_tensor_value_info(input_name, inp_dtype, [flat, 3])
    out = helper.make_tensor_value_info(reshaped, TensorProto.FLOAT, [height, width, 3])

    shape = helper.make_tensor("shape", TensorProto.INT64, [3], [height, width, 3])

    nodes = []
    if input_dtype == "uint8":
        scale = helper.make_tensor("scale", TensorProto.FLOAT, [1], [1.0 / 255.0])
        nodes.extend(
            [
                helper.make_node("Cast", [input_name], ["pre_pixel_values_f32"], to=TensorProto.FLOAT),
                helper.make_node("Mul", ["pre_pixel_values_f32", "scale"], ["pre_pixel_values_scaled"]),
                helper.make_node("Reshape", ["pre_pixel_values_scaled", "shape"], [reshaped]),
            ]
        )
        initializers = [scale, shape]
    elif input_dtype == "float16":
        if src_input_dtype == TensorProto.FLOAT16:
            nodes.append(helper.make_node("Reshape", [input_name, "shape"], [reshaped]))
        else:
            nodes.extend(
                [
                    helper.make_node("Cast", [input_name], ["pre_pixel_values_f32"], to=TensorProto.FLOAT),
                    helper.make_node("Reshape", ["pre_pixel_values_f32", "shape"], [reshaped]),
                ]
            )
        initializers = [shape]
    else:
        nodes.append(helper.make_node("Reshape", [input_name, "shape"], [reshaped]))
        initializers = [shape]

    pre = helper.make_graph(nodes, "preprocess", [inp], [out], initializer=initializers)
    pre_model = helper.make_model(pre, opset_imports=src.opset_import)
    pre_model.ir_version = src.ir_version

    src_input_name = f"src_{input_name}"
    merged = compose.merge_models(pre_model, src_prefixed, io_map=[(reshaped, src_input_name)])

    # Preserve source outputs, optionally rename/cast to match Genie expectations.
    del merged.graph.output[:]
    merged.graph.output.extend(src_prefixed.graph.output)

    if len(src_prefixed.graph.output) != 1:
        raise ValueError("Expected a single output to rename/cast")
    orig = src_prefixed.graph.output[0].name
    out_info = src_prefixed.graph.output[0]
    out_shape = [d.dim_value for d in out_info.type.tensor_type.shape.dim]
    out_elem = out_info.type.tensor_type.elem_type

    target_name = output_name or orig
    if output_dtype == "float32":
        target_elem = TensorProto.FLOAT
    elif output_dtype == "float16":
        target_elem = TensorProto.FLOAT16
    elif output_dtype == "auto":
        target_elem = out_elem
    else:
        raise ValueError(f"Unsupported output dtype: {output_dtype}")

    if target_name != orig or target_elem != out_elem:
        if target_elem != out_elem:
            cast_out = f"{target_name}"
            merged.graph.node.append(helper.make_node("Cast", [orig], [cast_out], to=target_elem))
        else:
            merged.graph.node.append(helper.make_node("Identity", [orig], [target_name], name="rename_output"))
        new_out = helper.make_tensor_value_info(target_name, target_elem, out_shape)
        del merged.graph.output[:]
        merged.graph.output.append(new_out)
    onnx.save(merged, out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--input-dtype", default="uint8", choices=["uint8", "float32", "float16"])
    parser.add_argument("--output-dtype", default="auto", choices=["auto", "float16", "float32"])
    args = parser.parse_args()

    build_wrapper(
        Path(args.src),
        Path(args.out),
        args.height,
        args.width,
        args.output_name,
        args.input_dtype,
        args.output_dtype,
    )


if __name__ == "__main__":
    main()

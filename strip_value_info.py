#!/usr/bin/env python3
import argparse

import onnx


def strip_value_info(in_path: str, out_path: str) -> None:
    model = onnx.load(in_path, load_external_data=False)
    io_names = {v.name for v in model.graph.input} | {v.name for v in model.graph.output}

    kept = [vi for vi in model.graph.value_info if vi.name not in io_names]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept)

    onnx.save(model, out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove value_info entries that duplicate IO names.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    strip_value_info(args.input, args.output)

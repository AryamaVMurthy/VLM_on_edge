#!/usr/bin/env python3
import argparse

import onnx


def strip_all_value_info(in_path: str, out_path: str) -> None:
    model = onnx.load(in_path, load_external_data=False)
    # Remove all intermediate shape annotations to avoid stale/inaccurate dims.
    del model.graph.value_info[:]
    onnx.save(model, out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove all value_info entries from an ONNX graph.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    strip_all_value_info(args.input, args.output)

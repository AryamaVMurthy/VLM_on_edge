#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import qai_hub as hub
from qai_hub.util.dataset_entries_converters import h5_to_dataset_entries

DEFAULT_DEVICE = "Snapdragon 8 Elite QRD"


def _dtype_from_str(dtype_str: str) -> np.dtype:
    if "float16" in dtype_str:
        return np.float16
    if "float32" in dtype_str:
        return np.float32
    if "int64" in dtype_str:
        return np.int64
    if "int32" in dtype_str:
        return np.int32
    raise ValueError(f"Unsupported dtype string: {dtype_str}")


def _load_metadata(path: Path) -> dict:
    return json.loads(path.read_text())


def _auto_pick_metadata(base_dir: Path) -> Path:
    candidates = sorted(base_dir.glob("**/metadata.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No metadata.json found under {base_dir}")
    return candidates[0]


def _build_inputs(input_specs: dict, input_order: list[str], seed: int, truncate_64bit: bool) -> dict:
    rng = np.random.default_rng(seed)
    inputs = {}
    for name in input_order:
        if name not in input_specs:
            raise ValueError(f"Input {name} missing from input_specs")
        shape, dtype_str = input_specs[name]
        dtype = _dtype_from_str(dtype_str)
        if truncate_64bit and dtype == np.int64:
            dtype = np.int32
        if "attention_mask" in name:
            data = np.ones(shape, dtype=dtype)
        elif "position_ids" in name:
            data = np.zeros(shape, dtype=dtype)
        elif "inputs_embeds" in name:
            data = rng.standard_normal(size=shape).astype(dtype)
        elif "past_key" in name or "past_value" in name:
            data = np.zeros(shape, dtype=dtype)
        else:
            data = np.zeros(shape, dtype=dtype)
        inputs[name] = [np.ascontiguousarray(data)]
    return inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QAI Hub inference on a compiled model.")
    parser.add_argument("--metadata", help="Path to metadata.json with compile_job_id and input_specs.")
    parser.add_argument("--compile-job-id", help="Compile job ID to use (overrides metadata).")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="QAI Hub device name.")
    parser.add_argument("--output-dir", default="qaihub_cloud_runs", help="Output directory for artifacts.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random inputs_embeds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = Path(args.metadata) if args.metadata else _auto_pick_metadata(Path("qaihub_bins"))
    meta = _load_metadata(metadata_path)

    compile_job_id = args.compile_job_id or meta.get("compile_job_id")
    if not compile_job_id:
        raise ValueError("compile_job_id not provided and not found in metadata.")

    print(f"Using compile job: {compile_job_id}")
    compile_job = hub.get_job(compile_job_id)
    target_model = compile_job.get_target_model()

    input_specs = compile_job.target_shapes
    if not input_specs:
        raise ValueError("compile job is missing target_shapes")

    truncate_64bit = "--truncate_64bit_io" in str(compile_job.options)
    input_order = list(input_specs.keys())
    inputs = _build_inputs(input_specs, input_order, args.seed, truncate_64bit)
    device = hub.Device(args.device)

    print("Submitting inference job...")
    inf_job = hub.submit_inference_job(
        target_model,
        device=device,
        inputs=inputs,
        name=f"fastvlm_cloud_infer_{compile_job_id}",
    )
    print(f"  ✓ Inference job ID: {inf_job.job_id}")
    print(f"  URL: {inf_job.url}")

    status = inf_job.wait()
    if not status.success:
        raise RuntimeError(f"Inference failed: {status.message}")

    out_dir = base_dir / inf_job.job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "output_data.h5"
    inf_job.download_output_data(str(out_file))

    with h5py.File(out_file, "r") as h5f:
        outputs = h5_to_dataset_entries(h5f)

    print("Outputs:")
    for name, batches in outputs.items():
        batch = batches[0] if batches else None
        shape = getattr(batch, "shape", None)
        dtype = getattr(batch, "dtype", None)
        print(f"  - {name}: {shape} {dtype}")

    report = {
        "compile_job_id": compile_job_id,
        "inference_job_id": inf_job.job_id,
        "device": args.device,
        "metadata": str(metadata_path),
        "output_file": str(out_file),
    }
    (out_dir / "run_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

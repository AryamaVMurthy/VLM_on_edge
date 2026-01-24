#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
import shutil

import onnx
import qai_hub as hub

DEVICE_NAME = "Snapdragon 8 Elite QRD"
COMPILE_OPTIONS = "--target_runtime qnn_context_binary --truncate_64bit_io"
JOB_PREFIX = "fastvlm_full"

DTYPE_MAP = {
    10: "float16",  # TensorProto.FLOAT16
    1: "float32",   # TensorProto.FLOAT
    7: "int64",     # TensorProto.INT64
    6: "int32",     # TensorProto.INT32
    2: "uint8",     # TensorProto.UINT8
}


def onnx_input_specs(onnx_path: Path):
    model = onnx.load(str(onnx_path), load_external_data=False)
    specs = {}
    for v in model.graph.input:
        tt = v.type.tensor_type
        shape = tuple(int(d.dim_value) for d in tt.shape.dim)
        dtype = DTYPE_MAP.get(int(tt.elem_type))
        if dtype is None:
            raise ValueError(f"Unsupported ONNX dtype for {v.name}: {tt.elem_type}")
        specs[v.name] = (shape, dtype)
    return specs


def onnx_output_names(onnx_path: Path):
    model = onnx.load(str(onnx_path), load_external_data=False)
    return [v.name for v in model.graph.output]


def write_metadata(out_dir: Path, payload: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metadata.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile ONNX to QNN context binary via QAI Hub.")
    parser.add_argument("--onnx", default="fastvlm_full_fp16_embedded.onnx", help="ONNX model path.")
    parser.add_argument("--output-dir", default="qaihub_bins", help="Output base directory.")
    parser.add_argument("--job-prefix", default=JOB_PREFIX, help="Prefix for output directory naming.")
    parser.add_argument("--options", default=COMPILE_OPTIONS, help="QAI Hub compile options string.")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.onnx)
    output_dir = Path(args.output_dir)
    job_prefix = args.job_prefix
    options = args.options

    if not model_path.exists():
        print(f"Error: ONNX not found at {model_path}")
        sys.exit(1)

    print(f"Reading ONNX signature from: {model_path}")
    specs = onnx_input_specs(model_path)
    output_names = onnx_output_names(model_path)

    if "--output_names" not in options:
        options = f"{options} --output_names {','.join(output_names)}"

    upload_path = model_path
    external_data = Path(str(model_path) + ".data")
    if external_data.exists():
        stage_dir = output_dir / f"{job_prefix}_upload"
        stage_dir.mkdir(parents=True, exist_ok=True)
        for p in stage_dir.iterdir():
            if p.is_file():
                p.unlink()
        shutil.copy2(model_path, stage_dir / model_path.name)
        shutil.copy2(external_data, stage_dir / external_data.name)
        upload_path = stage_dir

    print(f"Uploading newest model: {upload_path}")
    model = hub.upload_model(str(upload_path))
    print(f"  ✓ Model ID: {model.model_id}")

    print(f"\nSubmitting compile job...")
    job = hub.submit_compile_job(
        model,
        device=hub.Device(DEVICE_NAME),
        name=f"{job_prefix}_compile",
        options=options,
        input_specs=specs,
    )
    print(f"  ✓ Job ID: {job.job_id}")
    print(f"  URL: {job.url}")
    
    status = job.wait()
    if status.success:
        print(f"✓ Success!")
        out_dir = output_dir / f"{job_prefix}_{job.job_id}"
        job.download_target_model(str(out_dir))
        print(f"✓ Saved to {out_dir}")

        # Normalize name for GENIE config usage.
        bins = sorted(out_dir.glob("*.bin"))
        if not bins:
            print("Warning: no .bin found in output directory.")
        else:
            canonical = out_dir / "fastvlm_full.bin"
            bins[0].replace(canonical)
            print(f"✓ Renamed {bins[0].name} -> {canonical.name}")

        meta = {
            "model_id": model.model_id,
            "job_id": job.job_id,
            "device": DEVICE_NAME,
            "options": options,
            "onnx": str(model_path),
            "input_specs": specs,
            "output_names": output_names,
        }
        write_metadata(out_dir, meta)
    else:
        print(f"✗ Failed: {status.message}")

if __name__ == "__main__":
    main()

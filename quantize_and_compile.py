#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
import zipfile
import shutil

import numpy as np
import onnx
import qai_hub as hub

DEVICE_NAME = "Snapdragon 8 Elite QRD"
DEFAULT_COMPILE_OPTIONS = "--target_runtime qnn_context_binary --truncate_64bit_io"
DEFAULT_JOB_PREFIX = "fastvlm_full_w8a16"

DTYPE_MAP = {
    10: np.float16,  # TensorProto.FLOAT16
    1: np.float32,   # TensorProto.FLOAT
    7: np.int64,     # TensorProto.INT64
    6: np.int32,     # TensorProto.INT32
    2: np.uint8,     # TensorProto.UINT8
}

QAI_DTYPE_MAP = {
    "int4": hub.QuantizeDtype.INT4,
    "int8": hub.QuantizeDtype.INT8,
    "int16": hub.QuantizeDtype.INT16,
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

def _parse_dtype(name: str) -> np.dtype:
    name = name.lower()
    if name in ("float16", "fp16"):
        return np.float16
    if name in ("float32", "fp32"):
        return np.float32
    raise ValueError(f"Unsupported dtype: {name}")


def load_calib_embeddings(path: Path, emb_dim: int, dtype: np.dtype, num_samples: int):
    data = np.fromfile(path, dtype=dtype)
    if data.size % emb_dim != 0:
        raise ValueError(f"Embedding raw size {data.size} not divisible by emb_dim {emb_dim}")
    tokens = data.reshape(-1, emb_dim)
    if tokens.size == 0:
        raise ValueError(f"No embeddings found in {path}")
    sample_count = min(num_samples, tokens.shape[0])
    if sample_count <= 0:
        raise ValueError("num_samples must be > 0")
    if tokens.shape[0] == sample_count:
        idx = np.arange(tokens.shape[0])
    else:
        idx = np.linspace(0, tokens.shape[0] - 1, num=sample_count, dtype=int)
    samples = []
    for i in idx:
        samples.append(tokens[i : i + 1].reshape(1, 1, emb_dim))
    return samples


def _infer_context_len(specs) -> int | None:
    for name, (shape, _) in specs.items():
        if "attention_mask" in name and shape:
            return int(shape[-1])
    for name, (shape, _) in specs.items():
        if "past_key" in name and shape:
            return int(shape[-1]) + 1
    return None


def _calib_kv_scale(calib_embeds, fallback: float = 0.1) -> float:
    if not calib_embeds:
        return fallback
    flat = np.concatenate([c.reshape(-1).astype(np.float32, copy=False) for c in calib_embeds])
    std = float(np.std(flat)) if flat.size else 0.0
    return std if std > 0 else fallback


def build_rope_sin_cos(position_id: int, rope_dim: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    inv_freq = 1.0 / (theta ** (np.arange(0, rope_dim, 2, dtype=np.float32) / rope_dim))
    freqs = np.array([position_id], dtype=np.float32)[:, None] * inv_freq[None, :]
    emb = np.concatenate([freqs, freqs], axis=-1)
    sin = np.sin(emb).astype(np.float32, copy=False)
    cos = np.cos(emb).astype(np.float32, copy=False)
    return sin.reshape(1, 1, rope_dim, 1), cos.reshape(1, 1, rope_dim, 1)


def build_calibration_data(
    specs,
    num_samples: int,
    seed: int,
    calib_embeds=None,
    kv_scale: float | None = None,
    rope_dim: int = 64,
    rope_theta: float = 10000.0,
):
    rng = np.random.default_rng(seed)
    calib = {}
    effective_samples = num_samples
    if calib_embeds:
        effective_samples = min(num_samples, len(calib_embeds))
        if effective_samples <= 0:
            raise ValueError("calib_embeds provided but no samples available")

    ctx_len = _infer_context_len(specs)
    if kv_scale is None:
        kv_scale = _calib_kv_scale(calib_embeds)

    for name, (shape, dtype) in specs.items():
        samples = []
        if name == "inputs_embeds" and calib_embeds:
            for i in range(effective_samples):
                data = calib_embeds[i].astype(dtype, copy=False)
                samples.append(np.ascontiguousarray(data))
        elif "position_ids_sin" in name or "position_ids_cos" in name:
            for i in range(effective_samples):
                pos = i
                if ctx_len is not None:
                    pos = min(pos, ctx_len - 1)
                sin, cos = build_rope_sin_cos(pos, rope_dim, rope_theta)
                data = sin if "sin" in name else cos
                data = data.astype(dtype, copy=False)
                if data.shape != tuple(shape):
                    data = data.reshape(shape)
                samples.append(np.ascontiguousarray(data))
        elif "position_ids" in name and calib_embeds:
            for i in range(effective_samples):
                pos = i
                if ctx_len is not None:
                    pos = min(pos, ctx_len - 1)
                data = np.full(shape, pos, dtype=dtype)
                samples.append(np.ascontiguousarray(data))
        elif "attention_mask" in name and calib_embeds:
            for i in range(effective_samples):
                data = np.zeros(shape, dtype=dtype)
                pos = i
                if ctx_len is not None:
                    pos = min(pos, ctx_len - 1)
                if data.size:
                    data[..., : pos + 1] = 1
                samples.append(np.ascontiguousarray(data))
        else:
            for _ in range(effective_samples):
                if dtype == np.uint8:
                    data = rng.integers(0, 256, size=shape, dtype=dtype)
                elif "attention_mask" in name:
                    data = np.ones(shape, dtype=dtype)
                elif "position_ids_sin" in name or "position_ids_cos" in name:
                    sin, cos = build_rope_sin_cos(0, rope_dim, rope_theta)
                    data = sin if "sin" in name else cos
                    data = data.astype(dtype, copy=False)
                elif "position_ids" in name:
                    data = np.zeros(shape, dtype=dtype)
                elif "inputs_embeds" in name:
                    data = rng.standard_normal(size=shape).astype(dtype)
                elif "past_key" in name or "past_value" in name:
                    data = (rng.standard_normal(size=shape) * kv_scale).astype(dtype)
                else:
                    data = np.zeros(shape, dtype=dtype)
                samples.append(np.ascontiguousarray(data))
        calib[name] = samples
    return calib


def write_metadata(out_dir: Path, payload: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metadata.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def resolve_quantized_model(path: Path, extract_root: Path) -> Path:
    if path.is_dir():
        candidates = sorted(path.rglob("*.onnx"))
        if not candidates:
            print(f"✗ No .onnx found in directory: {path}")
            sys.exit(1)
        return next((p for p in candidates if p.name == "model.onnx"), candidates[0])
    if path.suffix == ".zip":
        extract_dir = extract_root / "quantized"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(extract_dir)
        candidates = sorted(extract_dir.rglob("*.onnx"))
        if not candidates:
            print(f"✗ No .onnx found after extracting {path}")
            sys.exit(1)
        return next((p for p in candidates if p.name == "model.onnx"), candidates[0])
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize ONNX via QAI Hub and compile to QNN context binary."
    )
    parser.add_argument("--onnx", default="fastvlm_full_fp16_embedded_renamed_geniefix.onnx")
    parser.add_argument(
        "--quantized-onnx",
        help="Skip quantization and compile an existing quantized ONNX (file, zip, or directory).",
    )
    parser.add_argument("--output-dir", default="qaihub_bins")
    parser.add_argument("--job-prefix", default=DEFAULT_JOB_PREFIX)
    parser.add_argument("--compile-options", default=DEFAULT_COMPILE_OPTIONS)
    parser.add_argument("--weights-dtype", default="int8", choices=sorted(QAI_DTYPE_MAP.keys()))
    parser.add_argument("--activations-dtype", default="int16", choices=sorted(QAI_DTYPE_MAP.keys()))
    parser.add_argument("--calib-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calib-embed-raw", help="Path to raw embeddings to use for inputs_embeds calibration.")
    parser.add_argument("--calib-embed-dtype", default="float16", help="Datatype for calib-embed-raw (float16/float32).")
    parser.add_argument("--calib-kv-scale", type=float, help="Scale for random KV calibration values.")
    parser.add_argument("--rope-dim", type=int, default=64, help="RoPE dimension for sin/cos inputs.")
    parser.add_argument("--rope-theta", type=float, default=10000.0, help="RoPE theta for sin/cos inputs.")
    return parser.parse_args()


def main():
    args = parse_args()
    onnx_path = Path(args.onnx)
    output_dir = Path(args.output_dir)

    quant_job_id = None
    if args.quantized_onnx:
        quant_onnx_path = Path(args.quantized_onnx)
        if not quant_onnx_path.exists():
            print(f"Error: quantized ONNX not found at {quant_onnx_path}")
            sys.exit(1)
        quant_onnx_path = resolve_quantized_model(quant_onnx_path, quant_onnx_path.parent)
        specs_source = onnx_path if onnx_path.exists() else quant_onnx_path
        specs = onnx_input_specs(specs_source)
    else:
        if not onnx_path.exists():
            print(f"Error: ONNX not found at {onnx_path}")
            sys.exit(1)
        specs = onnx_input_specs(onnx_path)
        calib_embeds = None
        if args.calib_embed_raw:
            calib_embeds = load_calib_embeddings(
                Path(args.calib_embed_raw),
                emb_dim=896,
                dtype=_parse_dtype(args.calib_embed_dtype),
                num_samples=args.calib_samples,
            )
            print(f"Using {len(calib_embeds)} calibration embeddings from {args.calib_embed_raw}")
        calib = build_calibration_data(
            specs,
            args.calib_samples,
            args.seed,
            calib_embeds=calib_embeds,
            kv_scale=args.calib_kv_scale,
            rope_dim=args.rope_dim,
            rope_theta=args.rope_theta,
        )

        upload_path = onnx_path
        external_data = Path(str(onnx_path) + ".data")
        if external_data.exists():
            stage_dir = output_dir / f"{args.job_prefix}_upload"
            stage_dir.mkdir(parents=True, exist_ok=True)
            for p in stage_dir.iterdir():
                if p.is_file():
                    p.unlink()
            shutil.copy2(onnx_path, stage_dir / onnx_path.name)
            shutil.copy2(external_data, stage_dir / external_data.name)
            upload_path = stage_dir

        print("Submitting quantize job...")
        quant_job = hub.submit_quantize_job(
            str(upload_path),
            calib,
            weights_dtype=QAI_DTYPE_MAP[args.weights_dtype],
            activations_dtype=QAI_DTYPE_MAP[args.activations_dtype],
            name=f"{args.job_prefix}_quantize",
        )
        print(f"  ✓ Quantize job ID: {quant_job.job_id}")
        print(f"  URL: {quant_job.url}")

        status = quant_job.wait()
        if not status.success:
            print(f"✗ Quantize failed: {status.message}")
            sys.exit(1)

        quant_job_id = quant_job.job_id
        quant_dir = output_dir / f"{args.job_prefix}_{quant_job.job_id}"
        quant_dir.mkdir(parents=True, exist_ok=True)
        quant_onnx_path = quant_dir / "quantized.onnx"
        downloaded = quant_job.download_target_model(str(quant_onnx_path))
        if isinstance(downloaded, str):
            quant_onnx_path = Path(downloaded)
        quant_onnx_path = resolve_quantized_model(quant_onnx_path, quant_dir)
    print(f"✓ Quantized ONNX saved to {quant_onnx_path}")

    compile_options = args.compile_options
    if "--output_names" not in compile_options:
        output_names = onnx_output_names(quant_onnx_path)
        compile_options = f"{compile_options} --output_names {','.join(output_names)}"

    compile_model = quant_onnx_path
    if list(quant_onnx_path.parent.glob("*.data")):
        compile_model = quant_onnx_path.parent
        print(f"Using ONNX model directory format: {compile_model}")

    print("Submitting compile job...")
    compile_job = hub.submit_compile_job(
        str(compile_model),
        device=hub.Device(DEVICE_NAME),
        name=f"{args.job_prefix}_compile",
        options=compile_options,
        input_specs={name: (shape, _numpy_to_qai_dtype(dtype)) for name, (shape, dtype) in specs.items()},
    )
    print(f"  ✓ Compile job ID: {compile_job.job_id}")
    print(f"  URL: {compile_job.url}")

    status = compile_job.wait()
    if not status.success:
        print(f"✗ Compile failed: {status.message}")
        sys.exit(1)

    out_dir = output_dir / f"{args.job_prefix}_{compile_job.job_id}"
    downloaded = compile_job.download_target_model(str(out_dir))
    if isinstance(downloaded, str):
        downloaded_path = Path(downloaded)
        if downloaded_path.is_file():
            out_dir.mkdir(parents=True, exist_ok=True)
            target = out_dir / downloaded_path.name
            if downloaded_path.resolve() != target.resolve():
                downloaded_path.replace(target)
    print(f"✓ Compiled artifacts saved to {out_dir}")

    bins = sorted(out_dir.glob("*.bin"))
    if bins:
        canonical = out_dir / "fastvlm_full.bin"
        bins[0].replace(canonical)
        print(f"✓ Renamed {bins[0].name} -> {canonical.name}")
    else:
        print("Warning: no .bin found in output directory.")

    meta = {
        "onnx": str(onnx_path),
        "quant_job_id": quant_job_id,
        "compile_job_id": compile_job.job_id,
        "device": DEVICE_NAME,
        "compile_options": compile_options,
        "weights_dtype": args.weights_dtype,
        "activations_dtype": args.activations_dtype,
        "calib_samples": args.calib_samples,
        "calib_embed_raw": args.calib_embed_raw,
        "calib_embed_dtype": args.calib_embed_dtype,
        "input_specs": {k: (list(v[0]), str(v[1])) for k, v in specs.items()},
        "output_names": onnx_output_names(quant_onnx_path),
    }
    write_metadata(out_dir, meta)


def _numpy_to_qai_dtype(dtype: np.dtype) -> str:
    if dtype == np.float16:
        return "float16"
    if dtype == np.float32:
        return "float32"
    if dtype == np.int64:
        return "int64"
    if dtype == np.int32:
        return "int32"
    if dtype == np.uint8:
        return "uint8"
    raise ValueError(f"Unsupported numpy dtype: {dtype}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import qai_hub as hub

ROOT_DIR = Path(__file__).resolve().parents[2]
HANDOFF_DIR = ROOT_DIR / "handoff"

MODEL_IDS_FILE = HANDOFF_DIR / "qaihub-model-ids.txt"
OUT_DIR = ROOT_DIR / "models" / "fastvlm-qaihub-fp16-context"
JOBS_FILE = HANDOFF_DIR / "qaihub-fp16-jobs.txt"

DEVICE_NAME = "Snapdragon 8 Elite QRD"
DEVICE_OS = "15"

# Compile to QNN context binary with float16 and truncate int64 I/O to int32.
COMPILE_OPTIONS = "--target_runtime qnn_context_binary --quantize_full_type float16 --truncate_64bit_io"


def load_model_ids(path: Path) -> list[tuple[str, str]]:
    models: list[tuple[str, str]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"Unexpected line in {path}: {line}")
        name, model_id = line.split(":", 1)
        models.append((name.strip(), model_id.strip()))
    return models


def main() -> None:
    if not MODEL_IDS_FILE.exists():
        raise SystemExit(f"Missing {MODEL_IDS_FILE}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = hub.Device(DEVICE_NAME, DEVICE_OS)
    models = load_model_ids(MODEL_IDS_FILE)

    job_lines = [
        f"device={DEVICE_NAME} os={DEVICE_OS}",
        f"compile_options={COMPILE_OPTIONS}",
        "",
    ]

    for name, model_id in models:
        print(f"Submitting compile job for {name} ({model_id})...")
        model = hub.get_model(model_id)
        job = hub.submit_compile_job(
            model,
            device=device,
            name=f"{name}_qnn_context_fp16",
            options=COMPILE_OPTIONS,
        )
        status = job.wait()
        job_dir = OUT_DIR / name
        job_dir.mkdir(parents=True, exist_ok=True)
        if status.success:
            job.download_results(str(job_dir))
            result = "SUCCESS"
        else:
            result = f"FAILED ({status.message})"
        job_lines.append(
            f"{name}: model_id={model_id} job_id={job.job_id} status={result} url={job.url}"
        )

    JOBS_FILE.write_text("\n".join(job_lines) + "\n")
    print(f"Wrote {JOBS_FILE}")


if __name__ == "__main__":
    main()

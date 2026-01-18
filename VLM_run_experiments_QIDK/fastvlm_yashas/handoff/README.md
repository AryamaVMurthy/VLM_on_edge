# FastVLM handoff (QAIRT 2.42.0.251225)

This folder contains the exact scripts and commands used to reproduce the current FastVLM artifacts and the end-to-end run (int8 vision encoder + fp16 embedder on device, Genie CPU backend decode). The intent is that someone can start from the repo root and reproduce all outputs that exist today.

All commands below are meant to be run from the repo root: `qairt/`.

## What this handoff includes

- Export scripts for fp32 ONNX and fixed-kv decoder.
- Calibration input generator for int8 quantization.
- QAI Hub compile script to produce fp16 QNN context binaries (vision + embedder + decoder).
- Local int8 vision encoder conversion + context binary build (HTP).
- Genie CPU backend decoder build from HF weights (fp32 + int8 Z8).
- End-to-end scripts that run vision + embedder on device and decode with Genie, with decoder/vision precision options.

## Prerequisites

- QAIRT/QNN SDK at `2.42.0.251225/` in the repo root.
- A working Python environment with: `torch`, `transformers`, `numpy`, `Pillow`, `onnx`, `onnxruntime`.
- `adb` installed and exactly one Android device connected.
- `ANDROID_NDK_ROOT` set for device-side decode (used to push `libc++_shared.so`).
- QAI Hub access with `qai-hub` configured (token in `~/.qai_hub/` or via env).

## Required repo layout (paths must match)

The scripts in this handoff folder assume the repo root is `qairt/` and that QAIRT is unpacked inside it:

```
qairt/
  2.42.0.251225/
  handoff/
  models/
```

If you move the handoff folder elsewhere, the relative paths will break. Keep the same layout or update the scripts.

## Always activate the environment

Activate `qnn_env` **before every command** (all scripts do this internally, but do it in your shell too):

```bash
source 2.42.0.251225/qnn_env/bin/activate
```

When using QNN tools (converter, model-lib-generator, context-binary-generator, Genie composer), also source envsetup:

```bash
set +u
source 2.42.0.251225/bin/envsetup.sh
set -u
```

## Step 1: Export fp32 ONNX

Script: `handoff/scripts/01_export_fp32_onnx.sh`

```bash
MODEL_DIR=/home/yashas/qairt/models/apple--FastVLM-0.5B \
  bash handoff/scripts/01_export_fp32_onnx.sh
```

Outputs:
- `models/fastvlm-onnx-fp32/vision_encoder_fp32.onnx`
- `models/fastvlm-onnx-fp32/embed_tokens_fp32.onnx`
- `models/fastvlm-onnx-fp32/decoder_fixed_kv_fp32.onnx`

## Step 2: Create calibration inputs (for int8 vision)

Script: `handoff/scripts/02_make_calib_inputs.sh`

Random calibration (matches current state):

```bash
bash handoff/scripts/02_make_calib_inputs.sh
```

Or use images:

```bash
bash handoff/scripts/02_make_calib_inputs.sh /path/to/images
```

Outputs:
- `models/fastvlm-calib-raw/input_list.txt`
- `models/fastvlm-calib-raw/sample_*.raw`

## Step 3: Upload ONNX to QAI Hub (if needed)

We compiled fp16 bins from QAI Hub model IDs recorded in `handoff/qaihub-model-ids.txt`. If you need to re-upload ONNX, use the CLI and update that file with the new model IDs:

```bash
qai-hub upload-model --model models/fastvlm-onnx-fp32/vision_encoder_fp32.onnx --name fastvlm_vision_encoder_fp32
qai-hub upload-model --model models/fastvlm-onnx-fp32/embed_tokens_fp32.onnx --name fastvlm_embed_tokens_fp32
qai-hub upload-model --model models/fastvlm-onnx-fp32/decoder_fixed_kv_fp32.onnx --name fastvlm_decoder_fixed_kv_fp32
```

Current model IDs (already recorded):
- `handoff/qaihub-model-ids.txt`

## Step 4: Compile fp16 QNN context binaries on QAI Hub

Script: `handoff/scripts/03_qaihub_compile_fp16.sh`

```bash
bash handoff/scripts/03_qaihub_compile_fp16.sh
```

Outputs (downloaded by the script):
- `models/fastvlm-qaihub-fp16-context/fastvlm_vision_encoder_fp32/job_<job>_optimized_bin_<target>.bin`
- `models/fastvlm-qaihub-fp16-context/fastvlm_embed_tokens_fp32/job_<job>_optimized_bin_<target>.bin`
- `models/fastvlm-qaihub-fp16-context/fastvlm_decoder_fixed_kv_fp32/job_<job>_optimized_bin_<target>.bin`

Job history is written to:
- `handoff/qaihub-fp16-jobs.txt`

## Step 5: Build int8 vision encoder context binary (HTP)

Script: `handoff/scripts/04_build_int8_vision_encoder.sh`

```bash
bash handoff/scripts/04_build_int8_vision_encoder.sh
```

Outputs:
- `models/fastvlm-qnn-int8/vision_encoder_int8_qnnmodel.cpp`
- `models/fastvlm-qnn-int8/vision_encoder_int8_qnnmodel.bin`
- `models/fastvlm-genie/local_context/vision_int8_qnnmodel/aarch64-android/libvision_encoder_int8_qnnmodel.so`
- `models/fastvlm-genie/local_context/vision_int8_qnnmodel/x86_64-linux-clang/libvision_encoder_int8_qnnmodel.so`
- `models/fastvlm-genie/local_context/vision_int8_qnnmodel/output/fastvlm_vision_encoder_int8_htp.bin`
- `models/fastvlm-genie/local_context/vision_int8_qnnmodel/vision_int8_ctx_info.json`

The exact converter command used is recorded in:
- `models/fastvlm-qnn-int8/vision_encoder_int8_qnnmodel.cpp`

## Step 6: Build Genie CPU backend bin (decoder)

Script: `handoff/scripts/05_build_genai_decoder_bin.sh`

To build fp32:

```bash
MODEL_DIR=/home/yashas/qairt/models/apple--FastVLM-0.5B \
  bash handoff/scripts/05_build_genai_decoder_bin.sh
```

To build int8 (Z8 weight-only):

```bash
MODEL_DIR=/home/yashas/qairt/models/apple--FastVLM-0.5B \
QUANTIZE=Z8 \
  bash handoff/scripts/05_build_genai_decoder_bin.sh
```

Outputs:
- `models/fastvlm-genie/fastvlm_model.bin` (fp32)
- `models/fastvlm-genie/fastvlm_model_z8.bin` (int8 Z8)
- `models/fastvlm-genie/LUT.bin`
- `models/fastvlm-genie/tokenizer.json`

Optional (int4 Q4):

```bash
MODEL_DIR=/home/yashas/qairt/models/apple--FastVLM-0.5B \
QUANTIZE=Q4 \
  bash handoff/scripts/05_build_genai_decoder_bin.sh
```

## Step 7: Run end-to-end (device + Genie)

### Decode on device (recommended)

Script: `handoff/scripts/run_fastvlm_device.sh`

```bash
./handoff/scripts/run_fastvlm_device.sh \
  /home/yashas/qairt/models/fastvlm_test.jpg \
  "Describe the image in detail." \
  50 \
  1024 \
  int8 \
  int8
```

Arguments:
1) image_path
2) prompt
3) max_new_tokens (0 = auto-cap)
4) image_size
5) decoder_precision: `fp32` or `int8` (optional `int4`/`q4` if built)
6) vision_precision: `int8` or `fp16`

Notes:
- Set `SKIP_PUSH=1` to skip re-pushing binaries/libs (inputs are still pushed).
- Profiling is always enabled; Genie profile stats are printed.

Advanced: direct script (same as wrapper target)

```bash
IMAGE_PATH=/home/yashas/qairt/models/fastvlm_test.jpg \
PROMPT="Describe the image in detail." \
MODEL_DIR=/home/yashas/qairt/models/apple--FastVLM-0.5B \
MAX_NEW_TOKENS=50 \
  bash handoff/scripts/run_e2e_int8_device.sh
```

This does:
- Runs int8 vision encoder on device (HTP) using `qnn-net-run`.
- Runs fp16 embedder on device (HTP) using `qnn-net-run`.
- Dequantizes vision output + concatenates with text embeds on host.
- Runs Genie CPU backend decode on device using `genie-t2t-run`.

Outputs:
- `models/fastvlm-genie/run_outputs/<timestamp>/combined_embeddings_fp32.raw`
- `models/fastvlm-genie/run_outputs/<timestamp>/fastvlm_genie_device_config.json`
- `models/fastvlm-genie/run_outputs/<timestamp>/vision_out/`
- `models/fastvlm-genie/run_outputs/<timestamp>/embed_out/`

### Decode on host (CPU)

Script: `handoff/scripts/run_e2e_int8_cpu.sh`

```bash
IMAGE_PATH=/home/yashas/qairt/models/fastvlm_test.jpg \
PROMPT="Describe the image in detail." \
MODEL_DIR=/home/yashas/qairt/models/apple--FastVLM-0.5B \
MAX_NEW_TOKENS=50 \
  bash handoff/scripts/run_e2e_int8_cpu.sh
```

## Notes and important details

- The vision encoder int8 output is `uint8` and must be dequantized using the scale/offset from `models/fastvlm-genie/local_context/vision_int8_qnnmodel/vision_int8_ctx_info.json`.
- The embedder output is float32; concatenation happens on host in the run scripts.
- Genie decode uses `--embedding_query_output_type text` because `token` mode segfaulted for this build.
- The QAI Hub fp16 decoder context binary fails on device with an HTP DMA error, so the working decode path is Genie CPU backend via `fastvlm_model.bin`.
- The int8 decoder bin uses weight-only Z8 quantization and the same LUT/tokenizer as fp32.
- Device runs assume HTP v79 (SM8750 class). Always confirm with `adb devices` before running.

## Scripts in this folder

- `handoff/scripts/01_export_fp32_onnx.sh`
- `handoff/scripts/02_make_calib_inputs.sh`
- `handoff/scripts/03_qaihub_compile_fp16.sh`
- `handoff/scripts/04_build_int8_vision_encoder.sh`
- `handoff/scripts/05_build_genai_decoder_bin.sh`
- `handoff/scripts/run_e2e_int8_cpu.sh`
- `handoff/scripts/run_e2e_int8_device.sh`
- `handoff/scripts/run_fastvlm_device.sh`
- `handoff/scripts/export_fastvlm_fp16.py`
- `handoff/scripts/export_fastvlm_decoder_fixed_kv.py`
- `handoff/scripts/make_calib_inputs.py`
- `handoff/scripts/patch_rmsnorm.py`
- `handoff/scripts/patch_fused_ops.py`
- `handoff/scripts/qaihub_compile_fp16.py`

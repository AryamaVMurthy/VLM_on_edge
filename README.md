# VLM_on_edge
**Project for Paper on Multimodal AI on Edge Devices**

![Status](https://img.shields.io/badge/Status-Active-success)
![Focus](https://img.shields.io/badge/Focus-Vision%20Language%20Models-blue)
![Platform](https://img.shields.io/badge/Platform-Edge%20Devices-orange)

## 📖 Overview
This project aims to explore the application of Vision-Language Models (VLMs) on edge devices, focusing on the challenges and opportunities presented by limited computational resources and network bandwidth. The repository serves as a central hub for experimental code, results, and reference literature.

## Decoder Export (FP16)
This worktree currently focuses on exporting a **decoder-only** ONNX with fixed KV-cache I/O for QAI Hub compilation.

```bash
# Export ONNX
python export_decoder_fp16.py \
  --model-dir /path/to/checkpoint \
  --output-dir .

# Validate signature (fails fast on shape/dtype mismatches)
python validate_onnx_signature.py \
  --onnx fastvlm_full_fp16.onnx \
  --cache-len 1023
```

## QAI Hub Compile (Stage 2)
Compile the exported ONNX to a QNN context binary using QAI Hub.

```bash
python rename_onnx_tensors.py \
  --input fastvlm_full_fp16_embedded.onnx \
  --output fastvlm_full_fp16_embedded_renamed.onnx

python validate_onnx_signature.py \
  --onnx fastvlm_full_fp16_embedded_renamed.onnx \
  --cache-len 1023 \
  --kv-style genie

python compile_final.py --onnx fastvlm_full_fp16_embedded_renamed.onnx
```

Outputs:
- `qaihub_bins/fastvlm_full_<JOB_ID>/fastvlm_full.bin`
- `qaihub_bins/fastvlm_full_<JOB_ID>/metadata.json`

## End-to-End VLM Bring-up (Stage 3/4)
This uses existing vision/text encoder binaries and the compiled decoder.

```bash
# Consolidated entrypoint (recommended)
./scripts/fastvlm.sh e2e /path/to/image.jpg "Describe this image"

# Direct end-to-end (image + prompt)
./run_e2e_vlm.sh /path/to/image.jpg "Describe this image"
```

Notes:
- The decoder KV cache is fixed to 1023 tokens (context size 1024). The pipeline truncates the prompt
  to fit that budget. This is a bring-up path, not the final full-context VLM.

## Consolidated Script Entry Point

Use `scripts/fastvlm.sh` to run the full pipeline from a single place:

```bash
# Cache static prompt/image then query
./scripts/fastvlm.sh cache /path/to/image.jpg "Describe the image."

# Export decoder ONNX
./scripts/fastvlm.sh export-decoder --model-dir /path/to/checkpoint --output-dir .

# Compile on QAI Hub
./scripts/fastvlm.sh compile --onnx fastvlm_full_fp16_embedded_renamed.onnx
```

## 📂 Repository Structure

```
VLM_on_edge/
├── VLM_run_experiments_QIDK/     # Main folder for all experiment codebases
│   └── [Experiment_Folders]      # Individual folders for each specific experiment
├── reference_materials/          # Research papers, data sheets, and references
│   └── [Resource_Sets]           # Grouped resources
└── README.md                     # Project rules and documentation
```

---

## 📏 Project Rules & Guidelines

To maintain a clean and collaborative environment, please adhere to the following rules when contributing to this repository.

### 📚 Reference Materials

Located in the `reference_materials` folder.

* **Storage:** Put any reference materials, papers found, and other relevant information here.
* **Organization:** Create new folders for each specific resource-set that you want to add.
* **Naming:** Ensure that all files and folders are organized and named appropriately for easy retrieval.

### 🧪 VLM Based Experiments

Located in the `VLM_run_experiments_QIDK` folder.

**Branching & Workflow:**

1. **Create a New Branch:** strict requirement for each new experiment. Do not work directly on the main branch.
2. **Pull Requests:** Once completed, send a pull request to the main branch to merge your changes.

**Folder Structure & Content:**

* **Isolation:** Create a new folder for each experiment under `VLM_run_experiments_QIDK`.
* **Containment:** Put the experiment code and all other relevant files *inside* that specific folder.
* **Do Not Modify:** Do not modify anything outside your specific experiment folder.

**Documentation Requirements:**

* **README.md:** Each experiment folder **MUST** have its own `README.md` file.
* **Dependencies:** Clearly mention all dependencies required to run the experiment.
* **Setup Instructions:** Provide detailed setup instructions within the experiment's README.

---

## 🚀 Contribution Workflow

1. Clone the repository.
2. Checkout a new branch: `git checkout -b experiment/your-experiment-name`.
3. Create your experiment folder inside `VLM_run_experiments_QIDK/`.
4. Add your code, assets, and the mandatory `README.md`.
5. Push your branch and open a Pull Request.

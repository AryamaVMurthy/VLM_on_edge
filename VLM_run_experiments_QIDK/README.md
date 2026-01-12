```markdown
# 🧪 VLM Experiments (QIDK)

This directory contains the source code and configuration files for all individual experiments conducted on the Qualcomm Hexagon NPU.

## ⚠️ Experiment Protocol

All contributors must strictly follow these rules to ensure experiment isolation and reproducibility:

### 1. Folder Structure
* **One Experiment = One Folder:** Every new experiment must be contained entirely within its own dedicated folder inside this directory.
* **Naming:** Use descriptive folder names (e.g., `SmolVLM_Quantization_Test`, `Genie_Pipeline_Setup`).

### 2. Isolation Rule
* **Sandbox Environment:** All code, assets, `genie_config.json` files, and scripts specific to an experiment must reside **inside** its experiment folder.
* **No External Modifications:** Do not modify files outside your specific experiment folder (e.g., don't change a shared library in a parent directory to fix a local issue).

### 3. Mandatory Documentation
Every experiment folder **MUST** contain a `README.md` file that includes:
* **Dependencies:** Exact SDK versions (QAIRT, Genie, etc.) required.
* **Setup Instructions:** Step-by-step guide to prepare the environment.
* **Execution:** The exact commands used to run the experiment.

### 4. Git Workflow
* **Branching:** Create a new branch for each experiment (e.g., `experiment/my-new-test`).
* **Pull Requests:** Once the experiment folder is complete and documented, submit a Pull Request to merge into `main`.

## 🚀 Directory Layout

```text
VLM_run_experiments_QIDK/
├── [Experiment_Name_A]/
│   ├── code/
│   ├── config.json
│   └── README.md  <-- Mandatory
└── README.md      <-- This file

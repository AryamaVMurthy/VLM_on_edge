# PROJECT KNOWLEDGE BASE

**Generated:** 2026-01-14
**Context:** VLM on Edge (Qualcomm QIDK)

## OVERVIEW
Repository for Vision-Language Model (VLM) experiments on Edge Devices. Focuses on Qualcomm Hexagon NPU optimization, Genie SDK usage, and isolated experiment execution.

## STRUCTURE
```
.
├── VLM_run_experiments_QIDK/     # PRIMARY WORKSPACE: Isolated experiment folders
├── Reference_materials/          # External knowledge, papers, Genie SDK examples
├── qcom_ai_stack/                # Qualcomm AI Stack SDK (Do not modify)
├── work/                         # Working directory artifacts
└── README.md                     # Project rules
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| **New Experiment** | `VLM_run_experiments_QIDK/` | Create new folder, strictly isolated |
| **C++ Examples** | `Reference_materials/GENIE_examples/` | Reference implementation for Genie SDK |
| **SDK Docs** | `qcom_ai_stack/docs/` | QAIRT/QNN official documentation |
| **Python Scripts** | `VLM_run_experiments_QIDK/*/` | Experiment-specific logic |

## CONVENTIONS
* **Isolation**: One experiment = One folder. ALL assets/code inside.
* **Branching**: `experiment/your-name`. PR to `main` when done.
* **Dependencies**: Mention SDK versions (QAIRT, Genie) in experiment `README.md`.
* **Pathing**: Use relative paths within experiment folders.

## ANTI-PATTERNS (THIS PROJECT)
* **FORBIDDEN**: Modifying files outside your specific experiment folder.
* **FORBIDDEN**: Committing to `main` directly.
* **AVOID**: Hardcoding absolute paths (breaks reproducibility).
* **AVOID**: Relying on global shared libraries if experiment-specific versions are needed.

## COMMANDS
```bash
# Common setup (Check experiment README)
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# C++ Build (Typical for Reference Materials)
mkdir build && cd build
cmake ..
make -j8
```

## NOTES
* **Platform**: Qualcomm Hexagon NPU.
* **Key SDKs**: QAIRT, QNN, Genie.
* **Status**: Active research/prototyping.

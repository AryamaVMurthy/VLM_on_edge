# VLM_on_edge
**Project for Paper on Multimodal AI on Edge Devices**

![Status](https://img.shields.io/badge/Status-Active-success)
![Focus](https://img.shields.io/badge/Focus-Vision%20Language%20Models-blue)
![Platform](https://img.shields.io/badge/Platform-Edge%20Devices-orange)

## 📖 Overview
This project aims to explore the application of Vision-Language Models (VLMs) on edge devices, focusing on the challenges and opportunities presented by limited computational resources and network bandwidth. The repository serves as a central hub for experimental code, results, and reference literature.

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

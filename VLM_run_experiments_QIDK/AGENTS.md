# EXPERIMENTS KNOWLEDGE BASE

**Context:** VLM Experiments Workspace

## OVERVIEW
Sandbox for individual VLM experiments. Each folder is a self-contained unit with its own code, config, and documentation.

## STRUCTURE
```
VLM_run_experiments_QIDK/
├── [Experiment_Name]/    # YOUR WORKSPACE
│   ├── code/             # Source code
│   ├── config.json       # Genie/Model config
│   └── README.md         # Mandatory docs
└── README.md             # Directory rules
```

## WORKFLOW
1. **Branch**: `git checkout -b experiment/my-experiment`
2. **Create**: `mkdir VLM_run_experiments_QIDK/my-experiment`
3. **Isolate**: Copy ALL needed assets into this folder.
4. **Document**: Write `README.md` with setup/execution steps.
5. **PR**: Submit pull request when stable.

## RULES
* **Strict Isolation**: No imports from sibling experiment folders.
* **Self-Contained**: If you need a utility, copy it in or propose a shared lib refactor (rare).
* **Reproducibility**: `README.md` must list EXACT SDK versions used.

## COMMON PATTERNS
* **Python**: `python main.py --config config.json`
* **Bash**: `chmod +x run.sh && ./run.sh`
* **Assets**: Store model artifacts in a local `models/` subfolder (git-ignore if large).

## ANTI-PATTERNS
* **Global Dependencies**: Don't assume the runner has your pip packages installed globally. Use `requirements.txt`.
* **Hardcoded Paths**: Use `os.path.dirname(__file__)` or relative paths.

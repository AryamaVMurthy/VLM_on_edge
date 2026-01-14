# REFERENCE KNOWLEDGE BASE

**Context:** External Resources & Genie SDK Examples

## OVERVIEW
Collection of reference implementations, SDK examples, and documentation. Primary source for C++ patterns and QAIRT/Genie usage.

## STRUCTURE
```
Reference_materials/
├── GENIE_examples/       # Genie SDK C++ Reference
│   └── Genie/
│       ├── Genie/src     # Core C++ source (Qualla engine)
│       └── genie-t2t-run # Text-to-Text runner example
└── [Papers/Docs]         # PDFs and datasheets
```

## KEY LOCATIONS (GENIE)
| Component | Path | Function |
|-----------|------|----------|
| **Core Logic** | `GENIE_examples/Genie/Genie/src/qualla` | Inference engine implementation |
| **QNN Engines** | `.../src/qualla/engines/` | HTP/CPU/API backends |
| **Pipeline** | `.../src/pipeline` | Data processing pipeline |
| **Config** | `.../configs/` | JSON configurations for models (Llama2/3) |

## BUILDING (GENIE EXAMPLES)
```bash
# Typical C++ Build
cd GENIE_examples/Genie/Genie
mkdir build && cd build
cmake ..
make -j
```

## CONVENTIONS (C++)
* **CMake**: Used for build configuration.
* **QNN Integration**: heavily relies on `qnn-api` and `qnn-htp` headers.
* **Modular Engines**: Different backends (CPU, HTP) are separated in `engines/`.

## NOTES
* **Read-Only**: Treat this directory as reference. Do not modify unless fixing a bug in the reference itself (rare).
* **Copy-Paste**: Copy useful code snippets to your experiment folder; do not link directly to ensure experiment isolation.

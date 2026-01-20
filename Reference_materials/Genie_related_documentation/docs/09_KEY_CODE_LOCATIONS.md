# Key Code Locations Reference

## Quick Reference Table

| Component | Primary Location |
|-----------|------------------|
| GENIE Headers | `/home/aryamavmurthy/work/QIDK/qcom_ai_stack/include/Genie/` |
| GENIE Source | `/home/aryamavmurthy/work/QIDK/qcom_ai_stack/examples/Genie/Genie/src/` |
| GENIE Configs | `/home/aryamavmurthy/work/QIDK/qcom_ai_stack/examples/Genie/configs/` |
| QNN HTP Headers | `/home/aryamavmurthy/work/QIDK/qcom_ai_stack/include/QNN/HTP/` |
| VLM Experiments | `/home/aryamavmurthy/work/QIDK/VLM_on_edge/VLM_run_experiments_QIDK/` |
| FastVLM E2E | `/home/aryamavmurthy/work/QIDK/VLM_on_edge/VLM_run_experiments_QIDK/fastvlm_yashas/` |
| Reference Materials | `/home/aryamavmurthy/work/QIDK/VLM_on_edge/Reference_materials/` |

---

## GENIE API Headers

```
/home/aryamavmurthy/work/QIDK/qcom_ai_stack/include/Genie/
├── GenieCommon.h        # Status codes, common types
├── GenieDialog.h        # High-level chat API ★ IMPORTANT
├── GenieEngine.h        # Engine management
├── GenieLog.h           # Logging configuration
├── GenieNode.h          # Pipeline node types ★ IMPORTANT
├── GeniePipeline.h      # Composable pipeline API ★ IMPORTANT
├── GenieProfile.h       # Profiling/tracing
├── GenieSampler.h       # Token sampling (temp, top-k, top-p)
└── GenieTokenizer.h     # Tokenization API
```

---

## GENIE Implementation Source

```
/home/aryamavmurthy/work/QIDK/qcom_ai_stack/examples/Genie/Genie/src/

# Top-level API implementations
├── GeniePipeline.cpp    # C API wrapper for pipeline
├── GenieDialog.cpp      # C API wrapper for dialog
├── GenieEngine.cpp      # Engine C API
├── GenieSampler.cpp     # Sampler C API
├── GenieTokenizer.cpp   # Tokenizer C API

# Internal implementations
├── Dialog.cpp           # Dialog class ★ Config validation
├── Engine.cpp           # Engine management
├── Sampler.cpp          # Sampling algorithms (temp, top-k, top-p)
├── Tokenizer.cpp        # Tokenization wrapper

# Pipeline components
├── pipeline/
│   ├── Pipeline.cpp     # Pipeline orchestration
│   ├── Pipeline.hpp
│   ├── Node.cpp         # Base node class
│   ├── Node.hpp
│   ├── Accumulator.cpp  # Embedding accumulator for VLM ★
│   ├── Accumulator.hpp
│   ├── ImageEncoder.cpp # Vision encoder node ★ IMPORTANT
│   ├── ImageEncoder.hpp
│   ├── TextEncoder.cpp  # Text encoder node
│   ├── TextEncoder.hpp
│   ├── TextGenerator.cpp # Text generation node ★ IMPORTANT
│   └── TextGenerator.hpp

# Qualla engine (internal library)
├── qualla/
│   ├── dialog.cpp       # Dialog implementation
│   ├── engine.cpp       # Engine factory
│   ├── encoder.cpp      # Encoder base
│   ├── embedding.cpp    # Embedding handling
│   ├── sampler.cpp      # Sampling implementation
│   ├── tokenizer.cpp    # Tokenizer implementation
│   │
│   ├── engines/
│   │   ├── lib.cpp      # Engine registration
│   │   └── qnn-htp/     # HTP backend ★ CRITICAL
│   │       ├── nsp-kvmanager.cpp   # KV cache management ★★★
│   │       ├── nsp-kvmanager.hpp
│   │       ├── nsp-model.cpp       # Model execution
│   │       ├── nsp-base-model.cpp
│   │       ├── nsp-base-model.hpp
│   │       ├── nsp-graph.cpp       # Graph management
│   │       ├── nsp-graph.hpp
│   │       └── nsp-image-model.cpp # VLM vision model
│   │
│   ├── encoders/
│   │   ├── image-encoders/
│   │   │   ├── imageEncoder.cpp    # Image encoder ★
│   │   │   └── imageEncoder.hpp
│   │   └── text-encoders/
│   │       ├── LUT.cpp             # Lookup table encoder
│   │       ├── LUT.hpp
│   │       ├── basic.cpp
│   │       └── basic.hpp
│   │
│   └── utils/
│       ├── threadpool.cpp
│       └── utils.cpp
```

---

## VLM Reference Materials

```
/home/aryamavmurthy/work/QIDK/VLM_on_edge/Reference_materials/

├── GENIE_examples/
│   └── Genie/Genie/src/     # Copy of GENIE source with annotations
│
├── Genie_related_documentation/
│   ├── GENIE_pipeline_docs1.txt
│   └── GENIE_pipeline_docs2.md
│
└── README.md
```

---

## FastVLM Experiment Files

```
/home/aryamavmurthy/work/QIDK/VLM_on_edge/VLM_run_experiments_QIDK/fastvlm_yashas/

├── handoff/
│   ├── scripts/
│   │   ├── run_e2e_int8_device.sh    # ★★★ MAIN E2E SCRIPT
│   │   ├── export_fastvlm_fp16.py    # Vision encoder export
│   │   └── 04_build_int8_vision_encoder.sh
│   │
│   └── models/                       # Model artifacts
│
├── 2.42.0.251225/                    # QAIRT SDK version
│   ├── bin/aarch64-android/
│   │   ├── genie-t2t-run            # GENIE runner binary
│   │   └── qnn-net-run              # QNN inference runner
│   │
│   ├── lib/aarch64-android/
│   │   ├── libGenie.so              # GENIE library
│   │   ├── libQnnHtp.so             # HTP backend
│   │   ├── libQnnHtpPrepare.so
│   │   └── libQnnHtpV79Stub.so      # V79 (8 Elite) stub
│   │
│   └── examples/Genie/              # SDK examples (same as qcom_ai_stack)
│
└── README.md
```

---

## FastVLM Export Project

```
/home/aryamavmurthy/work/QIDK/VLM_on_edge/fastvlm_npu_export_project/

├── export_decoder.py          # Export LLM decoder to ONNX segments
├── export_embeddings.py       # Extract embedding LUT
├── compile_qaihub.py          # Submit to QAI Hub for compilation
├── fastvlm_genie_npu.json     # GENIE config for NPU
├── clean_project.py           # Cleanup utility
└── README_CLEAN.md            # Documentation
```

---

## QNN HTP Headers

```
/home/aryamavmurthy/work/QIDK/qcom_ai_stack/include/QNN/HTP/

├── QnnHtpGraph.h              # Graph configuration (precision, VTCM)
├── QnnHtpDevice.h             # Device configuration (arch version)
├── QnnHtpContext.h            # Context binary options
├── QnnHtpPerfInfrastructure.h # Performance/power settings
├── QnnHtpMem.h                # Memory management
└── QnnHtpProfile.h            # Profiling
```

---

## Sample Configurations

### VLM Configuration (GLM-4V)
```
/home/aryamavmurthy/work/QIDK/qcom_ai_stack/examples/Genie/configs/glm-4v/
├── glm-4v.json                # Main text-generator config
├── siglip.json                # Vision encoder config
└── text-encoder.json          # Text encoder config
```

### LLM Configurations
```
/home/aryamavmurthy/work/QIDK/qcom_ai_stack/examples/Genie/configs/
├── llama2-7b/llama2-7b-htp.json
├── llama3-8b/llama3-8b-htp.json
├── llama3-3b/llama3-3b-htp.json
└── phi3-mini/phi3-mini-genaitransformer-htp-kv-share.json
```

---

## VLM Project Planning

```
/home/aryamavmurthy/work/QIDK/VLM-proj/plan.md
```

Contains master plan for deploying VLM on Snapdragon 8 Elite.

---

## Documentation

```
/home/aryamavmurthy/work/QIDK/qcom_ai_stack/docs/
├── QAIRT-Docs/Genie/index.html    # GENIE SDK documentation ★
├── QNN/index.html                  # QNN SDK documentation
└── HAP_compute_res.md              # VTCM window documentation
```

---

## Binary Tools

```
# Host tools (x86_64)
/home/aryamavmurthy/work/QIDK/qcom_ai_stack/bin/x86_64-linux-clang/
├── qnn-context-binary-generator   # Compile ONNX → context binary
├── qnn-onnx-converter             # ONNX → QNN format
└── qnn-context-binary-utility     # Inspect context binaries

# Device binaries (aarch64)
/home/aryamavmurthy/work/QIDK/qcom_ai_stack/bin/aarch64-android/
├── genie-t2t-run                  # GENIE text-to-text runner
├── genie-app                      # GENIE application runner
└── qnn-net-run                    # QNN inference runner
```

---

## Libraries

```
# Host libraries
/home/aryamavmurthy/work/QIDK/qcom_ai_stack/lib/x86_64-linux-clang/

# Device libraries
/home/aryamavmurthy/work/QIDK/qcom_ai_stack/lib/aarch64-android/
├── libGenie.so                    # GENIE runtime
├── libQnnHtp.so                   # HTP backend
├── libQnnHtpPrepare.so            # HTP graph preparation
├── libQnnCpu.so                   # CPU backend
├── libQnnGpu.so                   # GPU backend
├── libQnnGenAiTransformer.so      # CPU transformer backend
└── libQnnSystem.so                # System utilities

# Hexagon DSP libraries
/home/aryamavmurthy/work/QIDK/qcom_ai_stack/lib/hexagon-v79/unsigned/
└── libQnnHtpV79Skel.so            # V79 skeleton for NPU
```

# Comprehensive Setup & VLM Deployment Report - HDK 8750 (Snapdragon 8 Elite)

## 1. Environment Status
The core environment is fully configured in `/home/aryamavmurthy/work/QIDK/qcom_ai_stack`. 

### Key Variables Defined in `env_setup.sh`:
- **QAIRT Version**: 2.42.0 (Latest Dec 2025 release)
- **Architecture**: `v79` (Snapdragon 8 Elite)
- **QNN_SDK_ROOT**: `/home/aryamavmurthy/work/QIDK/qcom_ai_stack`
- **HEXAGON_TOOLS_ROOT**: `/home/aryamavmurthy/work/QIDK/qcom_ai_stack/tools/HEXAGON_Tools/19.0.07/Tools`
- **Python Venv**: `/home/aryamavmurthy/work/QIDK/venv` (includes `qai-hub`)

---

## 2. Immediate Next Steps (Setup Checklist)

### A. Initialize Terminal
Every time you open a new shell to work on this project, run:
```bash
source /home/aryamavmurthy/work/QIDK/VLM-proj/env_setup.sh
```

### B. Authenticate QAI Hub
Since you are using QAI Hub for cloud-based compilation (recommended for initial VLM tests), you need to login:
```bash
# Get your API token from https://aihub.qualcomm.com/
qai-hub configure
```

### C. Verify HTP Accessibility
Check if the QNN HTP (Hexagon Tensor Processor) libraries are correctly detected:
```bash
ls $QNN_BACKEND_LIB/libQnnHtp.so
```

---

## 3. VLM Deployment Strategy (Making it Work)

To run a Vision-Language Model (VLM) like **nanoVLM** or **SmolVLM** on the NPU:

### Step 1: Model Partitioning
VLMs must be split into three parts for optimal NPU execution:
1. **Vision Encoder**: (e.g., SigLIP) - Best run on NPU as a `v79` binary.
2. **Projection Layer**: Simple MLP - Can be fused with Vision or Decoder.
3. **LLM Decoder**: (e.g., SmolLM2) - Requires specialized quantization (W4A16) for the Genie runtime.

### Step 2: Conversion (Using QNN Tools)
Convert your ONNX/PyTorch models to QNN `.cpp/.bin`:
```bash
# Example for Vision part
qnn-onnx-converter -i vision_model.onnx -o vision_model.cpp
qnn-context-binary-generator --model vision_model.cpp --backend libQnnHtp.so --htp_arch v79 --binary_file vision.bin
```

### Step 3: Genie Runtime Integration
Qualcomm **Genie** is the high-level API for LLM/VLM.
1. **Configuration**: Create a `genie_config.json` mapping your `vision.bin` and `llm.bin`.
2. **Execution**: Use the `genie-t2t-run` utility found in your stack to launch the interactive prompt.

---

## 4. Hardware Verification (On HDK 8 Elite)
Once your binaries are ready, push them to the device via ADB:
```bash
adb shell mkdir -p /data/local/tmp/vlm
adb push /home/aryamavmurthy/work/QIDK/qcom_ai_stack/lib/aarch64-android/libQnnHtpV79Stub.so /data/local/tmp/vlm/
# ... push other libs and your compiled .bin files
```

---

## 5. Troubleshooting Reference
- **Arch Error**: If you see "Unsupported Arch", ensure you are using `--htp_arch v79`.
- **Memory Error**: For large VLMs, ensure you use `W4A16` quantization for the weights to fit into the NPU's TCM (Tightly Coupled Memory).
- **Missing Symbols**: Always ensure `LD_LIBRARY_PATH` includes the stub libraries when running on-device.

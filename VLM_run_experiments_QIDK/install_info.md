Part 1: The Complete Product List
(Copied from your output for reference)

adreno_opencl_ml_sdk

adreno_opencl_sdk

broadcast_sample_applications

computervisionsolutions

halide2.4

hexagon8.4

hexagon_ide

hexagon_kl

hexagon_open_access

hexagon_sdk

hexagonsdk4.x

hexagonsdk6.x (Critical for 8750)

hmx_plugin_qemu

lpai

pcat

qods

qpm3

qualcomm_ai_engine_direct (Critical - This is QAIRT)

qualcomm_ai_runtime_community

qualcomm_ai_runtime_sdk

qualcomm_aimet_pro (Critical - Optimization)

qualcomm_aware_location_android_sdk

qualcomm_aware_location_evaluation_tools

qualcomm_aware_location_native_sdk

qualcomm_broadcast_dash_sdk

qualcomm_broadcast_sdk

qualcomm_device_loader

qualcomm_fast_cv

qualcomm_ide_developer_edition

qualcomm_irf_sdk

qualcomm_irp_sdk

qualcomm_launcher

qualcomm_neural_processing_sdk (Legacy SNPE)

qualcomm_neural_processing_sdk_public

qualcomm_profiler

qualcomm_security_tools

qualcomm_software_center

qualcomm_studio_cli

qualcomm_userspace_driver

qualcommsimulationplatformsdk

qud

quts

quts2

robotics_ide

sdllvm_arm

sdllvm_cpu

snapdragon_guardian_app_images

snapdragon_guardian_infra

snapdragon_llvm_for_arm_open_access

snapdragon_llvm_windows_on_snapdragon_toolchain_oem

snapdragon_profiler

snapdragon_spaces_compatibility_plugin

snapdragon_spaces_platform

snapdragonprofilercliandroid

snapdragonprofilercliwindows

swiv

telematics_sdk_ide

telematics_sdk_ide_config

telematics_sdk_simulator

tutorial_for_llama2

tutorial_for_llama2_compute

tutorial_for_llama3_compute

tutorial_for_phi3_5_compute (Recommended for VLM/Genie)

tutorial_for_stable_diffusion

tutorial_for_stable_diffusion_2_1_compute

tutorial_for_stable_diffusion_compute

tvm.1.0

voiceai_asr

voiceai_asr_community

voiceai_lava

voiceai_translation

voiceai_tts

Part 2: The Exact Setup Plan for HDK 8750 (Snapdragon 8 Elite)
Based on your list, "Genie SDK" is not explicitly named qualcomm_genie_sdk in your entitlement. However, you have qualcomm_ai_engine_direct (the engine) and the tutorial_for_... packages.

Strategy: We will install the Engine (QAIRT), the Hexagon SDK (for the 8750 v80 architecture), AIMET (for quantization), and use the tutorial_for_phi3_5_compute to get the Genie runtime bits or configuration examples suited for modern Multimodal models.

Step 1: Install Qualcomm AI Engine Direct (QAIRT)
This is the most important package. It contains the converters and the HTP backend.

Activate License:

Bash

qpm-cli --license-activate qualcomm_ai_engine_direct
Install:

Bash

qpm-cli --install qualcomm_ai_engine_direct
Locate & Export:

The path will be roughly /opt/qcom/aistack/qualcomm_ai_engine_direct/<version>.

Run this to set the variable:

Bash

export QNN_SDK_ROOT=$(find /opt/qcom/aistack/qualcomm_ai_engine_direct -maxdepth 1 -type d -name "2.*" | sort -V | tail -n1)
echo "QNN SDK is at: $QNN_SDK_ROOT"
Step 2: Install Hexagon SDK 6.x
The Snapdragon 8 Elite (SM8750) uses the Hexagon V79/V80 architecture. SDK 4.x/5.x is too old. You must use hexagonsdk6.x.

Activate & Install:

Bash

qpm-cli --license-activate hexagonsdk6.x
qpm-cli --install hexagonsdk6.x
Why you need this: It contains the specific compiler libraries (libQnnHtpV79.so or V80) that QAIRT needs to build the context binary for your specific chip.

Step 3: Install AIMET Pro (Quantization)
For VLM/LLM deployment, you must quantize.

Activate & Install:

Bash

qpm-cli --license-activate qualcomm_aimet_pro
qpm-cli --install qualcomm_aimet_pro
Step 4: Get the Genie Environment (Via Tutorials)
Since qualcomm_genie_sdk is missing from your list, the best way to get the Genie config files, prompt processors, and potentially the runtime binaries for a VLM is to download the Phi-3.5 Vision Tutorial.

Activate & Install:

Bash

qpm-cli --license-activate tutorial_for_phi3_5_compute
qpm-cli --install tutorial_for_phi3_5_compute
Explore:

Go to the install directory (likely /opt/qcom/aistack/tutorial_for_phi3_5_compute/).

Look for a genie folder or bin folder inside. This package often bundles the necessary Genie runtime libraries or scripts to fetch them because Phi-3.5 requires the latest Genie features.

Step 5: Install QAI Hub (Python)
This is not in QPM. It is a Python tool for cloud compilation (easier than local setup).

Install:

Bash

pip install qai-hub
Configure:

Bash

qai-hub configure
# Enter your API token from aihub.qualcomm.com
Summary of Commands for Your Terminal
Copy and run these blocks one by one to set up your HDK 8750 environment.

Block 1: Licenses

Bash

qpm-cli --license-activate qualcomm_ai_engine_direct
qpm-cli --license-activate hexagonsdk6.x
qpm-cli --license-activate qualcomm_aimet_pro
qpm-cli --license-activate tutorial_for_phi3_5_compute
Block 2: Installation

Bash

qpm-cli --install qualcomm_ai_engine_direct
qpm-cli --install hexagonsdk6.x
qpm-cli --install qualcomm_aimet_pro
qpm-cli --install tutorial_for_phi3_5_compute
Block 3: Verify

Bash

ls /opt/qcom/aistack/
# You should see folders for all the installed products above.

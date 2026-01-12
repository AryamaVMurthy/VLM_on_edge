#!/bin/bash

export QCOM_AI_STACK=/home/aryamavmurthy/work/QIDK/qcom_ai_stack
export QNN_SDK_ROOT=$QCOM_AI_STACK
export HEXAGON_SDK_ROOT=$QCOM_AI_STACK
export HEXAGON_TOOLS_VERSION=19.0.07
export HEXAGON_TOOLS_ROOT=$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/$HEXAGON_TOOLS_VERSION/Tools

source /home/aryamavmurthy/work/QIDK/venv/bin/activate

export PATH=$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH

export PATH=$HEXAGON_TOOLS_ROOT/bin:$PATH

export PYTHONPATH=$QCOM_AI_STACK/lib/python:$PYTHONPATH
export QNN_BACKEND_LIB=$QNN_SDK_ROOT/lib/x86_64-linux-clang
export PATH=$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH
export LD_LIBRARY_PATH=$QNN_BACKEND_LIB:$LD_LIBRARY_PATH

# Qualcomm Python SDK Path
export PYTHONPATH=$QCOM_AI_STACK/lib/python:$PYTHONPATH

# QAIRT Backend Path (Crucial for HTP/NPU)
export QNN_BACKEND_LIB=$QNN_SDK_ROOT/lib/x86_64-linux-clang

# Final Path check for interactive use
export PATH=$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH
export LD_LIBRARY_PATH=$QNN_BACKEND_LIB:$LD_LIBRARY_PATH

echo "--------------------------------------------------------"
echo "Snapdragon 8 Elite (HDK 8750) Environment Ready"
echo "QAIRT: 2.42.0"
echo "QNN_SDK_ROOT: $QNN_SDK_ROOT"
echo "HEXAGON_SDK_ROOT: $HEXAGON_SDK_ROOT"
echo "Python Venv: Active (qai-hub installed)"
echo "--------------------------------------------------------"

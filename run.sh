#!/bin/bash
# Wrapper to run scripts using the local venv

VENV_PATH="/home/aryamavmurthy/work/QIDK/VLM_on_edge/.worktrees/fastvlm-npu-export/venv"
PYTHON="$VENV_PATH/bin/python3"

if [ ! -f "$PYTHON" ]; then
    echo "Error: Python interpreter not found at $PYTHON"
    exit 1
fi

# Execute arguments
exec "$PYTHON" "$@"

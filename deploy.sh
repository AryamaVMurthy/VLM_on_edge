#!/bin/bash
# FastVLM NPU Deployment Script (runtime + decoder artifacts)

set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Pushing runtime + decoder artifacts"
bash "${ROOT_DIR}/device/push_runtime.sh"
bash "${ROOT_DIR}/device/push_model.sh"

echo "==> Ready"

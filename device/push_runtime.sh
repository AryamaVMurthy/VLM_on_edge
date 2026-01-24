#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEVICE_DIR="${DEVICE_DIR:-/data/local/tmp/fastvlm}"
QCOM_ROOT="${QCOM_ROOT:-${ROOT_DIR}/../../../qcom_ai_stack}"

mkdir -p "${ROOT_DIR}/device"

echo "==> Pushing runtime to ${DEVICE_DIR}"
adb shell "mkdir -p ${DEVICE_DIR}"

adb push "${QCOM_ROOT}/bin/aarch64-android/genie-t2t-run" "${DEVICE_DIR}/"
adb push "${QCOM_ROOT}/bin/aarch64-android/qnn-net-run" "${DEVICE_DIR}/"

adb push "${QCOM_ROOT}/lib/aarch64-android/libGenie.so" "${DEVICE_DIR}/"
adb push "${QCOM_ROOT}/lib/aarch64-android/libQnnHtp.so" "${DEVICE_DIR}/"
adb push "${QCOM_ROOT}/lib/aarch64-android/libQnnSystem.so" "${DEVICE_DIR}/"
adb push "${QCOM_ROOT}/lib/aarch64-android/libQnnHtpPrepare.so" "${DEVICE_DIR}/"
adb push "${QCOM_ROOT}/lib/aarch64-android/libQnnHtpNetRunExtensions.so" "${DEVICE_DIR}/"
adb push "${QCOM_ROOT}/lib/aarch64-android/libQnnHtpV79Stub.so" "${DEVICE_DIR}/"
adb push "${QCOM_ROOT}/lib/aarch64-android/libQnnHtpV79CalculatorStub.so" "${DEVICE_DIR}/"

# Hexagon DSP skeleton (V79 for Snapdragon 8 Elite)
adb push "${QCOM_ROOT}/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so" "${DEVICE_DIR}/"
adb push "${QCOM_ROOT}/lib/hexagon-v79/unsigned/libQnnHtpV79.so" "${DEVICE_DIR}/"

# Backend extensions config referenced by GENIE JSON.
adb push "${QCOM_ROOT}/examples/Genie/configs/htp_backend_ext_config.json" "${DEVICE_DIR}/"

echo "==> Runtime pushed"

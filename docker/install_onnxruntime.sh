#!/bin/sh
# Detect arch and download ONNX Runtime prebuilt. Used inside Docker build.
set -e
VERSION="${1:?missing version}"
ROOT="${2:?missing install root}"

ARCH=$(uname -m)
case "$ARCH" in
  x86_64)    ONNX_ARCH=x64 ;;
  aarch64|arm64) ONNX_ARCH=aarch64 ;;
  *)         echo "Unsupported arch: $ARCH"; exit 1 ;;
esac

URL="https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/onnxruntime-linux-${ONNX_ARCH}-${VERSION}.tgz"
mkdir -p "$ROOT"
wget -q "$URL" -O /tmp/onnxruntime.tgz
tar xzf /tmp/onnxruntime.tgz -C "$ROOT" --strip-components=1
rm /tmp/onnxruntime.tgz

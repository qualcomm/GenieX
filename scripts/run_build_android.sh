#!/usr/bin/env bash
set -euo pipefail

IMAGE=ghcr.io/snapdragon-toolchain/arm64-android:v0.3
WORKDIR=$(pwd)

docker run -it --rm \
  -u 0:0 \
  --platform linux/amd64 \
  -v "$WORKDIR":/workspace \
  "$IMAGE" \
  bash -c "
    set -e
    apt-get update
    apt-get install -y make cmake build-essential
    cd /workspace
    rm -rf build-android
    bash scripts/build_android.sh
  "

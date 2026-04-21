#!/usr/bin/env bash
# Build the GenieX SDK on Linux (cross-compile to linux-arm64 by default).
#
# Environment inputs:
#   GENIEX_VERSION      (required)  Version string baked into binaries.
#   TOOLCHAIN_FILE      (optional)  Default: sdk/cmake/arm64-linux-gnu.cmake.
#   BUILD_DIR           (optional)  Default: sdk/build.
#   INSTALL_PREFIX      (optional)  Default: sdk/pkg-geniex.
#   EXTRA_CMAKE_FLAGS   (optional)  Appended verbatim to `cmake -B`.

set -euo pipefail

: "${GENIEX_VERSION:?GENIEX_VERSION is required}"
TOOLCHAIN_FILE="${TOOLCHAIN_FILE:-sdk/cmake/arm64-linux-gnu.cmake}"
BUILD_DIR="${BUILD_DIR:-sdk/build}"
INSTALL_PREFIX="${INSTALL_PREFIX:-sdk/pkg-geniex}"
EXTRA_CMAKE_FLAGS="${EXTRA_CMAKE_FLAGS:-}"

set -x
# shellcheck disable=SC2086  # EXTRA_CMAKE_FLAGS is intentionally word-split.
cmake -B "$BUILD_DIR" -S sdk \
  -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE" \
  -DGENIEX_VERSION="$GENIEX_VERSION" \
  -DGENIEX_TEST=OFF \
  -DGENIEX_DEBUG=OFF \
  -DGENIEX_DL=ON \
  -DGENIEX_PLUGIN_LLAMA_CPP=ON \
  -DGENIEX_PLUGIN_QAIRT=ON \
  -DGGML_OPENCL=OFF \
  $EXTRA_CMAKE_FLAGS

cmake --build "$BUILD_DIR" -j "$(nproc)"
cmake --install "$BUILD_DIR" --prefix "$INSTALL_PREFIX"

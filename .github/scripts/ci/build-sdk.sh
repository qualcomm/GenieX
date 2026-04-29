#!/usr/bin/env bash
# Build the GenieX SDK on Linux.
#
# Environment inputs:
#   GENIEX_VERSION      (required)  Version string baked into binaries.
#   PLATFORM            (optional)  linux-arm64 (default) or android-arm64.
#   BUILD_DIR           (optional)  Default: sdk/build.
#   INSTALL_PREFIX      (optional)  Default: sdk/pkg-geniex.
#   EXTRA_CMAKE_FLAGS   (optional)  Appended verbatim to `cmake -B`.
#
# android-arm64 additionally requires ANDROID_NDK_ROOT and expects the
# Snapdragon toolchain container (ghcr.io/snapdragon-toolchain/arm64-android)
# which provides HEXAGON_SDK_ROOT / HEXAGON_TOOLS_ROOT. See docs/build.md.

set -euo pipefail

: "${GENIEX_VERSION:?GENIEX_VERSION is required}"
PLATFORM="${PLATFORM:-linux-arm64}"
BUILD_DIR="${BUILD_DIR:-sdk/build}"
INSTALL_PREFIX="${INSTALL_PREFIX:-sdk/pkg-geniex}"
EXTRA_CMAKE_FLAGS="${EXTRA_CMAKE_FLAGS:-}"

case "$PLATFORM" in
  linux-arm64)
    TOOLCHAIN_FILE="${TOOLCHAIN_FILE:-sdk/cmake/arm64-linux-gnu.cmake}"
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
      -DGENIEX_MODEL_MANAGER=ON \
      -DGGML_OPENCL=OFF \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
      $EXTRA_CMAKE_FLAGS
    ;;
  android-arm64)
    : "${ANDROID_NDK_ROOT:?ANDROID_NDK_ROOT is required for android-arm64}"
    # CMakePresets.json drives the toolchain + Snapdragon options for Android;
    # keep the invocation aligned with docs/build.md so local and CI match.
    # GENIEX_MODEL_MANAGER is disabled on Android here because the Rust
    # crate has no cross-build wiring yet; tracked as #222.
    PRESET="arm64-android-snapdragon-release"
    BUILD_DIR="sdk/build-${PRESET}"
    set -x
    # shellcheck disable=SC2086  # EXTRA_CMAKE_FLAGS is intentionally word-split.
    cmake -S sdk --preset "$PRESET" \
      -DGENIEX_VERSION="$GENIEX_VERSION" \
      -DGENIEX_TEST=OFF \
      -DGENIEX_DL=ON \
      -DGENIEX_MODEL_MANAGER=OFF \
      $EXTRA_CMAKE_FLAGS
    ;;
  *)
    echo "Unsupported PLATFORM: $PLATFORM" >&2
    exit 1
    ;;
esac

cmake --build "$BUILD_DIR" -j "$(nproc)"
cmake --install "$BUILD_DIR" --prefix "$INSTALL_PREFIX"

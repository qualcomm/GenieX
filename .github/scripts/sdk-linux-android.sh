#!/usr/bin/env bash
# Build the GenieX SDK on Linux. Drives CMakePresets.json rather than passing
# toolchain/options ad-hoc so the CI build stays in sync with the presets
# documented in notes/build.md.
#
# Environment inputs:
#   GENIEX_VERSION      (required)  Version string baked into binaries.
#   PLATFORM            (optional)  linux-arm64 | android-arm64. Default:
#                                   linux-arm64. Selects the CMake preset.
#   INSTALL_PREFIX      (optional)  Default: sdk/pkg-geniex.
#   EXTRA_CMAKE_FLAGS   (optional)  Appended verbatim to `cmake --preset`.

set -euo pipefail

: "${GENIEX_VERSION:?GENIEX_VERSION is required}"
PLATFORM="${PLATFORM:-linux-arm64}"
INSTALL_PREFIX="${INSTALL_PREFIX:-sdk/pkg-geniex}"
EXTRA_CMAKE_FLAGS="${EXTRA_CMAKE_FLAGS:-}"

case "$PLATFORM" in
  linux-arm64)
    # Runs inside ghcr.io/qcom-ai-hub/geniex-toolchain-linux (derived
    # from snapdragon-toolchain/arm64-linux). Base provides
    # HEXAGON_SDK_ROOT + HEXAGON_TOOLS_ROOT + OPENCL_SDK_ROOT; the
    # derived layer adds rustup + aarch64-unknown-linux-gnu target, so
    # GGML_HEXAGON and GENIEX_MODEL_MANAGER stay on the preset defaults.
    PRESET="arm64-linux-snapdragon-release"
    ;;
  android-arm64)
    # Runs inside ghcr.io/snapdragon-toolchain/arm64-android which provides
    # $ANDROID_NDK_ROOT; the CI job pre-installs the aarch64-linux-android
    # Rust target. sdk/CMakeLists.txt handles the cargo --target +
    # CARGO_TARGET_*_LINKER wiring.
    PRESET="arm64-android-snapdragon-release"
    EXTRA_CMAKE_FLAGS="$EXTRA_CMAKE_FLAGS -DGENIEX_MODEL_MANAGER=ON"
    ;;
  *)
    echo "Unsupported PLATFORM: $PLATFORM" >&2
    exit 1
    ;;
esac

BUILD_DIR="sdk/build-${PRESET}"

set -x
# shellcheck disable=SC2086  # EXTRA_CMAKE_FLAGS is intentionally word-split.
cmake -S sdk --preset "$PRESET" \
  -DGENIEX_VERSION="$GENIEX_VERSION" \
  -DGENIEX_TEST=OFF \
  -DGENIEX_DL=ON \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  $EXTRA_CMAKE_FLAGS

cmake --build "$BUILD_DIR" -j "$(nproc)"
cmake --install "$BUILD_DIR" --prefix "$INSTALL_PREFIX"

#!/usr/bin/env bash
# Build the GenieX SDK on Linux. Drives CMakePresets.json rather than passing
# toolchain/options ad-hoc so the CI build stays in sync with the presets
# documented in docs/build.md.
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
    # Runs inside ghcr.io/snapdragon-toolchain/arm64-linux which provides
    # HEXAGON_SDK_ROOT + HEXAGON_TOOLS_ROOT + OPENCL_SDK_ROOT, so GGML_HEXAGON
    # and GENIEX_MODEL_MANAGER (Rust cross w/ aarch64-unknown-linux-gnu) can
    # stay on the preset defaults.
    PRESET="arm64-linux-snapdragon-release"
    ;;
  android-arm64)
    PRESET="arm64-android-snapdragon-release"
    # GENIEX_MODEL_MANAGER has no cross-build wiring for android-arm64 yet
    # (tracked as #222); keep it disabled until that lands.
    EXTRA_CMAKE_FLAGS="$EXTRA_CMAKE_FLAGS -DGENIEX_MODEL_MANAGER=OFF"
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

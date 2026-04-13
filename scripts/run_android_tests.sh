#!/bin/bash
#=============================================================================
#
#  Copyright (c) 2025 Qualcomm
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm
#
#=============================================================================

# Create a log directory
LOG_DIR="build_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/android_build_$TIMESTAMP.log"

# Redirect all output to both console and log file
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Build started at $(date)"
echo "=============================================="

# Don't exit immediately on error, we want to log as much as possible
set +e

# Record environment information
echo "Environment Information:"
echo "- Bash version: $BASH_VERSION"
echo "- OS: $(uname -a)"
echo "- CMake version: $(cmake --version | head -n 1)"
echo "- Environment variables:"
env | grep -E "ANDROID|PATH|HOME|CMAKE" | sort

# Check if NDK path is set
if [ -z "$ANDROID_NDK_ROOT" ]; then
    echo "ERROR: ANDROID_NDK_ROOT environment variable is not set."
    echo "Please set it to your Android NDK installation path, for example:"
    echo "  export ANDROID_NDK_ROOT=/path/to/android-ndk"
    exit 1
fi

echo "- Android NDK path: $ANDROID_NDK_ROOT"
echo "- Android NDK version: $(cat "$ANDROID_NDK_ROOT/source.properties" 2>/dev/null | grep Pkg.Revision || echo "Could not determine NDK version")"
echo "=============================================="

# Set default variables
BUILD_DIR="build-android"
ABI="arm64-v8a"  # Default to arm64-v8a
BUILD_TYPE="Release"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --abi)
            ABI="$2"
            shift
            shift
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift
            shift
            ;;
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --abi <abi>        Target ABI (arm64-v8a). Default: arm64-v8a"
            echo "  --build-dir <dir>  Build directory. Default: build-android"
            echo "  --debug            Build debug version. Default: Release"
            echo "  --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate ABI
VALID_ABIS=("arm64-v8a")
VALID_ABI=false
for valid_abi in "${VALID_ABIS[@]}"; do
    if [ "$ABI" = "$valid_abi" ]; then
        VALID_ABI=true
        break
    fi
done

if [ "$VALID_ABI" = false ]; then
    echo "ERROR: Invalid ABI specified: $ABI"
    echo "Valid ABIs are: ${VALID_ABIS[*]}"
    exit 1
fi


echo "Build configuration:"
echo "- ABI: $ABI"
echo "- Build type: $BUILD_TYPE"
echo "- Build directory: $BUILD_DIR"
echo "=============================================="

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR" || { echo "Failed to change to build directory"; exit 1; }

# Configure with CMake
echo "Configuring for Android with ABI: $ABI"
set -x  # Print commands before they're executed

# Note: Use -DGENIEX_PLUGIN_LLAMA_CPP=OFF -DGENIEX_DL=ON -DGENIEX_DEBUG=ON for unit testing (plugins loaded dynamically)
#       Use -DGENIEX_PLUGIN_LLAMA_CPP=ON -DGENIEX_DL=OFF -DGENIEX_DEBUG=OFF for Android App build (plugins statically linked)
cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
      -DANDROID_ABI="$ABI" \
      -DANDROID_PLATFORM=android-23 \
      -DANDROID_STL=c++_static \
      -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
      -DCMAKE_VERBOSE_MAKEFILE=ON \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_CXX_FLAGS="-Wno-error=unused-function -Wno-error=unused-local-typedef -Wno-error=for-loop-analysis -Wno-error" \
      -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined" \
      -DGENIEX_PLUGIN_QNN=ON \
      -DGENIEX_PLUGIN_LLAMA_CPP=OFF \
      -DGENIEX_DL=ON \
      -DGENIEX_DEBUG=ON \
      -DGENIEX_TEST=ON \
      -DGENIEX_BINDING_PYTHON=OFF \
      ..

CMAKE_RESULT=$?
set +x

if [ $CMAKE_RESULT -ne 0 ]; then
    echo "=============================================="
    echo "CMake configuration failed with error code $CMAKE_RESULT"
    echo "Please check the log file: $LOG_FILE"
    exit $CMAKE_RESULT
fi

# Build the QNN plugin
echo "=============================================="
echo "Building GENIEX_PLUGIN_QNN for Android..."
set -x
cmake --build . -j$(nproc) --verbose
BUILD_RESULT=$?
set +x

if [ $BUILD_RESULT -ne 0 ]; then
    echo "=============================================="
    echo "Build failed with error code $BUILD_RESULT"
    echo "Please check the log file: $LOG_FILE"
    
    echo "Checking for common errors in the build log..."
    grep -E "error:|undefined reference|no such file|cannot find" "$LOG_FILE" > "$LOG_DIR/error_summary_$TIMESTAMP.txt"
    echo "Error summary saved to: $LOG_DIR/error_summary_$TIMESTAMP.txt"

    exit $BUILD_RESULT
fi

echo "=============================================="
echo "Build completed successfully at $(date)"
echo "Build artifacts are located at:"
echo "  - Bridge Library: $BUILD_DIR/out/libgeniex_bridge.so"
echo "  - QNN Plugin: $BUILD_DIR/out/npu/libgeniex_plugin.so"
echo "  - QNN Runtime: $BUILD_DIR/out/npu/htp-files/"
echo "  - Test Executables:"
echo "    * $BUILD_DIR/out/tests/geniex_test_llm (liquid)"
echo "    * $BUILD_DIR/out/tests/geniex_test_vlm (omni-neural)"
echo "    * $BUILD_DIR/out/tests/geniex_test_asr (parakeet)"
echo "    * $BUILD_DIR/out/tests/geniex_test_cv (paddleocr)"
echo "    * $BUILD_DIR/out/tests/geniex_test_embedding (embed-gemma, embed-neural)"
echo "    * $BUILD_DIR/out/tests/geniex_test_rerank (jina-rerank)"
echo ""
echo "Full build log saved to: $LOG_FILE"
echo ""

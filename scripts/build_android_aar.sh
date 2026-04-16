#!/bin/bash
set -e

# Run in Docker by default
if [ ! -f "/.dockerenv" ]; then
    docker run -it --rm -u $(id -u):$(id -g) \
        --volume $(pwd):/workspace \
        --platform linux/amd64 \
        ghcr.io/snapdragon-toolchain/arm64-android:v0.3 \
        bash -c "cd /workspace && ./build_android_aar.sh"
    exit $?
fi

# Build SDK
cd sdk
cmake --preset arm64-android-snapdragon-release
cmake --build build-arm64-android-snapdragon-release -j$(nproc)
cmake --install build-arm64-android-snapdragon-release --prefix ../build-android/install

# Build AAR
cd ../bindings/android
chmod +x gradlew
./gradlew :app:assembleRelease --no-daemon

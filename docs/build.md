# Build CLI

This repository uses Bazel and Bazelisk for SDK and plugin builds.

Install Bazelisk:

- Windows: `winget install -e --id Bazel.Bazelisk`
- Linux: install `bazelisk` from your package manager

Then just build and run cli with `bazelisk run //cli -- infer Qwen/Qwen3-0.6B-GGUF`, all dependencies will be automatically downloaded and built by Bazel.

> [!IMPORTANT]
> Before running CLI with local SDK linkage, you must build and install the bridge first.
> Bazel local mode expects `sdk/pkg-geniex/lib/geniex.dll` (Windows) or `sdk/pkg-geniex/lib/libgeniex.so` (Linux) to already exist.

## Build Flags

There are also some optional flags for `bazelisk run`:

- `--//sdk:sdk_type=s3` WIP
- `--//sdk:sdk_type=local` default behavior, force local build of sdk instead of using prebuilt binaries, you should manually build the sdk first, see [Build SDK](#build-sdk) section below
- `--//sdk:sdk_type=bazel` WIP

## Tips

### Windows Symlink Requirements

Bazel requires symlink support on Windows. To enable this:

1. **Enable Developer Mode**: Settings → Privacy & Security → For developers → Developer Mode (toggle on)
2. **Grant Create Symlink Permission**:
   - Open Group Policy Editor: `gpedit.msc`
   - Navigate to: Computer Configuration → Windows Settings → Security Settings → Local Policies → User Rights Assignment
   - Find "Create symbolic links" and add your user account
   - Alternatively, set registry key: `HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\System` → `LocalAccountTokenFilterPolicy` = 1 (DWORD)
3. **Enable Long Paths**: Settings → Privacy & Security → For developers → Long paths (toggle on)
4. If you still encounter symlink issues, comment out `startup --windows_enable_symlinks` in `.bazelrc`, but be aware this may cause other issues due to how the SDK is structured.

### Running the CLI

If you want to manually run the generated executable, you can find it in `bazel-bin/cli/cmd/geniex/geniex_/` and runtime files in `bazel-bin/cli/cmd/geniex/geniex_/geniex.runfiles/_main`.

# Geniex SDK

## Build Bridge/Plugin First (Required for local SDK)

Build and install the SDK bridge and plugins into `sdk/pkg-geniex` first, then run CLI.

Use the SDK subproject instructions in `sdk/README.md` and the platform-specific steps in the [Build & Install](#build--install) section below.

Change `CMakeLists.txt:86` in `tokenizer-cpp`:

```cmake
elseif (CMAKE_SYSTEM_NAME STREQUAL "Windows")
  if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(TOKENIZERS_CPP_CARGO_TARGET aarch64-pc-windows-msvc)
  else()
    set(TOKENIZERS_CPP_CARGO_TARGET x86_64-pc-windows-msvc)
  endif()
endif()
```

## Build & Install

### Linux

```bash
cd sdk
cmake -S . -B build
cmake --build build -j
cmake --install build --prefix pkg-geniex
```

### Windows ARM64 (Snapdragon)

> [!NOTE]
> The Hexagon toolchain has a 250-character path limit. Use `subst` to shorten the source path before building:
>
> ```powershell
> subst G: C:\path\to\geniex
> cd G:\sdk
> ```

```powershell
cd sdk
cmake --preset arm64-windows-snapdragon-release -DGENIEX_TEST=OFF
cmake --build --preset arm64-windows-snapdragon-release -j 8
cmake --install build-arm64-windows-snapdragon-release --prefix pkg-geniex
```

### Android (cross-compilation from Linux)

```bash
docker run --rm -u $(id -u):$(id -g) \
    --volume $(pwd):/workspace \
    --platform linux/amd64 \
    ghcr.io/snapdragon-toolchain/arm64-android:v0.3

apt update && apt install -y make

cmake -G "Unix Makefiles" -B build-arm64-android-snapdragon-debug -S . -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
      -DANDROID_ABI=arm64-v8a \
      -DANDROID_PLATFORM=android-23 \
      -DANDROID_STL=c++_static \
      -DCMAKE_VERBOSE_MAKEFILE=ON \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_CXX_FLAGS="-Wno-error=unused-function -Wno-error=unused-local-typedef -Wno-error=for-loop-analysis -Wno-error" \
      -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined" \
      -DHEXAGON_SDK_ROOT="/opt/hexagon/6.4.0.2" \
      -DGENIEX_DEBUG=ON \
      -DGENIEX_PLUGIN_LLAMA_CPP=ON \
            -DGGML_OPENCL=ON \
            -DGGML_HEXAGON=on -DPREBUILT_LIB_DIR=android_aarch64

cmake --build build-arm64-android-snapdragon-debug -j 8
cmake --install build-arm64-android-snapdragon-debug --prefix pkg-geniex

adb push pkg-geniex /data/local/tmp/geniex
adb push Qwen3-0.6B-Q4_0.gguf /data/local/tmp/geniex/modelfiles/llama_cpp/
adb shell "cd /data/local/tmp/geniex && LD_LIBRARY_PATH=./lib:./lib/llama_cpp GENIEX_PLUGIN_PATH=./lib ./bin/geniex_test_llm"
```

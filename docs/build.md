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

# Model Prepare

Currently we must manually prepare the modelfiles.

```
C:\USERS\REMIL\.CACHE
└───geniex
    │   update_check
    │
    └───models
        ├───nexa4ai
        │   │   geniex-qairt-models.lock
        │   │   granite4_micro.lock
        │   │
        │   └───granite4_micro
        │       │   embed_tokens.npy
        │       │   geniex.manifest
        │       │   htp_backend_ext_config.json
        │       │   tokenizer.json
        │       │   weight_sharing_model_1_of_2.serialized.bin
        │       │   weight_sharing_model_2_of_2.serialized.bin
        │       │
        │       └───htp-files
        │               calculator.dll
        │               calculator_htp.dll
        │               libCalculator_skel.so
        │               libqnnhtpv73.cat
        │               libQnnHtpV73.so
        │               libQnnHtpV73QemuDriver.so
        │               libQnnHtpV73Skel.so
        │               libQnnSaver.so
        │               libQnnSystem.so
        │               libsnpehtpv73.cat
        │               libSnpeHtpV73Skel.so
        │               PlatformValidatorShared.dll
        │               QnnChrometraceProfilingReader.dll
        │               QnnChrometraceProfilingReader.lib
        │               QnnHtp.dll
        │               QnnHtp.lib
        │               QnnHtpNetRunExtensions.dll
        │               QnnHtpOptraceProfilingReader.dll
        │               QnnHtpPrepare.dll
        │               QnnHtpProfilingReader.dll
        │               QnnHtpV73CalculatorStub.dll
        │               QnnHtpV73Stub.dll
        │               QnnHtpV81CalculatorStub.dll
        │               QnnHtpV81Stub.dll
        │               QnnIr.dll
        │               QnnIr.lib
        │               QnnJsonProfilingReader.dll
        │               QnnModelDlc.dll
        │               QnnModelDlc.lib
        │               QnnNetRunDirectV81Stub.dll
        │               QnnSaver.dll
        │               QnnSystem.dll
        │
        └───Qwen
            │   Qwen3-0.6B-GGUF.lock
            │
            └───Qwen3-0.6B-GGUF
                    geniex.manifest
                    Qwen3-0.6B-Q8_0.gguf
```

file content of `geniex.manifest`:

```json
{"Name":"nexa4ai/granite4_micro","ModelName":"granite4","ModelType":"llm","PluginId":"qairt","DeviceId":"","MinSDKVersion":"","ModelFile":{"N/A":{"Name":"embed_tokens.npy","Downloaded":true,"Size":1}},"MMProjFile":{"Name":"","Downloaded":false,"Size":0},"TokenizerFile":{"Name":"","Downloaded":false,"Size":0},"ExtraFiles":null}
```

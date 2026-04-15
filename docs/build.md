# Build CLI

This repository uses Bazel and Bazelisk for SDK and plugin builds.

Install Bazelisk:

- Windows: `winget install -e --id Bazel.Bazelisk`
- Linux: install `bazelisk` from your package manager

Then just build and run cli with `bazelisk run //cli -- infer Qwen/Qwen3-0.6B-GGUF`, all dependencies will be automatically downloaded and built by Bazel.

There are also some optional flags for `bazelisk run`:

- `--//sdk:sdk_type=s3` default behavior, WIP
- `--//sdk:sdk_type=local` to force local build of sdk instead of using prebuilt binaries, you should manually build the sdk first, see [Build SDK](#build-sdk) section below
- `--//sdk:sdk_type=bazel` WIP

# Geniex SDK

## Build & Install

### Linux

```bash
cd sdk
cmake -S . -B build
cmake --build build -j
cmake --install build --prefix pkg-geniex
```

### Windows ARM64 (Snapdragon)

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

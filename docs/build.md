# Build

This repository uses Bazel and Bazelisk for SDK and plugin builds.

## Prerequisites

Install Bazelisk:

- Windows: `winget install -e --id Bazel.Bazelisk`
- Linux: install `bazelisk` from your package manager, plus `cmake` and `ninja`

Windows ARM64 builds also require:

- Visual Studio Build Tools 2022 or newer
- LLVM toolchain bundled with Visual Studio Build Tools
- Windows SDK
- Python 3

## Bazel Configs

- `sdk_debug`: enables `GENIEX_DEBUG`
- `llama_cpp_hexagon`: enables `GGML_HEXAGON` (Windows only)
- `llama_cpp_no_hexagon`: disables `GGML_HEXAGON`

An optional local override file `.bazelrc.user` is imported automatically for machine-specific environment forwarding.

## Common Builds

### Linux

Build the SDK shared library:

```bash
bazelisk build //sdk/src:geniex
```

Build the llama.cpp plugin (CPU, no OpenCL/Hexagon):

```bash
bazelisk build //sdk/plugins/llama_cpp:geniex_plugin
```

### Windows ARM64

Use the local wrapper script for day-to-day development:

```powershell
.\scripts\build_llama_cpp_local.ps1
```

With Hexagon enabled:

```powershell
.\scripts\build_llama_cpp_local.ps1 -Hexagon
```

Or invoke Bazel directly after setting the required environment variables:

```powershell
$env:BAZEL_VS='C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools'
$env:BAZEL_VC='C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC'
bazelisk build --config=llama_cpp_no_hexagon //sdk/plugins/llama_cpp:geniex_plugin
```

Hexagon build additionally requires:

```powershell
$env:HEXAGON_HTP_CERT='<path-to-pfx>'
$env:WINDOWS_SDK_BIN='C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0'
$env:PYTHON3_EXECUTABLE='<path-to-python.exe>'
bazelisk build --config=llama_cpp_hexagon //sdk/plugins/llama_cpp:geniex_plugin
```

## Outputs

| Platform | Target | Output |
|---|---|---|
| Linux | `//sdk/src:geniex` | `bazel-bin/sdk/src/libgeniex.so` |
| Linux | `//sdk/plugins/llama_cpp:geniex_plugin` | `bazel-bin/sdk/plugins/llama_cpp/libgeniex_plugin.so` |
| Windows ARM64 | `//sdk/src:geniex` | `bazel-out/arm64_windows-fastbuild/bin/sdk/src/geniex.dll` |
| Windows ARM64 | `//sdk/plugins/llama_cpp:geniex_plugin` | `bazel-out/arm64_windows-fastbuild/bin/sdk/plugins/llama_cpp/geniex_plugin.dll` |

## Notes

- CMake and Ninja are invoked by the llama.cpp Bazel genrule; they are not used for the SDK itself
- Linux builds default to the host toolchain (CPU-only). OpenCL and Hexagon backends are Windows-only at this time
- `llama_cpp_no_hexagon` is the recommended default for CPU/OpenCL validation on Windows

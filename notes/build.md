# Build

The CLI is built with [Bazelisk](https://github.com/bazelbuild/bazelisk) (Bazel launcher); the SDK bridge and native plugins are built with CMake. Bazel fetches all CLI-side dependencies automatically.

Install Bazelisk:

- Windows: `winget install --id Bazel.Bazelisk`
- Linux: install `bazelisk` from your package manager

Quick smoke test:

```bash
bazelisk run //cli -- infer Qwen/Qwen3-0.6B-GGUF
```

> `//cli` is a convenience alias for `//cli/cmd/geniex:geniex`. Both are used interchangeably in these docs.

> [!IMPORTANT]
> When running the CLI in local-SDK mode (the default), build and install the SDK bridge into `sdk/pkg-geniex/` **first** — Bazel expects `sdk/pkg-geniex/lib/geniex.dll` (Windows) or `sdk/pkg-geniex/lib/libgeniex.so` (Linux) to already exist. See [Build the SDK](#build-the-sdk) below.

## CLI build options

### Flags

Flags for `bazelisk build` and `bazelisk run`:

| Flag                                    | Meaning                                                                                                     |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `--//sdk:sdk_type={local,s3,bazel}`     | SDK source. `local` (default) links against `sdk/pkg-geniex`; `s3` and `bazel` are WIP.                     |
| `--config={linux_arm64,windows_arm64}`  | Cross-compile to the target platform (Go toolchain + CGO + `oci_image` base). `sdk/pkg-geniex/` must match. |

### Development and release targets

Development targets:

- `bazelisk run //cli/release/linux:docker` — build and load the Docker image for the Linux release.

Package release artifacts:

| Target                                 | Output                                               |
| -------------------------------------- | ---------------------------------------------------- |
| `bazelisk build //cli:artifact`        | `bazel-bin/cli/artifact.zip`                         |
| `bazelisk build //cli/release/windows` | `bazel-bin/cli/release/windows/geniex-cli-setup.exe` |
| `bazelisk build //cli/release/linux`   | `bazel-bin/cli/release/linux/geniex-cli-docker.tar`  |

Generated executable (for manual invocation): `bazel-bin/cli/cmd/geniex/geniex_/`, with runtime files under `geniex.runfiles/_main`.

### Test coverage

Use `bazelisk coverage` instead of `bazelisk test`:

```bash
bazelisk coverage //... --combined_report=lcov
```

Bazel's default `--instrumentation_filter` only covers packages that own a test target, so every Go package ships at least a placeholder `package_test.go` to keep itself in the denominator. Combined lcov lands at `bazel-out/_coverage/_coverage_report.dat`. Render it with `genhtml bazel-out/_coverage/_coverage_report.dat -o coverage-html` (gitignored), or summarize on the CLI with `lcov --list <report>`.

## Python bindings

See [bindings/python/README.md](../bindings/python/README.md).

## Windows prerequisites

### Symlink support (Bazel)

1. Enable **Developer Mode**: Settings → Privacy & Security → For developers.
2. Grant **Create symbolic links** rights via `gpedit.msc` → Computer Configuration → Windows Settings → Security Settings → Local Policies → User Rights Assignment, or set `HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\System\LocalAccountTokenFilterPolicy = 1` (DWORD).
3. Enable **Long paths**: Settings → Privacy & Security → For developers.
4. If symlink errors persist, comment out `startup --windows_enable_symlinks` in `.bazelrc` — but be aware this can break other SDK paths.

### Native SDKs (for full Snapdragon build)

The `arm64-windows-snapdragon-release` preset requires:

- **Hexagon SDK** — `HEXAGON_SDK_ROOT`, `HEXAGON_TOOLS_ROOT`
- **OpenCL SDK** — `OPENCL_SDK_ROOT` (headers + `OpenCL.lib`; runtime ICD ships with the Snapdragon GPU driver)
- **Windows Driver Kit** — provides `inf2cat.exe`
- **Self-signed HTP cert** (`.pfx`) and Windows test-signing enabled — see [run.md § Self-signed fallback](run.md#self-signed-fallback) for cert generation and test-signing setup

## Build the SDK

### Windows ARM64 (Snapdragon)

> [!NOTE]
> The Hexagon toolchain has a 250-character path limit. Shorten the source path with `subst` before building:
>
> ```powershell
> subst G: C:\path\to\geniex
> cd G:\sdk
> ```

```powershell
cd sdk
cmake --preset arm64-windows-snapdragon-release -B build
cmake --build build -j
cmake --install build --prefix pkg-geniex
```

### Linux (cross-compile from x86_64)

Build the SDK inside the derived Snapdragon Linux toolchain container — it extends [ghcr.io/snapdragon-toolchain/arm64-linux](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/linux.md#snapdragon-based-linux-devices) with `build-essential`, `ccache`, `rustup`, the `aarch64-unknown-linux-gnu` Rust target, and the cc-rs cross-compiler symlinks baked in (see [`.github/docker/toolchain-linux.Dockerfile`](../.github/docker/toolchain-linux.Dockerfile)). Run from the repo root.

One-shot:

```bash
docker run --rm -u $(id -u):$(id -g) \
    --volume $(pwd):/workspace \
    --workdir /workspace/sdk \
    -e CCACHE_DIR=/workspace/.ccache \
    --platform linux/amd64 \
    ghcr.io/qcom-ai-hub/geniex-toolchain-linux:v0.0.2 \
    bash -c 'cmake --preset arm64-linux-snapdragon-debug -B build-linux . \
      && cmake --build build-linux -j \
      && cmake --install build-linux --prefix pkg-geniex'
```

Interactive:

```bash
docker run --rm -it -u $(id -u):$(id -g) \
    --volume $(pwd):/workspace \
    --workdir /workspace/sdk \
    -e CCACHE_DIR=/workspace/.ccache \
    --platform linux/amd64 \
    ghcr.io/qcom-ai-hub/geniex-toolchain-linux:v0.0.2 bash
# then, inside the container:
cmake --preset arm64-linux-snapdragon-debug -B build-linux .
cmake --build build-linux -j
cmake --install build-linux --prefix pkg-geniex
```

### Android (cross-compile from Linux)

Build the SDK inside the derived Snapdragon Android toolchain container — it extends [ghcr.io/snapdragon-toolchain/arm64-android](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/README.md#android) with `build-essential`, `ccache`, `rustup`, and the `aarch64-linux-android` Rust target baked in (see [`.github/docker/toolchain-android.Dockerfile`](../.github/docker/toolchain-android.Dockerfile)). Run from the repo root.

One-shot:

```bash
docker run --rm -u $(id -u):$(id -g) \
    --volume $(pwd):/workspace \
    --workdir /workspace/sdk \
    -e CCACHE_DIR=/workspace/.ccache \
    --platform linux/amd64 \
    ghcr.io/qcom-ai-hub/geniex-toolchain-android:v0.0.1 \
    bash -c 'cmake --preset arm64-android-snapdragon-debug -B build-android . \
      && cmake --build build-android -j \
      && cmake --install build-android --prefix pkg-geniex'
```

Interactive:

```bash
docker run --rm -it -u $(id -u):$(id -g) \
    --volume $(pwd):/workspace \
    --workdir /workspace/sdk \
    -e CCACHE_DIR=/workspace/.ccache \
    --platform linux/amd64 \
    ghcr.io/qcom-ai-hub/geniex-toolchain-android:v0.0.1 bash
# then, inside the container:
cmake --preset arm64-android-snapdragon-debug -B build-android .
cmake --build build-android -j
cmake --install build-android --prefix pkg-geniex
```

Deploy and smoke-test on device:

```bash
adb push pkg-geniex /data/local/tmp/geniex
adb push Qwen3-0.6B-Q4_0.gguf /data/local/tmp/geniex/modelfiles/llama_cpp/
adb shell "cd /data/local/tmp/geniex && \
  LD_LIBRARY_PATH=./lib:./lib/llama_cpp \
  GENIEX_PLUGIN_PATH=./lib \
  ./bin/geniex_test_llm"
```

Build the Android app (requires Android SDK + Gradle on the host — installing them inside the container is possible but significantly more complex):

```bash
cd examples/android
gradle build
```

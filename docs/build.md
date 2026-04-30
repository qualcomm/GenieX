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

Flags for `bazelisk run`:

| Flag                          | Meaning                                                                 |
|-------------------------------|-------------------------------------------------------------------------|
| `--//sdk:sdk_type=local`      | Default. Link against a locally built SDK in `sdk/pkg-geniex`.          |
| `--//sdk:sdk_type=s3`         | WIP.                                                                    |
| `--//sdk:sdk_type=bazel`      | WIP.                                                                    |

Development targets:

- `bazelisk run //cli:gen` — generate development files (protobuf Go bindings).
- `bazelisk run //cli:clean` — clean generated files.
- `bazelisk run //cli/release/linux:docker` — build and load the Docker image for the Linux release.

Package release artifacts:

| Target                                | Output                                             |
|---------------------------------------|----------------------------------------------------|
| `bazelisk build //cli:artifact`       | `bazel-bin/cli/artifact.zip`                       |
| `bazelisk build //cli/release/windows`| `bazel-bin/cli/release/windows/geniex-cli-setup.exe` |
| `bazelisk build //cli/release/linux`  | `bazel-bin/cli/release/linux/geniex-cli-docker.tar` |

Generated executable (for manual invocation): `bazel-bin/cli/cmd/geniex/geniex_/`, with runtime files under `geniex.runfiles/_main`.

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

For a minimal NPU build that skips Hexagon/OpenCL and drives the NPU only through QAIRT, use the `arm64-windows-snapdragon-cpu-release` preset instead.

## Build the SDK

### Linux

```bash
cd sdk
cmake -S . -B build
cmake --build build -j
cmake --install build --prefix pkg-geniex
```

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
cmake --preset arm64-windows-snapdragon-release -DGENIEX_TEST=OFF
cmake --build --preset arm64-windows-snapdragon-release -j 8
cmake --install build-arm64-windows-snapdragon-release --prefix pkg-geniex
```

Swap the preset to `arm64-windows-snapdragon-cpu-release` for the QAIRT-only minimal build described above.

### Android (cross-compile from Linux)

Start the Snapdragon Android toolchain container — follow [llama.cpp's instructions](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/README.md#android) to launch:

```bash
docker run --rm -u $(id -u):$(id -g) \
    --volume $(pwd):/workspace \
    --platform linux/amd64 \
    ghcr.io/snapdragon-toolchain/arm64-android:v0.3
```

Install build tools and Rust inside the container:

```bash
# Root shell (docker exec -u 0 -it <container_id> bash):
apt update -y && apt install -y make gcc
# Exit back to the normal user shell, then:
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
rustup target add aarch64-linux-android
```

Build the SDK with the `arm64-android-snapdragon-debug` preset:

```bash
cd sdk
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

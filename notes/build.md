# Build

The CLI is built with [Bazelisk](https://github.com/bazelbuild/bazelisk) (Bazel launcher); the SDK bridge and native plugins are built with CMake. Bazel fetches all CLI-side dependencies automatically.

Install Bazelisk:

- Windows: `winget install --id Bazel.Bazelisk`
- Linux: install `bazelisk` from your package manager

> [!IMPORTANT]
> Build in this order. In local-SDK mode (the default) the CLI links against the SDK bridge at `sdk/pkg-geniex/` — Bazel expects `sdk/pkg-geniex/lib/geniex.dll` (Windows) or `sdk/pkg-geniex/lib/libgeniex.so` (Linux) to already exist. So build and install the SDK ([Build the SDK](#build-the-sdk)) **first**, then build the CLI ([Build and run the CLI](#build-and-run-the-cli)).

## Windows prerequisites

Run every command below from **PowerShell**, not `cmd.exe`, Git Bash, or WSL. The `cmd /c '"...\VsDevCmd.bat" ... && ...'` commands below rely on PowerShell quoting (outer single quotes, inner double quotes).

### Symlink support (Bazel and CMake)

Also needed by the SDK's CMake configure: a vendored submodule (sentencepiece, under `third-party/geniex-qairt`) creates a symlink and fails with "A required privilege is not held by the client" without it.

1. Enable **Developer Mode**: Settings → Privacy & Security → For developers.
2. Grant **Create symbolic links** rights via `gpedit.msc` → Computer Configuration → Windows Settings → Security Settings → Local Policies → User Rights Assignment.
3. Enable **Long paths**: Settings → Privacy & Security → For developers.
4. **Sign out and back in (or reboot)** after enabling Developer Mode; open sessions keep the old token. Verify in a new terminal with `whoami /priv | findstr SymbolicLink` (no output means it hasn't taken effect).
5. If symlink errors persist, comment out `startup --windows_enable_symlinks` in `.bazelrc` — but be aware this can break other SDK paths.

**Corporate/domain-joined machines:** Group Policy can block step 4 even after a reboot. In that case, run the build (`cmake --preset ...` and `bazelisk ...`) from an **elevated** terminal ("Run as Administrator"); local Administrators get `SeCreateSymbolicLinkPrivilege` once elevated. `whoami /groups` showing `BUILTIN\Administrators` as "Group used for deny only" confirms you're an admin filtered by UAC.

### Toolchain (SDK build)

The SDK's Rust model manager (`sdk/model-manager`) is built by `cargo` from CMake, and its build scripts compile C code, so a native Windows ARM64 build also needs:

- **clang** — used by `cc-rs` and the Snapdragon presets. Install the ARM64 MSYS2 build (the plain LLVM installer has no `aarch64`-hosted `clang.exe`):

  ```powershell
  winget install --id MSYS2.MSYS2
  C:\msys64\usr\bin\pacman.exe -S --noconfirm mingw-w64-clang-aarch64-clang mingw-w64-clang-aarch64-lld
  ```

  Then add `C:\msys64\clangarm64\bin` to `PATH` (ahead of any other `clang.exe`, e.g. one bundled with Visual Studio).

- **cargo** and the Windows ARM64 target:

  ```powershell
  winget install --id Rustlang.Rustup
  rustup target add aarch64-pc-windows-msvc
  ```

- **Visual Studio** — for the MSVC libs/linker only. Install the **"Desktop development with C++"** workload with the **ARM64 build tools** component.

- **Python 3** — the build runs it to embed the OpenCL kernels and to post-process each HTP skel (`htp-swiv.py`). If more than one is installed, choose with `-DPython3_EXECUTABLE=<path>` at configure.

`clang` targets the MSVC ABI, so it needs MSVC's `INCLUDE`/`LIB` and `lld-link` on `PATH`. Without them, CMake's compiler check fails with `lld-link: error: could not open 'oldnames.lib'`. Configure and build from a Visual Studio **Developer** environment for ARM64 (see [Build the SDK](#windows-arm64-snapdragon)).

**Alternative: Visual Studio's own clang, which CI uses.** Instead of MSYS2, add Visual Studio's **C++ Clang Compiler for Windows** component (`Microsoft.VisualStudio.Component.VC.Llvm.Clang`), which includes native ARM64 `clang` and `lld-link` in `VC\Tools\Llvm\ARM64\bin`, and build as in [the alternative below](#windows-arm64-snapdragon).

### Native SDKs (for full Snapdragon build)

The `arm64-windows-snapdragon-release` preset requires:

- **Hexagon SDK** — `HEXAGON_SDK_ROOT`, `HEXAGON_TOOLS_ROOT`
- **OpenCL SDK** — `OPENCL_SDK_ROOT` (headers + `OpenCL.lib`; runtime ICD ships with the Snapdragon GPU driver)
- **Windows Driver Kit** — provides `inf2cat.exe`
- **Self-signed HTP cert** (`.pfx`) and Windows test-signing enabled — see [run.md § Self-signed fallback](run.md#self-signed-fallback) for cert generation and test-signing setup

CI installs trimmed Windows on Snapdragon builds of both SDKs, [Hexagon SDK 6.7.0.0](https://github.com/snapdragon-toolchain/hexagon-sdk/releases/tag/v6.7.0.0) and [Adreno OpenCL SDK 2.3.2](https://github.com/snapdragon-toolchain/opencl-sdk/releases/tag/v2.3.2): it extracts each without its top-level folder and sets these variables, with `HEXAGON_TOOLS_ROOT` at `tools\HEXAGON_Tools\21.0.04` (see [`setup-snapdragon-sdks`](../.github/actions/setup-snapdragon-sdks/action.yml)). The full SDKs from Qualcomm Software Center need extra steps:

- **Hexagon SDK:** building on Windows needs its Windows on Snapdragon add-on. Without it, configure stops where the SDK's `build/cmake/hexagon_fun.cmake` includes `addons/wos/build/cmake/windows_fun.cmake`. CMake also runs the SDK's IDL compiler from `ipc\fastrpc\qaic\bin\`, but the SDK ships it in `ipc\fastrpc\qaic\WinNT\`: copy `WinNT\*` into `bin\`, as that folder's `Makefile` does on Windows.
- **Adreno OpenCL SDK:** CMake's `FindOpenCL` doesn't search its `inc\` and `libs\Windows\` folders, so pass `-DOpenCL_INCLUDE_DIR=<sdk>/inc -DOpenCL_LIBRARY=<sdk>/libs/Windows/OpenCL.lib` at configure.

For the WDK and the cert, see [Signing the HTP skels](#signing-the-htp-skels).

## Build the SDK

With a Snapdragon preset, on any platform:

- **Configure patches llama.cpp.** It applies `sdk/patches/*.patch` to `third-party/llama.cpp`, checking each with `git apply --reverse --check` first, so reconfiguring is safe and `git status` shows the submodule as modified. When a new commit moves the llama.cpp pin, `git submodule update --force` discards the patches, and the next configure applies them again; if a patch no longer applies, configure stops and asks for it to be refreshed against the new pin.
- **Every build rebuilds the HTP skels from scratch.** `sdk/plugins/llama_cpp/CMakeLists.txt` deletes their ExternalProject folders once the skels are copied out, so that `Inf2Cat`'s scan sees no stray files.
- **CMake caches what it finds,** such as the OpenCL library and the signing tools, so after switching SDKs, reconfigure with `--fresh`.
- **`-DGENIEX_VERSION=<version>`** sets the SDK's version (default `v0.0.0`). CI passes the release tag, or `sha-` and the 7-character commit for untagged builds, and gives the CLI the same one (see [Flags](#flags)).

### Windows ARM64 (Snapdragon)

> [!NOTE]
> The Hexagon toolchain has a 250-character path limit. Shorten the source path with `subst` before building:
>
> ```powershell
> subst G: C:\path\to\geniex
> cd G:\sdk
> ```

Run from a "Developer PowerShell for VS" (ARM64), elevated if the symlink privilege won't take effect (see [Symlink support](#symlink-support-bazel-and-cmake)):

```powershell
cd sdk
cmake --preset arm64-windows-snapdragon-release -B build
cmake --build build -j
cmake --install build --prefix pkg-geniex
```

Or from a plain PowerShell, chaining through `VsDevCmd.bat` (adjust the path/edition for your install):

```powershell
cmd /c '"C:\Program Files\Microsoft Visual Studio\<edition>\Common7\Tools\VsDevCmd.bat" -arch=arm64 -host_arch=x64 && cmake --preset arm64-windows-snapdragon-release -B build && cmake --build build -j && cmake --install build --prefix pkg-geniex'
```

Or, with Visual Studio's own clang instead of MSYS2's (see [Toolchain](#toolchain-sdk-build)), as CI does ([`setup-vcvars`](../.github/actions/setup-vcvars/action.yml), [`_build-sdk.yml`](../.github/workflows/_build-sdk.yml)): from a plain PowerShell, elevated if needed, load Visual Studio's native ARM64 environment into the session, put its ARM64 `clang` first on `PATH`, and build. For a signed build, set `$winsdk` to the Windows SDK version that has your WDK's `Inf2Cat.exe` (see [Signing the HTP skels](#signing-the-htp-skels)); otherwise leave it empty, and vcvars picks the newest SDK:

```powershell
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * `
  -requires Microsoft.VisualStudio.Component.VC.Tools.ARM64 -property installationPath
$winsdk = "10.0.26100.0"   # a folder under C:\Program Files (x86)\Windows Kits\10\bin with x86\Inf2Cat.exe
cmd /c "`"$vs\VC\Auxiliary\Build\vcvarsarm64.bat`" $winsdk && set" | ForEach-Object {
  if ($_ -match '^(.+?)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') }
}
$env:PATH = "$vs\VC\Tools\Llvm\ARM64\bin;$env:PATH"
cd sdk
cmake --preset arm64-windows-snapdragon-release -B build
cmake --build build -j
cmake --install build --prefix pkg-geniex
```

#### Signing the HTP skels

On Windows, llama.cpp's NPU backend can load its HTP skels only when they're listed in a catalog, `libggml-htp.cat`, signed with a trusted certificate (llama.cpp's [Windows guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/windows.md)). While `HEXAGON_HTP_CERT` is set, the build makes and signs that catalog, and configure stops if it can't find the WDK's `Inf2Cat.exe` and the Windows SDK's `signtool.exe`:

- Install the [Windows Driver Kit](https://learn.microsoft.com/windows-hardware/drivers/download-the-wdk) that matches a Windows SDK you have; CI installs WDK 10.0.26100. The WDK adds `Inf2Cat.exe` to that SDK's `bin\<version>\x86` folder.
- CMake looks for both tools on `PATH`, in the `bin` folder of the Windows SDK that the Visual Studio environment selected, and under `WINDOWS_SDK_BIN`. So select the SDK your WDK matches (`VsDevCmd.bat -winsdk=<version>`, or `vcvarsarm64.bat <version>`), or set `WINDOWS_SDK_BIN` to its `bin\<version>` folder, as llama.cpp's guide does. CMake appends `arm64` and `x86` to `WINDOWS_SDK_BIN`, so leave both off.
- `GGML_HEXAGON_HTP_CERT` is cached, and takes `HEXAGON_HTP_CERT` only as its default, so to change it in an existing build tree, pass it at configure.

To build without signing, and without the WDK, configure with `-DGGML_HEXAGON_HTP_CERT=` (empty), as CI does for fork PRs. The build then makes no catalog, so llama.cpp can't use the NPU. To switch an existing setup to unsigned, use a new build directory and an empty install prefix: an earlier signed build's `libggml-htp.cat` survives `--fresh` in the build tree, which still installs it, and `cmake --install` never removes it from `pkg-geniex`.

### Linux (cross-compile from x86_64)

Build the SDK inside the derived Snapdragon Linux toolchain container — it extends [ghcr.io/snapdragon-toolchain/arm64-linux](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/snapdragon/linux.md#snapdragon-based-linux-devices) with `build-essential`, `ccache`, `rustup`, the `aarch64-unknown-linux-gnu` Rust target, and the cc-rs cross-compiler symlinks baked in (see [`.github/docker/toolchain-linux.Dockerfile`](../.github/docker/toolchain-linux.Dockerfile)). Run from the repo root.

One-shot:

```bash
docker run --rm -u $(id -u):$(id -g) \
    --volume $(pwd):/workspace \
    --workdir /workspace/sdk \
    -e CCACHE_DIR=/workspace/.ccache \
    --platform linux/amd64 \
    docker.io/qualcomm/geniex-toolchain-linux:v0.2.0 \
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
    docker.io/qualcomm/geniex-toolchain-linux:v0.2.0 bash
# then, inside the container:
cmake --preset arm64-linux-snapdragon-debug -B build-linux .
cmake --build build-linux -j
cmake --install build-linux --prefix pkg-geniex
```

#### CPU-only variant (baseline armv8.0-a)

NPU-less Dragonwing IoT boards (unoq) are baseline ARMv8.0 and trap on the LSE
atomics the `snapdragon` presets inline (see
[#1217](https://github.com/qualcomm/GenieX/issues/1217)). Swap the preset for
`arm64-linux-cpu-{debug,release}` — same container, plain `-march=armv8-a`, and
CPU-only (no QAIRT, Hexagon, or OpenCL):

```bash
cmake --preset arm64-linux-cpu-debug -B build-linux-cpu .
cmake --build build-linux-cpu -j
cmake --install build-linux-cpu --prefix pkg-geniex
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
    docker.io/qualcomm/geniex-toolchain-android:v0.2.0 \
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
    docker.io/qualcomm/geniex-toolchain-android:v0.2.0 bash
# then, inside the container:
cmake --preset arm64-android-snapdragon-debug -B build-android .
cmake --build build-android -j
cmake --install build-android --prefix pkg-geniex
```

#### CPU-only variant (baseline armv8.0-a)

`minSdk` is 27, so the AAR still targets phones that predate the armv8.7 ISA the
`snapdragon` presets bake in and trap at startup (see
[#1217](https://github.com/qualcomm/GenieX/issues/1217)). Swap the preset for
`arm64-android-cpu-{debug,release}` — same container, plain `-march=armv8-a`, and
CPU-only (no QAIRT, Hexagon, or OpenCL):

```bash
cmake --preset arm64-android-cpu-debug -B build-android-cpu .
cmake --build build-android-cpu -j
cmake --install build-android-cpu --prefix pkg-geniex
```

`bindings/android` needs no changes: `assembleRelease` packages whatever
`sdk/pkg-geniex/lib/` holds, so the same gradle project yields the CPU-only AAR
that CI publishes as `geniex-android-aar-cpu-<tag>.aar`.

Deploy and smoke-test on device:

```bash
adb push pkg-geniex /data/local/tmp/geniex
adb push Qwen3-0.6B-Q4_0.gguf /data/local/tmp/geniex/
adb shell "cd /data/local/tmp/geniex && \
  LD_LIBRARY_PATH=./lib:./lib/llama_cpp \
  ADSP_LIBRARY_PATH=./lib/llama_cpp \
  GENIEX_PLUGIN_PATH=./lib \
  ./bin/geniex-bench --plugin llama_cpp --device npu -m Qwen3-0.6B-Q4_0.gguf"
```

The Android demo app is no longer hosted in this repo — it lives in [`qualcomm/ai-hub-apps`](https://github.com/qualcomm/ai-hub-apps/tree/main/apps/geniex_chat_android). Build the AAR here, then point the demo app at it.

### Building against a different QAIRT SDK's headers

The `qairt` plugin currently compiles against QAIRT 2.51.0.260929 QNN C API 2.40 headers vendored in `third-party/geniex-qairt/qnn-api/include/`; its independent runtime floor remains QNN C API 2.27 (`kMinApiMinor = 27`). Provider-query checks do not establish end-to-end compatibility with every runtime version. To use another set (e.g. an internal QAIRT checkout), pass `-DQAIRT_QNN_HEADERS=...` to any preset above:

```powershell
cmake --preset arm64-windows-snapdragon-release -B build -DQAIRT_QNN_HEADERS=C:\path\to\qairt\include\QNN
```

See the [plugin README](https://github.com/qualcomm/geniex-qairt-plugin#using-a-different-qairt-runtime) for the expected directory shape and caveats.

## Build and run the CLI

With the SDK built and installed into `sdk/pkg-geniex/`, build and run the CLI. Quick smoke test:

```bash
bazelisk run //cli -- infer Qwen/Qwen3-0.6B-GGUF
```

> `//cli` is a convenience alias for `//cli/cmd/geniex:geniex`. Both are used interchangeably in these docs.

On Windows, the CLI build needs no Visual Studio environment: Bazel brings its own Go SDK and a hermetic Zig C toolchain (`hermetic_cc_toolchain`).

### Flags

Flags for `bazelisk build` and `bazelisk run`:

| Flag                                    | Meaning                                                                                                     |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `--//sdk:sdk_type={local,s3,bazel}`     | SDK source. Only `local` (default, links against `sdk/pkg-geniex`) works; `s3` and `bazel` are not wired up yet.                     |
| `--config={linux_arm64,windows_arm64}`  | Cross-compile to the target platform (Go toolchain + CGO + `oci_image` base). `sdk/pkg-geniex/` must match. |
| `--define=VERSION=<version>`            | The version `geniex version` reports (default `v0.0.0-dev.0`, from `.bazelrc`). CI passes the release tag, or `sha-` and the 7-character commit for untagged builds, and gives the SDK the same one. |

### Development and release targets

Development targets:

- `bazelisk run //cli/release/linux:docker` — build and load the Docker image for the Linux release.

Package release artifacts:

| Target                                 | Output                                               |
| -------------------------------------- | ---------------------------------------------------- |
| `bazelisk build //cli:artifact`        | `bazel-bin/cli/artifact.zip`                         |
| `bazelisk build //cli/release/windows` | `bazel-bin/cli/release/windows/geniex-cli-setup.exe` |
| `bazelisk build //cli/release/linux`   | `bazel-bin/cli/release/linux/geniex-cli-docker.tar`  |

Generated executable (for manual invocation): rules_go builds `geniex` in a transitioned configuration, so it isn't under `bazel-bin` but in a `bazel-out/<config>-ST-<hash>/bin/cli/cmd/geniex/geniex_/` folder, with runtime files beside it under `geniex.runfiles/_main` (`geniex.exe.runfiles/_main` on Windows). Ask Bazel for the path, with the same flags as the build:

```bash
bazelisk cquery //cli/cmd/geniex:geniex --//sdk:sdk_type=local --output=files
```

### Test coverage

Use `bazelisk coverage` instead of `bazelisk test`:

```bash
bazelisk coverage //... --combined_report=lcov
```

Bazel's default `--instrumentation_filter` only covers packages that own a test target, so every Go package ships at least a placeholder `package_test.go` to keep itself in the denominator. Combined lcov lands at `bazel-out/_coverage/_coverage_report.dat`. Render it with `genhtml bazel-out/_coverage/_coverage_report.dat -o coverage-html` (gitignored), or summarize on the CLI with `lcov --list <report>`. lcov 2.x honours `LCOV_EXCL_*` markers with `--filter region --rc c_file_extensions=c,h,cc,cpp,go`; lcov 1.x (msys2, older distros) ignores them, so render on Linux with 2.x for a clean report.

## Python bindings

See [bindings/python/README.md](../bindings/python/README.md).

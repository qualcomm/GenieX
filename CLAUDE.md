# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Onboarding checklist (run first on a new machine)

Run these probes before attempting any build — missing tools cause confusing failures deep in the build, not clean "command not found" errors. Targets are Windows ARM64 (Snapdragon X) primarily; Linux/Android paths are CI-only.

```bash
# 1. Bazel
ls /c/Users/zhic/AppData/Local/Microsoft/WinGet/Packages/Bazel.Bazelisk_*/bazelisk.exe  # ← winget path; not on PATH by default

# 2. C/C++ toolchain (for sdk/ build)
"/c/Program Files/LLVM/bin/clang.exe" --version                            # arm64-pc-windows-msvc expected
ls "/c/Program Files/Microsoft Visual Studio/18/Community/Common7/Tools/VsDevCmd.bat"
cmake --version                                                            # >= 3.20

# 3. Rust (tokenizers-cpp dependency inside geniex-qairt)
ls /c/Users/zhic/.cargo/bin/cargo.exe

# 4. Submodules populated (llama.cpp + geniex-qairt)
git submodule status    # both should show a commit SHA, not "-"

# 5. Windows symlink support (Bazel execroot + sdk_local_bundle runfiles both need this)
powershell -NoProfile -Command "(Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\AppModelUnlock' -ErrorAction SilentlyContinue).AllowDevelopmentWithoutDevLicense"
# Expect: 1 — if empty, Developer Mode is OFF and Bazel will fail with
#   "createSymbolicLinkW failed (permission denied). Either Windows developer mode
#    or admin privileges are required."
# User must enable it manually: Settings → Privacy & Security → For developers → Developer Mode.
# If enabling Dev Mode is blocked by corp policy, alternatives are documented under
# "Platform gotchas → Windows symlinks" below.

# 6. (Optional) Full Snapdragon-release prerequisites — only if building the
#    llama.cpp NPU (Hexagon) + GPU (OpenCL) backends via the
#    `arm64-windows-snapdragon-release` preset. Skip if using `-cpu-release`.
powershell -NoProfile -Command "@('HEXAGON_SDK_ROOT','HEXAGON_TOOLS_ROOT','OPENCL_SDK_ROOT','HEXAGON_HTP_CERT','WINDOWS_SDK_BIN') | ForEach-Object { [PSCustomObject]@{ Name=\$_; Machine=[Environment]::GetEnvironmentVariable(\$_,'Machine') } } | Format-List"
# Expected values (all five must be set, typically at Machine scope via `setx /M`):
#   HEXAGON_SDK_ROOT   = C:\Qualcomm\Hexagon_SDK\6.4.0.2
#   HEXAGON_TOOLS_ROOT = C:\Qualcomm\Hexagon_SDK\6.4.0.2\tools\HEXAGON_Tools\19.0.04
#   OPENCL_SDK_ROOT    = C:\Qualcomm\OpenCL_SDK\2.3.2
#   HEXAGON_HTP_CERT   = C:\Users\<user>\Certs\ggml-htp-v1.pfx   ← MUST be .pfx, not .cer
#   WINDOWS_SDK_BIN    = C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0
#                        ← version dir only. CMake appends /arm64 and /x86 on top
#                        (ggml-hexagon/CMakeLists.txt:92-93), so a trailing \arm64
#                        here makes find_program(INF2CAT ...) fail with "Could NOT find INF2CAT".

# 7. (Same as #6) WDK for inf2cat.exe — required to generate the libggml-htp.cat catalog
ls "/c/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x86/Inf2Cat.exe"
# NOT shipped with plain Windows SDK. Must install the full WDK via `wdksetup.exe`
# (~1 GB). The WDK VSIX plugin from the VS Installer does NOT ship Inf2Cat.exe.

# 8. (Same as #6) Test-signing + HTP cert trust (required to LOAD the signed .cat at runtime)
powershell -NoProfile -Command "bcdedit /enum | Select-String testsigning"
# Expected: testsigning             Yes
# Also: ggml-htp-v1.cer must be imported into BOTH Trusted Root Certification
# Authorities AND Trusted Publishers (Local Machine). See docs/run.md §3 cert
# setup + §2 cert generation (makecert + pvk2pfx).
```

### If something is missing

| Missing | Install command | Notes |
|---|---|---|
| Bazelisk | `winget install -e --id Bazel.Bazelisk --accept-package-agreements --accept-source-agreements` | Lands in `…\WinGet\Packages\Bazel.Bazelisk_*\bazelisk.exe`. **Winget does NOT auto-create the `…\WinGet\Links\` shim dir on fresh machines** — if missing, create it and symlink `bazelisk.exe` (and `bazel.exe`) into it, then add `%LOCALAPPDATA%\Microsoft\WinGet\Links` to user PATH. On first invocation, bazelisk downloads bazel 9.x from `releases.bazel.build` and starts the server. |
| LLVM / clang | `winget install -e --id LLVM.LLVM --accept-package-agreements --accept-source-agreements` | ~370 MB download. Installs to `C:\Program Files\LLVM\bin`. The `arm64-windows-llvm` cmake preset hard-codes `clang`/`clang++`, so clang must be on PATH when configuring the SDK. |
| Visual Studio (MSVC) | Already installed at `C:\Program Files\Microsoft Visual Studio\18\Community`. `VsDevCmd.bat -arch=arm64 -host_arch=arm64` sets up the MSVC env that clang's `arm64-pc-windows-msvc` target links against. |
| Cargo / Rust | `winget install -e --id Rustlang.Rustup` then `rustup default stable` | `tokenizers-cpp` (pulled in via geniex-qairt → geniex-proc) invokes `cargo build` during CMake configure. |
| Submodules | `git submodule update --init --recursive` | Without these, CMake fails in `plugins/qairt` or `plugins/llama_cpp`. |
| Developer Mode | Settings UI only — cannot be scripted reliably. See "Platform gotchas" for workarounds when blocked. |
| Go (for direct `go build`) | Not needed when using Bazel — `rules_go` provides a hermetic Go 1.24.13 toolchain (pinned in `MODULE.bazel`). Only install Go if you're going the `cli/Makefile` route. |
| Hexagon SDK / OpenCL SDK | See `third-party/llama.cpp/docs/backend/snapdragon/windows.md` — pinned to Hexagon SDK 6.4.0.2 + Hexagon Tools 19.0.04, Adreno OpenCL SDK 2.3.2. Extract each to `C:\Qualcomm\...`. Only required for the `-release` preset (not `-cpu-release`). |
| WDK (for `inf2cat.exe`) | Install via `wdksetup.exe` from MS (~1 GB). **Not** the VSIX plugin from VS Installer — that one does not ship `Inf2Cat.exe`. Required only for `-release` preset. |
| HTP signing cert (`.pfx`) | Generate locally with `makecert` + `pvk2pfx` (see `docs/run.md` §2). Point `HEXAGON_HTP_CERT` at the `.pfx` (not `.cer`). Downloaded `.cer` files from other people's signed binaries are unusable — they have no private key, and importing random third-party roots is a security risk. |
| Test-signing + cert trust | `bcdedit /set TESTSIGNING ON` (reboot), then `certlm.msc` (elevated) → import `ggml-htp-v1.cer` into **both** Trusted Root CAs and Trusted Publishers. Non-elevated `certlm.msc` silently errors with "store was read only". |

### Known transient: Go module proxy flake during CLI build

Bazel fetches Go deps through `storage.googleapis.com/proxy-golang-org-prod` with short-lived signed URLs. If downloads stall you'll see:

```
fetch_repo: github.com/<pkg>@<ver>: Get "...": http2: server sent GOAWAY ... debug="session_timed_out"
```

This is an upstream signed-URL timeout, not a config issue. Simplest fix: **retry**, optionally pinning the proxy via `--repo_env=GOPROXY="https://proxy.golang.org,direct"`.

## Quick-start build (Windows ARM64, minimal Snapdragon — llama.cpp CPU + QAIRT NPU)

This is the fastest path to a working CLI on a Snapdragon X Elite machine with no Hexagon SDK / OpenCL SDK installed. It still exercises the NPU via the QAIRT plugin's bundled QNN libs. The committed preset `arm64-windows-snapdragon-cpu-release` (in `sdk/CMakePresets.json`) is the shared Snapdragon preset with `GGML_HEXAGON=OFF` and `GGML_OPENCL=OFF`.

```cmd
:: 1. Enter MSVC ARM64 env and put clang on PATH (required by the arm64-windows-llvm toolchain file)
call "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat" -arch=arm64 -host_arch=arm64
set "PATH=C:\Program Files\LLVM\bin;%PATH%"

:: 2. Configure + build + install the SDK
cd sdk
cmake --preset arm64-windows-snapdragon-cpu-release
cmake --build --preset arm64-windows-snapdragon-cpu-release -j 8
cmake --install build-arm64-windows-snapdragon-cpu-release --prefix pkg-geniex
cd ..
```

```bash
# 3. Verify install layout
ls sdk/pkg-geniex/lib/            # geniex.dll, llama_cpp/, qairt/
ls sdk/pkg-geniex/lib/llama_cpp/  # geniex_plugin.dll, llama.dll, ggml*.dll, mtmd.dll, libomp140.aarch64.dll
ls sdk/pkg-geniex/lib/qairt/      # geniex_plugin.dll, geniex_core.dll, htp-files/

# 4. Build + run CLI (first run is ~10 min; downloads zig toolchain + all Go deps)
bazelisk run //cli/cmd/geniex:geniex -- list    # empty table proves plugin chain loads
```

**Important: always launch the CLI via `bazelisk run`, not the raw exe.** The exe at `bazel-bin/cli/cmd/geniex/geniex_/geniex.exe` fails with `error while loading shared libraries` unless the runfiles env (`RUNFILES_DIR` / `RUNFILES_MANIFEST_FILE`) is set — which `bazelisk run` does automatically. Note: when stdout is a tty, the DLL-load error is silently swallowed and the exe appears to "succeed" with exit 0 and empty output. Always redirect stdout/stderr to files if a run seems suspicious.

## Full Snapdragon build (Hexagon NPU + Adreno GPU)

Swap `-cpu-release` for `-release` once the prerequisites in onboarding steps #6–#8 are satisfied. The preset wires up `GGML_HEXAGON=ON`, `GGML_OPENCL=ON`, `GENIEX_PLUGIN_QAIRT=ON`. Key extras vs. the cpu-release path:

- **Path-length workaround**: the Hexagon toolchain enforces a 250-char limit, so `subst G: C:\path\to\geniex` and build from `G:\sdk`.
- **Code-signing step**: build invokes `inf2cat` → `signtool` to produce and sign `libggml-htp.cat` using `HEXAGON_HTP_CERT`. If the inf2cat step fails with "libggml-htp-v*.so is missing or cannot be decompressed", it's a ninja race (cat step ran before the `.so`s finished writing) — just re-run `cmake --build`.
- **Install layout** adds these over `-cpu-release`:
  - `sdk/pkg-geniex/lib/llama_cpp/ggml-hexagon.dll`
  - `sdk/pkg-geniex/lib/llama_cpp/ggml-opencl.dll`
  - `sdk/pkg-geniex/lib/llama_cpp/libggml-htp-v{68,69,73,75,79,81}.so` + `libggml-htp.cat`
- **Runtime verify**: `signtool verify /v /pa sdk\pkg-geniex\lib\llama_cpp\libggml-htp.cat` should show `Issued to: GGML.HTP.v1` and `Successfully verified`.

## Running on a specific backend (CLI)

The CLI has **no** `--device` / `--backend` flag. Backend selection is driven by the `DeviceId` field in the model's `geniex.json` manifest at `%USERPROFILE%\.cache\geniex\models\<name>\geniex.json`. Device name strings the `llama_cpp` plugin recognises (from `sdk/plugins/llama_cpp/src/llm.cpp:73-114`):

| Target      | `DeviceId` value                  | Notes |
|-------------|-----------------------------------|-------|
| Hexagon NPU | `HTP0` (or `HTP0,HTP1,HTP2,HTP3`) | Starting with `HTP0` also forces KV cache to Q8_0 + flash-attn ON (`llm.cpp:136-140`); plugin auto-sets `GGML_HEXAGON_NDEV=4`. |
| Adreno GPU  | `GPUOpenCL`                       | Use `-n 999` on `infer` to offload all layers. |
| CPU         | `CPU` or empty                    | Default. |

End-to-end (PowerShell) for a local GGUF:

```powershell
bazelisk run //cli/cmd/geniex:geniex -- pull <name> --model-hub localfs --local-path <dir> --model-type llm
# Then flip DeviceId in %USERPROFILE%\.cache\geniex\models\<name>\geniex.json and
bazelisk run //cli/cmd/geniex:geniex -- infer <name> -p "..."
```

See `RUN_LOCAL_MODEL.md` (repo root, temp) for the full PowerShell recipe including the `ConvertFrom-Json` / `ConvertTo-Json` one-liner to flip `DeviceId`. Note: `localfs` pull names the model using the last two path components (e.g. `modelfiles` dir under `NexaAI` → model name `NexaAI/<folder>`).

## Build system

Bazel (via **Bazelisk**) is the canonical, unified root build. CMake is used *inside* `sdk/` to produce the native C/C++ library + plugins; everything else (CLI, Python sdist, Go bindings, Windows installer) is driven from Bazel. `cli/Makefile` is a legacy/alternative path — prefer Bazel.

A release-style version is injected at build time via `--define=VERSION=...`; default is `0.0.0.dev0`.

### Common commands

```bash
# Build + run the Go CLI end-to-end (downloads/builds all deps, incl. SDK via local mode)
bazelisk run //cli -- infer Qwen/Qwen3-0.6B-GGUF

# Explicit targets
bazelisk build //cli/cmd/geniex:geniex            # CLI binary
bazelisk build //cli:artifact                      # zipped CLI + SDK runfiles
bazelisk build //cli/release/windows               # Inno Setup installer (Windows ARM64)
bazelisk build //bindings/python:geniex_sdist      # Python source distribution

# Go tests (Bazel)
bazelisk test //cli/...
bazelisk test //cli/cmd/geniex:geniex_test         # single package

# Regenerate Go BUILD.bazel files after editing imports
bazelisk run //:gazelle

# Release-style version
bazelisk build //bindings/python:geniex_sdist --define=VERSION=v0.0.3-alpha.1
```

### SDK (C/C++) — required before `--//sdk:sdk_type=local` (the default)

Bazel's local mode expects `sdk/pkg-geniex/lib/geniex.dll` (Windows) or `libgeniex.so` (Linux) to already exist. Build and install first:

```bash
# Linux native
cd sdk && cmake -S . -B build && cmake --build build -j && cmake --install build --prefix pkg-geniex

# Windows ARM64 / Snapdragon (Hexagon toolchain has a 250-char path limit — use `subst G: <repo>` first)
cd sdk
cmake --preset arm64-windows-snapdragon-release -DGENIEX_TEST=OFF
cmake --build --preset arm64-windows-snapdragon-release -j 8
cmake --install build-arm64-windows-snapdragon-release --prefix pkg-geniex
```

Other presets live in `sdk/CMakePresets.json` (`arm64-linux-snapdragon-*`, `arm64-android-snapdragon-*`). Toggle backends with `-DGENIEX_PLUGIN_LLAMA_CPP=ON/OFF`, `-DGENIEX_PLUGIN_QAIRT=ON/OFF`; `-DGENIEX_DL=ON` for dynamic plugin loading.

`--//sdk:sdk_type=` flag values: `local` (default, consume `sdk/pkg-geniex`), `s3` (WIP), `bazel` (WIP).

### Lint / format

CI enforces three checks; run them locally on changed files before pushing:

- **C/C++**: `clang-format-20 -i <files>` — config in `.clang-format` (Google base, 120 col, 4-space indent). Excludes `sdk/include/external/`, `third-party/`, `sdk/plugins/qnn/libs/`.
- **Python**: `ruff==0.6.9 check <file>` and `ruff format --check --diff <file>` (only `bindings/python/**.py`).
- **Go**: `cd cli && go mod tidy` must leave `go.mod`/`go.sum` clean.

## Native dependency matrix

What each SDK unlocks, at which layer. All four rows are independent — any subset can be toggled on/off.

| Dep | Gated by | Plugin affected | Hardware target | Bundled in-tree? |
|---|---|---|---|---|
| Hexagon SDK | `-DGGML_HEXAGON=ON` (snapdragon preset) | `llama_cpp` only | Snapdragon NPU (HTP) via ggml-hexagon skels | No — external, `HEXAGON_SDK_ROOT`/`HEXAGON_TOOLS_ROOT` required |
| OpenCL SDK | `-DGGML_OPENCL=ON` (snapdragon preset) | `llama_cpp` only | Adreno GPU | No — external, `OPENCL_SDK_ROOT` required (headers + `OpenCL.lib`). Runtime ICD ships with the Snapdragon driver |
| QAIRT / QNN | `-DGENIEX_PLUGIN_QAIRT=ON` | `qairt` plugin only | Snapdragon NPU (HTP) via QNN runtime | **Yes** — `third-party/geniex-qairt/third-party/{windows,android,linux-gcc11.2}/` bundles `QnnHtp.dll`, `Genie.dll`, HTP skels (`libQnnHtpV73/V81.so`, etc.). An externally installed QAIRT is **not** needed for building |
| (none) | CPU-only llama.cpp | `llama_cpp` | CPU | Always works |

Both `GGML_HEXAGON` and `GENIEX_PLUGIN_QAIRT` ultimately target the HTP but via **two different user-space runtimes** (ggml's own Hexagon DSP skels vs. Qualcomm's QNN). They consume different model formats (GGUF vs. QAIRT `.bin` shards) and are not substitutes for each other.

Minimal Snapdragon build that still exercises NPU: `GGML_HEXAGON=OFF`, `GGML_OPENCL=OFF`, `GENIEX_PLUGIN_QAIRT=ON`. llama.cpp runs on CPU; QAIRT plugin still drives the NPU through its bundled QNN libs.

## Architecture

The repo is a bridge/adapter layer: it builds **one C ABI** (the SDK) that multiple AI-inference backends implement as dynamic plugins, and then exposes that single ABI to every language/runtime consumer.

```
  Backends (plugins)          Unified SDK              Consumers
  ─────────────────           ───────────              ─────────
  sdk/plugins/llama_cpp  ─┐                        ┌─ cli/           (Go, cobra)
  sdk/plugins/qairt      ─┼── sdk/include/*.h ─────┼─ bindings/python (pybind11)
  (more via GENIEX_DL)   ─┘   sdk/src/*.cpp        ├─ bindings/go    (cgo)
                              → libgeniex.{so,dll} └─ bindings/android(JNI, 4 layers)
```

Key structural points worth knowing before changing things:

- **Plugin interfaces** are `sdk/include/plugin/I*.h` (ILlm, IVlm, IAsr, etc.). Each backend under `sdk/plugins/<id>/` implements those interfaces; plugin IDs (`llama_cpp`, `qairt`) are baked in as compile definitions in `sdk/CMakeLists.txt`. Dynamic loading is gated by `GENIEX_DL`.
- **CLI (Go)** in `cli/cmd/geniex` uses cobra/viper and links the SDK through `bindings/go` (cgo). It is cross-compiled by Bazel using `hermetic_cc_toolchain` (zig-based — no host C toolchain needed). Registered Go deps are pinned in `MODULE.bazel` (`use_repo(go_deps, ...)`); add new imports there *and* re-run `//:gazelle`.
- **Python binding** publishes as an **sdist only**: `bindings/python/setup.py` has a custom `build_py` that, at `pip install` time, downloads the SDK zip for the current platform from the matching GitHub Release, verifies its SHA-256 sidecar, and bundles libs into the wheel. `GENIEX_SDK_DOWNLOAD_URL` overrides the source; `GENIEX_SKIP_SDK_DOWNLOAD=1` + runtime `GENIEX_LIB_PATH` lets you point at a locally built `sdk/pkg-geniex/lib`.
- **Android binding** uses a 4-layer pattern (public Kotlin wrapper → JNI interface → C++ JNI bridge → core C ABI). Gradle (`bindings/android/`) drives the build; see `bindings/android/README.md` for layer/file mapping.
- **Artifact bundle**: `//cli:artifact` pairs the CLI binary with the SDK runfiles bundled by `sdk/runfiles.bzl` (`sdk_local_bundle` rule), so the `geniex.exe` zip ships with all the `.dll`/`.so`/`.cat` siblings it needs.

## Platform gotchas

- **Windows symlinks**: `.bazelrc` sets `startup --windows_enable_symlinks` and `build --enable_runfiles`. These are used for **two distinct** things: (a) Bazel's execroot — it symlinks each source file into `…\_bazel_<user>\…\execroot\_main\`; (b) this repo's `sdk_local_bundle` rule in `sdk/runfiles.bzl`, which uses `ctx.runfiles(root_symlinks=…)` to place `geniex.dll` + plugins next to the built CLI. Both require the `SeCreateSymbolicLinkPrivilege` user right, which by default only Administrators have. Ways to get it, in order of preference:
    1. **Enable Developer Mode** (one-time, no admin after that). Settings → Privacy & Security → For developers → Developer Mode.
    2. **Grant the "Create symbolic links" user right** via `secpol.msc` → User Rights Assignment. Requires admin once.
    3. **Run the shell elevated** — Administrator tokens always have the privilege.
    4. If all three are blocked by corp policy: file a ticket with IT for option 2, or fall back to `cli/Makefile` (non-Bazel Go build using junctions, which don't need the privilege). Commenting out the two `.bazelrc` lines is a last resort and likely requires patching `sdk/runfiles.bzl` to use `symlinks=` / `files=` instead of `root_symlinks=`.
- **Windows ARM64 (Snapdragon)**: Hexagon toolchain enforces a 250-character path limit. Use `subst G: C:\path\to\geniex` before building `sdk/`.
- **Hexagon NPU (`llama_cpp` + HTP)**: loading `libggml-htp.cat` on Windows requires `bcdedit /set TESTSIGNING ON` **and** importing `ggml-htp-v1.cer` into *both* `Trusted Root Certification Authorities` and `Trusted Publishers` (Local Machine). See `docs/run.md` for the full procedure. Recurring traps:
    - `HEXAGON_HTP_CERT` must point at the `.pfx`, not the `.cer` (signtool needs the private key).
    - `WINDOWS_SDK_BIN` must be the version dir, not the arch subdir — CMake appends `/arm64` and `/x86` itself (`ggml-hexagon/CMakeLists.txt:92-95`). Trailing `\arm64` → `Could NOT find INF2CAT`.
    - `inf2cat.exe` lives in the **WDK** (install via `wdksetup.exe`), not the Windows SDK and not the WDK-as-VSIX from VS Installer.
    - `certlm.msc` must be launched elevated or imports fail with "store was read only".
- **Python + `llama-cpp-python` conflict**: do not install both in the same env — both ship llama.cpp DLLs and collide at load time. Use separate virtualenvs.
- **pybind11**: never hold pybind objects/functions in global scope (statics, ctor/dtor bodies, C++ struct members). Keep them strictly local — see the pybind docs linked from `sdk/README.md`.

## Repo conventions

- External-release libraries and tools live in the repo root (`sdk/`, `cli/`, `bindings/`); directory placement is not used to scope visibility — Bazel `visibility` attrs are. Single unified build system by design.
- `third-party/` holds submodules (`llama.cpp` public upstream, `geniex-qairt` private plugin repo). Don't lint/format inside it.
- Tutorials, cookbook, sample apps live in a separate repo: https://github.com/geniex-app.

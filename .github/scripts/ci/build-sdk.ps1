#Requires -Version 5.1
<#
Build the GenieX SDK on Windows ARM64 (with Hexagon HTP + OpenCL + QAIRT).

This script replaces the inline "Build SDK (Windows)" step that used to live in
.github/workflows/build-sdk.yml. It is called by that workflow after two
preparatory steps have run:

  1. .github/actions/setup-vcvars   -> exports CC, CXX, CMAKE, NINJA,
                                       TOOLCHAIN_FILE via $GITHUB_ENV.
  2. .github/actions/setup-snapdragon-sdks -> exports OPENCL_SDK_ROOT,
                                       HEXAGON_SDK_ROOT, HEXAGON_TOOLS_ROOT,
                                       WINDOWS_SDK_BIN via $GITHUB_ENV.
  3. "Configure HTP signing cert" inline step -> exports HEXAGON_HTP_CERT.

Additional environment inputs read directly here:
  GENIEX_VERSION     (required)  Version string baked into binaries.
  BUILD_DIR          (optional)  Default: sdk/build-windows-arm64.
  INSTALL_PREFIX     (optional)  Default: sdk/pkg-geniex.

Why two-phase build:
  libggml-htp-cat DEPENDS only on the htp-v*.so paths, so ninja schedules the
  .cat signing step in parallel with the htp-v* ExternalProject build steps.
  Inf2Cat then recursively scans the /driver: dir and trips over transient
  files like .ninja_lock under htp-v*-prefix/. Phase 1 builds all HTP skels
  with explicit --target, phase 2 runs everything else (including cat signing
  + geniex) after we clean the prefix dirs.
#>

$ErrorActionPreference = "Stop"

function Require-Env([string]$name) {
  if (-not (Test-Path "Env:$name") -or [string]::IsNullOrEmpty((Get-Item "Env:$name").Value)) {
    throw "Environment variable '$name' is required"
  }
}

Require-Env 'GENIEX_VERSION'
Require-Env 'CC'
Require-Env 'CXX'
Require-Env 'CMAKE'
Require-Env 'NINJA'
Require-Env 'TOOLCHAIN_FILE'
Require-Env 'OPENCL_SDK_ROOT'
Require-Env 'HEXAGON_SDK_ROOT'
Require-Env 'HEXAGON_TOOLS_ROOT'
Require-Env 'HEXAGON_HTP_CERT'
Require-Env 'WINDOWS_SDK_BIN'

$BuildDir      = if ($env:BUILD_DIR)      { $env:BUILD_DIR }      else { 'sdk/build-windows-arm64' }
$InstallPrefix = if ($env:INSTALL_PREFIX) { $env:INSTALL_PREFIX } else { 'sdk/pkg-geniex' }

Write-Host "cmake:  $env:CMAKE"
Write-Host "ninja:  $env:NINJA"
Write-Host "clang:  $env:CC"

# Remove stale cmake cache if present (guards against generator/toolchain mismatch
# after a runner reuse).
Remove-Item -Recurse -Force $BuildDir -ErrorAction SilentlyContinue

& $env:CMAKE -B $BuildDir -G "Ninja" -S sdk --log-level=VERBOSE `
  "-DCMAKE_MAKE_PROGRAM=$env:NINJA" `
  "-DCMAKE_C_COMPILER=$env:CC" `
  "-DCMAKE_CXX_COMPILER=$env:CXX" `
  "-DCMAKE_TOOLCHAIN_FILE=$env:TOOLCHAIN_FILE" `
  -DCMAKE_BUILD_TYPE=Release `
  "-DGENIEX_VERSION=$env:GENIEX_VERSION" `
  -DGENIEX_TEST=OFF `
  -DGENIEX_DEBUG=OFF `
  -DGENIEX_DL=ON `
  -DGENIEX_PLUGIN_LLAMA_CPP=ON `
  -DGENIEX_PLUGIN_QAIRT=ON `
  -DGGML_OPENCL=ON `
  -DGGML_HEXAGON=ON `
  "-DCMAKE_PREFIX_PATH=$env:OPENCL_SDK_ROOT" `
  "-DHEXAGON_SDK_ROOT=$env:HEXAGON_SDK_ROOT" `
  "-DHEXAGON_TOOLS_ROOT=$env:HEXAGON_TOOLS_ROOT" `
  "-DHEXAGON_HTP_CERT=$env:HEXAGON_HTP_CERT" `
  "-DWINDOWS_SDK_BIN=$env:WINDOWS_SDK_BIN" `
  -DPREBUILT_LIB_DIR=windows_aarch64
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed: $LASTEXITCODE" }

# Phase 1: build all Hexagon HTP skels first.
& $env:CMAKE --build $BuildDir -j 8 --target htp-v68 htp-v69 htp-v73 htp-v75 htp-v79 htp-v81
if ($LASTEXITCODE -ne 0) { throw "cmake --build (htp skels) failed: $LASTEXITCODE" }

# Clean up ExternalProject workspaces so Inf2Cat's recursive scan of /driver:
# only sees finalised .so + .inf.
$hexDir = Join-Path $BuildDir "third-party\llama.cpp\ggml\src\ggml-hexagon"
Get-ChildItem $hexDir -Directory -Filter "htp-v*-prefix" -ErrorAction SilentlyContinue |
  ForEach-Object { Remove-Item -Recurse -Force $_.FullName }

# Phase 2: build everything else (cat signing + geniex itself).
& $env:CMAKE --build $BuildDir -j 8
if ($LASTEXITCODE -ne 0) { throw "cmake --build failed: $LASTEXITCODE" }

& $env:CMAKE --install $BuildDir --prefix $InstallPrefix
if ($LASTEXITCODE -ne 0) { throw "cmake install failed: $LASTEXITCODE" }

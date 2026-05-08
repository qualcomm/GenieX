#Requires -Version 5.1
<#
Build the GenieX SDK on Windows ARM64 (with Hexagon HTP + OpenCL + QAIRT).

This script replaces the inline "Build SDK (Windows)" step that used to live in
.github/workflows/build-sdk.yml. It is called by that workflow after these
preparatory steps have run:

  1. .github/actions/setup-vcvars           -> exports VCVARS_BAT, VCVARS_ARGS,
                                               LLVM_BIN, CC, CXX, CMAKE, NINJA,
                                               TOOLCHAIN_FILE via $GITHUB_ENV.
  2. .github/actions/setup-snapdragon-sdks  -> exports OPENCL_SDK_ROOT,
                                               HEXAGON_SDK_ROOT, HEXAGON_TOOLS_ROOT,
                                               WINDOWS_SDK_BIN via $GITHUB_ENV.
  3. "Configure HTP signing cert" inline step -> exports HEXAGON_HTP_CERT.

This script sources vcvars into its own PowerShell process so the cmake child
process sees the full vcvars environment (PATH entries for the Windows SDK bin
dirs, INCLUDE, LIB, etc.) — matching the semantics of `call vcvars && cmake`.

Additional environment inputs read directly here:
  GENIEX_VERSION     (required)  Version string baked into binaries.
  BUILD_DIR          (optional)  Default: sdk/build-windows-arm64.
  INSTALL_PREFIX     (optional)  Default: sdk/pkg-geniex.

Hexagon HTP skel <-> inf2cat ordering:
  install-htp-skels in sdk/plugins/llama_cpp/CMakeLists.txt stages the
  htp-v*.so files into ggml-hexagon's binary dir and scrubs the
  htp-v*-prefix/ ExternalProject workspaces; libggml-htp-cat
  add_dependencies on that target so inf2cat's recursive /driver: scan
  only sees finalised .so + .inf. A single `cmake --build` is enough.
#>

$ErrorActionPreference = "Stop"

function Require-Env([string]$name) {
  if (-not (Test-Path "Env:$name") -or [string]::IsNullOrEmpty((Get-Item "Env:$name").Value)) {
    throw "Environment variable '$name' is required"
  }
}

Require-Env 'GENIEX_VERSION'
Require-Env 'VCVARS_BAT'
Require-Env 'LLVM_BIN'
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

# Import the full vcvars environment into this PowerShell process so cmake and
# its downstream tools (clang, ninja, and crucially inf2cat.exe in the Windows
# SDK x86 bin dir) see the same PATH/INCLUDE/LIB as `call vcvars && cmake`.
$envDump = cmd /c "`"$env:VCVARS_BAT`" $env:VCVARS_ARGS && set"
foreach ($line in $envDump) {
  if ($line -match "^(.+?)=(.*)$") {
    [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
  }
}
# Prepend VS-bundled Llvm\bin so unqualified clang lookups hit the right binary.
$env:PATH = "$env:LLVM_BIN;$env:PATH"

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
  -DCMAKE_C_COMPILER_LAUNCHER=ccache `
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache `
  "-DGENIEX_VERSION=$env:GENIEX_VERSION" `
  -DGENIEX_TEST=OFF `
  -DGENIEX_DEBUG=OFF `
  -DGENIEX_DL=ON `
  -DGENIEX_PLUGIN_LLAMA_CPP=ON `
  -DGENIEX_PLUGIN_QAIRT=ON `
  -DGENIEX_MODEL_MANAGER=ON `
  -DGGML_OPENCL=ON `
  -DGGML_HEXAGON=ON `
  "-DCMAKE_PREFIX_PATH=$env:OPENCL_SDK_ROOT" `
  "-DHEXAGON_SDK_ROOT=$env:HEXAGON_SDK_ROOT" `
  "-DHEXAGON_TOOLS_ROOT=$env:HEXAGON_TOOLS_ROOT" `
  "-DHEXAGON_HTP_CERT=$env:HEXAGON_HTP_CERT" `
  "-DWINDOWS_SDK_BIN=$env:WINDOWS_SDK_BIN" `
  -DPREBUILT_LIB_DIR=windows_aarch64
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed: $LASTEXITCODE" }

$Jobs = [Environment]::ProcessorCount
Write-Host "Building with -j $Jobs"

& $env:CMAKE --build $BuildDir -j $Jobs
if ($LASTEXITCODE -ne 0) { throw "cmake --build failed: $LASTEXITCODE" }

& $env:CMAKE --install $BuildDir --prefix $InstallPrefix
if ($LASTEXITCODE -ne 0) { throw "cmake install failed: $LASTEXITCODE" }

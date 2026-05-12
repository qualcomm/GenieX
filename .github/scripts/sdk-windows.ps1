#Requires -Version 5.1
<#
Build the GenieX SDK on Windows ARM64 via the arm64-windows-snapdragon-release
CMake preset. Called by .github/workflows/_build-sdk.yml after:

  1. .github/actions/setup-vcvars           -> VCVARS_BAT, VCVARS_ARGS, LLVM_BIN
  2. .github/actions/setup-snapdragon-sdks  -> OPENCL_SDK_ROOT, HEXAGON_SDK_ROOT,
                                               HEXAGON_TOOLS_ROOT, WINDOWS_SDK_BIN
  3. "Configure HTP signing cert" inline    -> HEXAGON_HTP_CERT

The preset references those env vars via $env{...}; cmake configure fails loudly
if any are missing.

Environment inputs read here:
  GENIEX_VERSION   (required)  Version string baked into binaries.
  BUILD_DIR        (optional)  Default: sdk/build-arm64-windows-snapdragon-release.
  INSTALL_PREFIX   (optional)  Default: sdk/pkg-geniex.
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

$BuildDir      = if ($env:BUILD_DIR)      { $env:BUILD_DIR }      else { 'sdk/build-arm64-windows-snapdragon-release' }
$InstallPrefix = if ($env:INSTALL_PREFIX) { $env:INSTALL_PREFIX } else { 'sdk/pkg-geniex' }

# Source vcvars into this process so cmake's child processes see PATH/INCLUDE/LIB
# for the Windows SDK (clang, ninja, inf2cat.exe).
$envDump = cmd /c "`"$env:VCVARS_BAT`" $env:VCVARS_ARGS && set"
foreach ($line in $envDump) {
  if ($line -match "^(.+?)=(.*)$") {
    [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
  }
}
$env:PATH = "$env:LLVM_BIN;$env:PATH"

Remove-Item -Recurse -Force $BuildDir -ErrorAction SilentlyContinue

cmake -S sdk --preset arm64-windows-snapdragon-release -B $BuildDir --log-level=VERBOSE `
  "-DGENIEX_VERSION=$env:GENIEX_VERSION" `
  -DCMAKE_C_COMPILER_LAUNCHER=ccache `
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed: $LASTEXITCODE" }

cmake --build $BuildDir -j $([Environment]::ProcessorCount)
if ($LASTEXITCODE -ne 0) { throw "cmake --build failed: $LASTEXITCODE" }

cmake --install $BuildDir --prefix $InstallPrefix
if ($LASTEXITCODE -ne 0) { throw "cmake install failed: $LASTEXITCODE" }

param(
    [Parameter(Mandatory=$true)][string]$PhononSource,
    [Parameter(Mandatory=$true)][string]$QairtRoot,
    [Parameter(Mandatory=$true)][string]$BuildDirectory,
    [Parameter(Mandatory=$true)][string]$InstallDirectory,
    [string]$CMake = 'cmake',
    [switch]$WithQairt
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path "$PSScriptRoot/../../..").Path
$PhononSource = (Resolve-Path -LiteralPath $PhononSource).Path
$QairtRoot = (Resolve-Path -LiteralPath $QairtRoot).Path
if (!(Test-Path -LiteralPath "$QairtRoot/include/QNN/QnnInterface.h")) { throw 'QAIRT headers not found.' }
$qairtPlugin = if ($WithQairt) { 'ON' } else { 'OFF' }
& $CMake -S "$repo/sdk" -B $BuildDirectory -A ARM64 `
    -DGENIEX_PLUGIN_LLAMA_CPP=OFF "-DGENIEX_PLUGIN_QAIRT=$qairtPlugin" `
    -DGENIEX_PLUGIN_PHONON=ON -DGENIEX_TTS_TESTS=ON `
    "-DGENIEX_PHONON_SOURCE=$PhononSource" "-DQAIRT_ROOT=$QairtRoot"
if ($LASTEXITCODE) { throw 'CMake configure failed.' }
# MSVC's file tracker retains a shorter path limit than the compiler. Disable
# tracking for this build invocation, without changing machine configuration.
& $CMake --build $BuildDirectory --config Release --parallel 4 -- /p:TrackFileAccess=false /nodeReuse:false
if ($LASTEXITCODE) { throw 'Build failed. Check native ARM64 Cargo, clang and MSVC prerequisites.' }
$ctest = Join-Path (Split-Path (Get-Command $CMake).Source) 'ctest.exe'
& $ctest --test-dir $BuildDirectory -C Release --output-on-failure
if ($LASTEXITCODE) { throw 'TTS API contracts failed.' }
& $CMake --install $BuildDirectory --config Release --prefix $InstallDirectory
if ($LASTEXITCODE) { throw 'Install failed.' }
Write-Output "Built TTS SDK at $InstallDirectory (QAIRT plugin: $qairtPlugin). The installed GenieX CLI is unchanged."

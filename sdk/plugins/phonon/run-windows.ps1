param(
    [Parameter(Mandatory=$true)][string]$SdkDirectory,
    [Parameter(Mandatory=$true)][string]$QairtRoot,
    [Parameter(Mandatory=$true)][string]$TextLibrary,
    [Parameter(Mandatory=$true)][string[]]$Models,
    [Parameter(Mandatory=$true)][string]$OutDirectory,
    [string]$SocModel = 'SC8380XP',
    [string]$Python = 'python',
    [int]$Repeats = 6,
    [switch]$Play
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path "$PSScriptRoot/../../..").Path
$SdkDirectory = (Resolve-Path -LiteralPath $SdkDirectory).Path
$QairtRoot = (Resolve-Path -LiteralPath $QairtRoot).Path
$TextLibrary = (Resolve-Path -LiteralPath $TextLibrary).Path
$Models = @($Models | ForEach-Object { (Resolve-Path -LiteralPath $_).Path })
$OutDirectory = [System.IO.Path]::GetFullPath($OutDirectory)
$keys = @('PYTHONPATH','GENIEX_LIB_PATH','GENIEX_PLUGIN_PATH','PHONON_QNN_PROFILE','ADSP_LIBRARY_PATH')
$previous = @{}
foreach ($key in $keys) { $previous[$key] = [Environment]::GetEnvironmentVariable($key, 'Process') }
try {
    $env:PYTHONPATH = "$repo/bindings/python"
    $env:GENIEX_LIB_PATH = "$SdkDirectory/lib/geniex.dll"
    $env:GENIEX_PLUGIN_PATH = "$SdkDirectory/lib"
    $env:PHONON_QNN_PROFILE = '1'
    $env:ADSP_LIBRARY_PATH = "$QairtRoot/lib/hexagon-v73/unsigned"
    for ($index=0; $index -lt $Models.Count; $index++) {
        $out = Join-Path $OutDirectory "model-$index"
        & $Python "$repo/examples/python/phonon_tts.py" --model $Models[$index] `
            --runtime "$QairtRoot/lib/aarch64-windows-msvc" --text-library $TextLibrary `
            --soc $SocModel --repeats $Repeats --out $out
        if ($LASTEXITCODE) { throw "NPU synthesis failed for $($Models[$index]). No CPU/GPU retry was attempted." }
        if ($Play) {
            $wav = Join-Path $out 'run-0.wav'
            $player = New-Object System.Media.SoundPlayer $wav
            try { $player.PlaySync() } finally { $player.Dispose() }
        }
    }
} finally {
    foreach ($key in $keys) { [Environment]::SetEnvironmentVariable($key, $previous[$key], 'Process') }
}

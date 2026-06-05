# geniex_benchmark entry script for QDC Windows (POWERSHELL framework).
#
# QDC extracts the artifact zip to C:\Temp\TestContent\ and runs this via
# PowerShell. Anything under C:\Temp\QDC_Logs\ is auto-uploaded. run_qdc_jobs.py
# fills the here-string below with `name|plugin|csv_devices|url|kind` lines.

$ErrorActionPreference = "Continue"

$LOG = "C:\Temp\QDC_Logs"
$OUT = "$LOG\results"
$MODELS = "C:\Temp\models"
$BUNDLE = "C:\Temp\TestContent\pkg-geniex"
$TSV = "C:\Temp\matrix.tsv"

New-Item -ItemType Directory -Force -Path $LOG, $OUT, $MODELS | Out-Null
Start-Transcript -Path "$LOG\script.log" -Force | Out-Null

# Trust the self-signed cert the HTP .cat catalogs are signed with, or the
# Hexagon backends fail their code-integrity check at load.
$cert = "C:\Temp\TestContent\ggml-htp-v1.cer"
if (Test-Path $cert) {
    & certutil.exe -addstore -f Root $cert | Out-Null
    & certutil.exe -addstore -f TrustedPublisher $cert | Out-Null
}

Set-Location $BUNDLE
$env:GENIEX_PLUGIN_PATH = "$BUNDLE\lib"
$env:PATH = "$BUNDLE\lib;$BUNDLE\lib\llama_cpp;$BUNDLE\lib\qairt;$BUNDLE\lib\qairt\htp-files;$env:PATH"

$rows = @'
{MODELS}
'@ -split "`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ }

Remove-Item $TSV -ErrorAction SilentlyContinue
foreach ($row in $rows) {
    $name, $plugin, $devs, $url, $kind = $row -split '\|'
    $dir = "$MODELS\$name"
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    Write-Output "=== fetch $name ($kind) ==="
    if ($kind -eq "bundle") {
        $mpath = "$dir\bundle"
        if (-not (Test-Path $mpath)) {
            try {
                Invoke-WebRequest -Uri $url -OutFile "$dir\b.zip"
                Expand-Archive -Path "$dir\b.zip" -DestinationPath $mpath -Force
                $entries = Get-ChildItem $mpath
                if ($entries.Count -eq 1 -and $entries[0].PSIsContainer) {
                    Move-Item "$($entries[0].FullName)\*" $mpath
                    Remove-Item $entries[0].FullName
                }
            } catch {
                Write-Output "WARNING: $name fetch failed, skipping"
                Remove-Item $mpath -Recurse -ErrorAction SilentlyContinue
                continue
            }
        }
    } else {
        $mpath = "$dir\model.gguf"
        if (-not (Test-Path $mpath)) {
            try {
                Invoke-WebRequest -Uri $url -OutFile $mpath
            } catch {
                Write-Output "WARNING: $name fetch failed, skipping"
                continue
            }
        }
    }
    $mpathfwd = $mpath -replace '\\', '/'
    foreach ($d in $devs -split ',') {
        "{0}-{1}-{2}`t{1}`t{2}`t{3}" -f $name, $plugin, $d, $mpathfwd | Add-Content $TSV
    }
}

Write-Output "=== matrix ==="
Get-Content $TSV
& "$BUNDLE\bin\geniex_benchmark.exe" --matrix-file $TSV --output-json-dir "$OUT" --warmup 1 --repeat 3
Write-Output "rc=$LASTEXITCODE  ($((Get-ChildItem $OUT).Count) cell json files)"
Write-Output "=== done ==="
Stop-Transcript | Out-Null
exit 0

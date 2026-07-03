# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# geniex-bench entry script for QDC Windows (POWERSHELL framework).
#
# QDC extracts the artifact zip to C:\Temp\TestContent\ and runs this via
# PowerShell. Anything under C:\Temp\QDC_Logs\ is auto-uploaded. run_qdc_jobs.py
# substitutes:
#   {MODELS}  → `name|plugin|csv_devices|model_id|vlm|image` lines
#   {CHIPSET} → AI Hub chipset slug (e.g. qualcomm-snapdragon-x-elite)
# Each cell's column-4 model_id is resolved on the device by the model-
# manager C API (multi-connection HTTPS, byte-range resume) on first
# reference; the cached copy is reused across the ctx sweep — replacing
# the Invoke-WebRequest loop the previous script ran (which OOMed on
# >~7 GB GGUFs on X2 Elite).
#
# We sweep ctx in {512, 1024, 4096} per cell to align with test-llama.cpp's
# PERFORMANCE SESSION. Two prefill modes coexist:
#   - llama_cpp cells use random-ids prefill (`-p N`, mirrors llama-bench
#     `pp{N}`), so reported pp is exactly the ctx value;
#   - qairt cells go through prompt_utf8 (the plugin doesn't accept
#     pre-tokenized input_ids — see issue #1008), with a pre-trimmed
#     `sample_prompt_${ctx}.txt` per ctx so prompt length is bounded.
# Each plugin gets its own per-ctx TSV so the two invocations don't mix.

$ErrorActionPreference = "Continue"

$LOG = "C:\Temp\QDC_Logs"
$OUT = "$LOG\results"
$MM_CACHE = "C:\Temp\geniex-cache"
$TC = "C:\Temp\TestContent"
$BUNDLE = "$TC\pkg-geniex"
$PROMPTS = "$TC\prompts"

New-Item -ItemType Directory -Force -Path $LOG, $OUT, $MM_CACHE | Out-Null
Start-Transcript -Path "$LOG\script.log" -Force | Out-Null

# Trust the self-signed cert the HTP .cat catalogs are signed with, or the
# Hexagon backends fail their code-integrity check at load.
$cert = "$TC\ggml-htp-v1.cer"
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

$IMG = "$TC/test.png" -replace '\\', '/'

$ctxList = @(512, 1024, 4096)
$tsvByPluginCtx = @{}
foreach ($plugin in @("llama", "qairt")) {
    foreach ($ctx in $ctxList) {
        $tsvByPluginCtx["$plugin-$ctx"] = "C:\Temp\matrix-$plugin-$ctx.tsv"
        Remove-Item $tsvByPluginCtx["$plugin-$ctx"] -ErrorAction SilentlyContinue
    }
}

foreach ($row in $rows) {
    $name, $plugin, $devs, $model_id, $vlm, $image = $row -split '\|'
    Write-Output "=== plan $name id=$model_id ==="
    $imgpath = if ($image -eq "1") { $IMG } else { "" }
    $bucket = if ($plugin -eq "qairt") { "qairt" } elseif ($plugin -eq "llama_cpp") { "llama" } else { "" }
    if (-not $bucket) {
        Write-Output "WARN: unknown plugin $plugin in $name, skipping"
        continue
    }
    foreach ($d in $devs -split ',') {
        foreach ($ctx in $ctxList) {
            # Columns 5/6 (tokenizer/mmproj) intentionally blank: the model
            # manager fills both from the resolved manifest.
            "{0}-{1}-{2}-c{3}`t{1}`t{2}`t{4}`t`t`t{5}`t{6}" -f `
                $name, $plugin, $d, $ctx, $model_id, $imgpath, $vlm `
                | Add-Content $tsvByPluginCtx["$bucket-$ctx"]
        }
    }
}

foreach ($ctx in $ctxList) {
    $llamaTsv = $tsvByPluginCtx["llama-$ctx"]
    $qairtTsv = $tsvByPluginCtx["qairt-$ctx"]

    if ((Test-Path $llamaTsv) -and ((Get-Item $llamaTsv).Length -gt 0)) {
        Write-Output "=== matrix llama_cpp ctx=$ctx (random-ids prefill) ==="
        Get-Content $llamaTsv
        & "$BUNDLE\bin\geniex-bench.exe" --matrix-file $llamaTsv --output-json-dir "$OUT" -r 3 `
            -c $ctx -p $ctx `
            --mm-data-dir $MM_CACHE --chipset "{CHIPSET}"
        Write-Output "rc=$LASTEXITCODE  ($((Get-ChildItem $OUT).Count) cell json files so far)"
    }

    if ((Test-Path $qairtTsv) -and ((Get-Item $qairtTsv).Length -gt 0)) {
        Write-Output "=== matrix qairt ctx=$ctx (prompt-file) ==="
        Get-Content $qairtTsv
        & "$BUNDLE\bin\geniex-bench.exe" --matrix-file $qairtTsv --output-json-dir "$OUT" -r 3 `
            -c $ctx --prompt-file "$PROMPTS\sample_prompt_$ctx.txt" `
            --mm-data-dir $MM_CACHE --chipset "{CHIPSET}"
        Write-Output "rc=$LASTEXITCODE  ($((Get-ChildItem $OUT).Count) cell json files so far)"
    }
}
Write-Output "=== done ==="
Stop-Transcript | Out-Null
exit 0

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
# PERFORMANCE SESSION. Three buckets, each with its own per-ctx TSV so
# their invocations don't mix:
#   - llama_cpp cells use random-ids prefill (`-p N`);
#   - qairt cells go through prompt_utf8 with `sample_prompt_${ctx}.txt`
#     because the plugin doesn't accept pre-tokenized input_ids (#1008);
#   - spec (llama_cpp speculative-decoding) cells share random-ids
#     prefill and additionally pass --spec-type/--draft-model/--draft-tokens
#     as CLI-level flags per matrix invocation.

$ErrorActionPreference = "Continue"

$LOG = "C:\Temp\QDC_Logs"
$OUT = "$LOG\results"
$MM_CACHE = "C:\Temp\geniex-cache"
$TC = "C:\Temp\TestContent"
$BUNDLE = "$TC\pkg-geniex"
$PROMPTS = "$TC\prompts"

New-Item -ItemType Directory -Force -Path $LOG, $OUT, $MM_CACHE | Out-Null
# QDC reuses the same physical host across jobs, so $OUT can hold stale cell
# JSON files from earlier sessions. Wipe them so log-upload can't ship them
# back and pollute this job's cell set.
Remove-Item -Path "$OUT\*.json" -Force -ErrorAction SilentlyContinue
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

# Sweep dimensions come from workflow inputs (with defaults filled host-side).
# ctxList / ppList / tgList are parallel arrays of equal length.
$ctxList = @({CTX_LIST})
$ppList  = @({PP_LIST})
$tgList  = @({TG_LIST})
$tsvByPluginCtx = @{}
foreach ($plugin in @("llama", "qairt", "spec")) {
    foreach ($ctx in $ctxList) {
        $tsvByPluginCtx["$plugin-$ctx"] = "C:\Temp\matrix-$plugin-$ctx.tsv"
        Remove-Item $tsvByPluginCtx["$plugin-$ctx"] -ErrorAction SilentlyContinue
    }
}
# Spec CLI params are per-invocation, not per-cell. Keyed by "$ctx".
$specParamsByCtx = @{}

foreach ($row in $rows) {
    $name, $plugin, $devs, $model_id, $vlm, $image, $spec_type, $draft_model_id, $draft_tokens = $row -split '\|'
    Write-Output "=== plan $name id=$model_id ==="
    $bucket = if ($spec_type) { "spec" } elseif ($plugin -eq "qairt") { "qairt" } elseif ($plugin -eq "llama_cpp") { "llama" } else { "" }
    if (-not $bucket) {
        Write-Output "WARN: unknown plugin $plugin in $name, skipping"
        continue
    }
    if ($bucket -ne "qairt") { $vlm = ""; $image = "" }
    $imgpath = if ($image -eq "1") { $IMG } else { "" }
    foreach ($d in $devs -split ',') {
        foreach ($ctx in $ctxList) {
            # Columns 5/6 (tokenizer/mmproj) intentionally blank: the model
            # manager fills both from the resolved manifest.
            "{0}-{1}-{2}-c{3}`t{1}`t{2}`t{4}`t`t`t{5}`t{6}" -f `
                $name, $plugin, $d, $ctx, $model_id, $imgpath, $vlm `
                | Add-Content $tsvByPluginCtx["$bucket-$ctx"]
            if ($bucket -eq "spec") {
                $specParamsByCtx["$ctx"] = @{
                    type   = $spec_type
                    draft  = $draft_model_id
                    tokens = $draft_tokens
                }
            }
        }
    }
}

for ($i = 0; $i -lt $ctxList.Count; $i++) {
    $ctx = $ctxList[$i]
    $pp  = $ppList[$i]
    $tg  = $tgList[$i]
    $llamaTsv = $tsvByPluginCtx["llama-$ctx"]
    $qairtTsv = $tsvByPluginCtx["qairt-$ctx"]

    if ((Test-Path $llamaTsv) -and ((Get-Item $llamaTsv).Length -gt 0)) {
        Write-Output "=== matrix llama_cpp ctx=$ctx pp=$pp tg=$tg (random-ids prefill) ==="
        Get-Content $llamaTsv
        & "$BUNDLE\bin\geniex-bench.exe" --matrix-file $llamaTsv --output-json-dir "$OUT" -r 3 `
            -c $ctx -p $pp -n $tg `
            --mm-data-dir $MM_CACHE --chipset "{CHIPSET}"
        Write-Output "rc=$LASTEXITCODE  ($((Get-ChildItem $OUT).Count) cell json files so far)"
    }

    if ((Test-Path $qairtTsv) -and ((Get-Item $qairtTsv).Length -gt 0)) {
        Write-Output "=== matrix qairt ctx=$ctx tg=$tg (prompt-file) ==="
        Get-Content $qairtTsv
        & "$BUNDLE\bin\geniex-bench.exe" --matrix-file $qairtTsv --output-json-dir "$OUT" -r 3 `
            -c $ctx -n $tg --prompt-file "$PROMPTS\sample_prompt_$ctx.txt" `
            --mm-data-dir $MM_CACHE --chipset "{CHIPSET}"
        Write-Output "rc=$LASTEXITCODE  ($((Get-ChildItem $OUT).Count) cell json files so far)"
    }

    $specTsv = $tsvByPluginCtx["spec-$ctx"]
    if ((Test-Path $specTsv) -and ((Get-Item $specTsv).Length -gt 0)) {
        $sp = $specParamsByCtx["$ctx"]
        # Spec-decoding needs `--draft-tokens` extra KV slots on the last
        # decode step (target + draft), or llama.cpp trips
        # "decode: failed to find a memory slot for batch of size N+1".
        # Bench defaults pp+tg = ctx exactly, so trim tg by that margin.
        $draftHeadroom = if ($sp.tokens) { [int]$sp.tokens + 1 } else { 4 }
        $specTg = [Math]::Max(1, [int]$tg - $draftHeadroom)
        Write-Output "=== matrix spec ctx=$ctx pp=$pp tg=$specTg type=$($sp.type) draft=$($sp.draft) n_max=$($sp.tokens) (random-ids prefill) ==="
        Get-Content $specTsv
        $extra = @()
        if ($sp.tokens) { $extra += @("--draft-tokens", $sp.tokens) }
        & "$BUNDLE\bin\geniex-bench.exe" --matrix-file $specTsv --output-json-dir "$OUT" -r 1 --no-warmup `
            -c $ctx -p $pp -n $specTg `
            --spec-type $sp.type --draft-model $sp.draft @extra `
            --mm-data-dir $MM_CACHE --chipset "{CHIPSET}"
        Write-Output "rc=$LASTEXITCODE  ($((Get-ChildItem $OUT).Count) cell json files so far)"
    }
}
Write-Output "=== done ==="
Stop-Transcript | Out-Null
exit 0

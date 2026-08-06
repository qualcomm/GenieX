# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

# QDC pytest entry — Windows ARM64.
# The QDC image runs PowerShell 5.1 Desktop — do not use 7+ syntax.

$ErrorActionPreference = "Continue"

$LOG = "C:\Temp\QDC_Logs"
$ROOT = "C:\Temp\TestContent"
$PY_DIR = "$ROOT\python-embed"

New-Item -ItemType Directory -Force -Path $LOG | Out-Null
Start-Transcript -Path "$LOG\harness.log" -Force | Out-Null

# HTP self-signed cert — without it the Hexagon backend crashes 0xC0000409 pre-main.
$cert = "$ROOT\ggml-htp-v1.cer"
if (Test-Path $cert) {
    & certutil.exe -addstore -f Root $cert
    & certutil.exe -addstore -f TrustedPublisher $cert
}

$PY_VER = "3.13.1"
$PY_ZIP = "$ROOT\python-embed.zip"
$PY_URL = "https://www.python.org/ftp/python/$PY_VER/python-$PY_VER-embed-arm64.zip"
Invoke-WebRequest -Uri $PY_URL -OutFile $PY_ZIP -UseBasicParsing
Expand-Archive -Path $PY_ZIP -DestinationPath $PY_DIR -Force
# Embed build's ._pth is isolated-mode: PYTHONPATH is ignored, only these dirs count.
$pth = Get-ChildItem "$PY_DIR\python*._pth" | Select-Object -First 1
$content = (Get-Content $pth.FullName) -replace '^#import site', 'import site'
$content += "$ROOT\bindings\python"
$content | Set-Content $pth.FullName
Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile "$ROOT\get-pip.py" -UseBasicParsing
& "$PY_DIR\python.exe" "$ROOT\get-pip.py" --no-warn-script-location 2>&1
& "$PY_DIR\python.exe" -m pip install --no-warn-script-location "pytest>=7.0" "pytest-reportlog>=0.4" "tqdm>=4.65" 2>&1

$env:GENIEX_DEVICE_TEST = "1"
$env:GENIEX_LIB_PATH = "$ROOT\pkg-geniex\lib"
$env:PATH = "$ROOT\pkg-geniex\lib;$ROOT\pkg-geniex\lib\llama_cpp;$ROOT\pkg-geniex\lib\qairt;$ROOT\pkg-geniex\lib\qairt\htp-files;$env:PATH"
# QDC's shared link times out fixtures if all model pulls burst in parallel.
$env:HF_HUB_DOWNLOAD_CONCURRENCY = "1"

& "$PY_DIR\python.exe" -c "import geniex; geniex.init(); geniex.deinit()" 2>&1

Set-Location "$ROOT\tests"
& "$PY_DIR\python.exe" -m pytest . -v `
    --tb=short `
    --junitxml="$LOG\device-results.xml" `
    --report-log="$LOG\device-report.log" `
    -m "llama_cpp or qairt" 2>&1

Stop-Transcript | Out-Null
exit 0

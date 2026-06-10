# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# QDC pytest entry — X Elite Windows ARM64.
#
# QDC's POWERSHELL framework extracts the artifact zip to C:\Temp\TestContent\
# and runs this script there; anything written under C:\Temp\QDC_Logs\ is
# auto-uploaded as job logs. The artifact (built by run_qdc_pytest.py) ships
# pkg-geniex (windows-arm64 SDK), the bindings/python tree, the tests/ tree,
# the HTP self-signed cert, and the VLM sample image — see build_windows_artifact.

$ErrorActionPreference = "Continue"

$LOG = "C:\Temp\QDC_Logs"
$ROOT = "C:\Temp\TestContent"
$PY_DIR = "$ROOT\python-embed"

New-Item -ItemType Directory -Force -Path $LOG | Out-Null
Start-Transcript -Path "$LOG\harness.log" -Force | Out-Null

# --- 1. Trust the HTP self-signed cert.
# Without it the Hexagon backend fails its code-integrity check at .so load
# time and crashes pre-main with 0xC0000409. Mirrors run_windows.ps1 (scorecard).
$cert = "$ROOT\ggml-htp-v1.cer"
if (Test-Path $cert) {
    & certutil.exe -addstore -f Root $cert | Out-Null
    & certutil.exe -addstore -f TrustedPublisher $cert | Out-Null
}

# --- 2. Bootstrap a portable Python.
# QDC's POWERSHELL sandbox has no preinstalled Python. python.org's
# "embeddable" zip is the lightest option: one zip, no installer required,
# extract-and-run. The embed build ships without pip, so we fetch get-pip
# separately and uncomment the import-site line in pythonNN._pth so site
# packages and PYTHONPATH are honoured.
$PY_VER = "3.13.1"
$PY_ZIP = "$ROOT\python-embed.zip"
$PY_URL = "https://www.python.org/ftp/python/$PY_VER/python-$PY_VER-embed-arm64.zip"
Write-Output "=== fetch python $PY_VER (arm64 embed) ==="
Invoke-WebRequest -Uri $PY_URL -OutFile $PY_ZIP
Expand-Archive -Path $PY_ZIP -DestinationPath $PY_DIR -Force
$pth = Get-ChildItem "$PY_DIR\python*._pth" | Select-Object -First 1
(Get-Content $pth.FullName) -replace '^#import site', 'import site' | Set-Content $pth.FullName
Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile "$ROOT\get-pip.py"
& "$PY_DIR\python.exe" "$ROOT\get-pip.py" --no-warn-script-location
& "$PY_DIR\python.exe" -m pip install --no-warn-script-location "pytest>=7.0" "tqdm>=4.65"

# --- 3. Run the SDK pytest matrix against the windows-arm64 SDK.
# tests/conftest.py treats Windows + arm64 as a Snapdragon host, so the
# `snapdragon`-marked llama_cpp NPU/Hybrid + qairt cells all run here.
$env:GENIEX_DEVICE_TEST = "1"
$env:GENIEX_LIB_PATH = "$ROOT\pkg-geniex\lib"
$env:PYTHONPATH = "$ROOT\bindings\python"
# Plugin libs live one dir down from GENIEX_LIB_PATH; PATH lets the OS loader
# find their transitive deps when ctypes opens libgeniex.dll.
$env:PATH = "$ROOT\pkg-geniex\lib;$ROOT\pkg-geniex\lib\llama_cpp;$ROOT\pkg-geniex\lib\qairt;$ROOT\pkg-geniex\lib\qairt\htp-files;$env:PATH"
# Cap concurrent HF downloads — the QDC sandbox link is shared with other
# jobs and bursting all four model pulls in parallel times out fixtures.
$env:HF_HUB_DOWNLOAD_CONCURRENCY = "1"

Set-Location "$ROOT\tests"
Write-Output "=== pytest ==="
& "$PY_DIR\python.exe" -m pytest . -v `
    --tb=short `
    --junitxml="$LOG\device-results.xml" `
    -m "not api" `
    -o "norecursedirs=cli qdc"
$rc = $LASTEXITCODE
Write-Output "=== pytest exit $rc ==="

Stop-Transcript | Out-Null
exit 0

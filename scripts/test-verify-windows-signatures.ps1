# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

<#
.SYNOPSIS
    Self-test for scripts/verify-windows-signatures.ps1.

.DESCRIPTION
    Creates real signed and unsigned Windows PE fixtures and asserts the
    verifier accepts a fully signed tree and rejects one containing an
    unsigned binary. Exits non-zero on any failed assertion. Windows only.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts/test-verify-windows-signatures.ps1
#>

$ErrorActionPreference = "Stop"

$verify = Join-Path $PSScriptRoot "verify-windows-signatures.ps1"
if (-not (Test-Path -LiteralPath $verify)) {
    throw "verifier not found: $verify"
}

$work = Join-Path $env:TEMP "geniex-verify-test"
if (Test-Path -LiteralPath $work) {
    Remove-Item -LiteralPath $work -Recurse -Force
}
New-Item -ItemType Directory -Path $work | Out-Null

$signedDir = Join-Path $work "signed"
$unsignedDir = Join-Path $work "unsigned"
$mixedDir = Join-Path $work "mixed"
foreach ($d in @($signedDir, $unsignedDir, $mixedDir)) {
    New-Item -ItemType Directory -Path $d | Out-Null
}

# Signed fixture: a Microsoft-signed system DLL.
$signedDll = (Get-ChildItem "$env:WINDIR\System32\*.dll" |
    Where-Object { (Get-AuthenticodeSignature -LiteralPath $_.FullName).Status -eq "Valid" } |
    Select-Object -First 1 -ExpandProperty FullName)
if (-not $signedDll) {
    throw "No Microsoft-signed DLL found under $env:WINDIR\System32 to use as a signed fixture."
}
Copy-Item -LiteralPath $signedDll -Destination (Join-Path $signedDir "signed.dll")
Copy-Item -LiteralPath $signedDll -Destination (Join-Path $mixedDir "signed.dll")

# Unsigned fixture: a freshly compiled managed DLL has no Authenticode signature.
$unsignedCs = @"
public static class GenieXUnsignedFixture {
    public static int Value() { return 42; }
}
"@
$unsignedDll = Join-Path $unsignedDir "unsigned.dll"
Add-Type -TypeDefinition $unsignedCs -OutputAssembly $unsignedDll -OutputType Library
Copy-Item -LiteralPath $unsignedDll -Destination (Join-Path $mixedDir "unsigned.dll")

function Invoke-Verify([string]$Target, [int]$ExpectedExit) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File $verify -Path $Target -Quiet
    $code = $LASTEXITCODE
    if ($code -ne $ExpectedExit) {
        throw "verifier exit $code for '$Target', expected $ExpectedExit"
    }
    Write-Host "PASS: '$Target' exited $code (expected $ExpectedExit)"
}

try {
    Invoke-Verify -Target $signedDir -ExpectedExit 0
    Invoke-Verify -Target $unsignedDir -ExpectedExit 1
    Invoke-Verify -Target $mixedDir -ExpectedExit 1

    # Single-file form must behave the same as its containing directory.
    Invoke-Verify -Target (Join-Path $signedDir "signed.dll") -ExpectedExit 0
    Invoke-Verify -Target (Join-Path $unsignedDir "unsigned.dll") -ExpectedExit 1

    Write-Host ""
    Write-Host "All assertions passed."
    exit 0
}
finally {
    Remove-Item -LiteralPath $work -Recurse -Force
}
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

<#
.SYNOPSIS
    Verifies the Authenticode signature status of Windows PE files.

.DESCRIPTION
    Recursively scans a path for Windows PE files (.exe / .dll) and reports
    each file's Authenticode signature status. Exits non-zero if any file
    does not carry the required status, so it can gate a release: Windows
    Smart App Control and SmartScreen block unsigned / invalid binaries
    (see qualcomm/GenieX issue #1398).

.PARAMETER Path
    Directory or single file to scan.

.PARAMETER Extension
    File extensions treated as PE files. Default: .exe, .dll.

.PARAMETER RequiredStatus
    Signature status every scanned file must have. Default: Valid.

.PARAMETER Quiet
    Only print a one-line summary, not per-file lines.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts/verify-windows-signatures.ps1 -Path sdk-windows-arm64
#>
param(
    [Parameter(Mandatory = $true)]
    [string]$Path,
    [string[]]$Extension = @('.exe', '.dll'),
    [ValidateSet('Valid', 'HashMismatch', 'NotSigned', 'UnknownError', 'NotTrusted', 'Incomplete')]
    [string]$RequiredStatus = 'Valid',
    [switch]$Quiet
)

$ErrorActionPreference = 'Stop'

function Test-IsPeFile {
    param([System.IO.FileInfo]$File)
    try {
        $bytes = [System.IO.File]::ReadAllBytes($File.FullName)
        # DOS stub
        if ($bytes.Length -lt 64 -or $bytes[0] -ne 0x4D -or $bytes[1] -ne 0x5A) {
            return $false
        }
        # PE header offset at 0x3C; 'PE\0\0'
        $peOffset = [BitConverter]::ToInt32($bytes, 0x3C)
        return ($peOffset -ge 0 -and ($peOffset + 4) -le $bytes.Length -and
                $bytes[$peOffset] -eq 0x50 -and $bytes[$peOffset + 1] -eq 0x45 -and
                $bytes[$peOffset + 2] -eq 0 -and $bytes[$peOffset + 3] -eq 0)
    }
    catch {
        return $false
    }
}

if (Test-Path -LiteralPath $Path -PathType Leaf) {
    $items = @(Get-Item -LiteralPath $Path)
}
else {
    $items = @(Get-ChildItem -LiteralPath $Path -Recurse -File | Where-Object { $_.Extension -in $Extension })
}

$peFiles = @($items | Where-Object { Test-IsPeFile -File $_ })
if ($peFiles.Count -eq 0) {
    Write-Host "verify-windows-signatures: no PE files found under '$Path'"
    exit 0
}

$failed = @()
foreach ($file in $peFiles) {
    $sig = Get-AuthenticodeSignature -LiteralPath $file.FullName
    $ok = ($sig.Status.ToString() -eq $RequiredStatus)
    if (-not $Quiet) {
        $mark = if ($ok) { 'OK  ' } else { 'FAIL' }
        Write-Host ("{0} {1,-14} {2}" -f $mark, $sig.Status, $file.FullName)
    }
    if (-not $ok) {
        $failed += $file.FullName
    }
}

if ($failed.Count -gt 0) {
    Write-Host ""
    Write-Host "verify-windows-signatures: FAILED - $($failed.Count) file(s) do not have a '$RequiredStatus' Authenticode signature:"
    foreach ($f in $failed) {
        Write-Host "  $f"
    }
    exit 1
}

Write-Host "verify-windows-signatures: OK - all $($peFiles.Count) PE file(s) have a '$RequiredStatus' Authenticode signature."
exit 0
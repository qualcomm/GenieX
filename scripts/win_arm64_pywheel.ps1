# This script is used to build the Python wheel for GenieX-Bridge on Windows ARM64

# Example:
# 
# .\scripts\win_arm64_pywheel.ps1 -Version 1.0.21rc14 -UploadToPyPI -PyPIToken "" -CleanBuild -PythonPaths "C:\Program Files\Python311-arm64\python.exe" "C:\Program Files\Python312-arm64\python.exe" "C:\Program Files\Python\Python313-arm64\python.exe"

# Powershell: 
# .\scripts\win_arm64_pywheel.ps1 -Version 1.0.27 -UploadToPyPI -PyPIToken "" -CleanBuild -PythonPaths @("C:\Users\dev\AppData\Local\Programs\Python\Python311-arm64\python.exe", "C:\Users\dev\AppData\Local\Programs\Python\Python312-arm64\python.exe", "C:\Users\dev\AppData\Local\Programs\Python\Python313-arm64\python.exe")

param(
    [string]$Version = "1.0.21rc14",
    [string]$BuildType = "Release",
    [int]$ParallelJobs = 16,
    [switch]$UploadToPyPI = $false,
    [string]$PyPIRepository = "pypi",
    [string]$PyPIToken = "",
    [switch]$CleanBuild = $false,
    [switch]$Debug = $false,
    [switch]$Verbose = $false,
    [string[]]$PythonPaths = @(
        "C:\Program Files\Python311-arm64\python.exe",
        "C:\Program Files\Python312-arm64\python.exe",
        "C:\Program Files\Python\Python313-arm64\python.exe"
    )
)

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $color = switch ($Level) { "ERROR" { "Red" } "WARNING" { "Yellow" } "SUCCESS" { "Green" } default { "White" } }
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] [$Level] $Message" -ForegroundColor $color
}

function New-CleanVenv {
    param([string]$PythonPath)
    $pythonVersion = & $PythonPath --version 2>&1 | ForEach-Object { $_.Split()[1].Split('.')[0, 1] -join '.' }
    $venvName = ".venv_py$pythonVersion"
    
    if (Test-Path $venvName) { Remove-Item -Recurse -Force $venvName -ErrorAction Stop }
    & $PythonPath -m venv $venvName
    if ($LASTEXITCODE -ne 0) { throw "Failed to create venv" }
    & ".\$venvName\Scripts\Activate.ps1"
    if ($LASTEXITCODE -ne 0) { throw "Failed to activate venv" }
    pip install --upgrade build setuptools wheel twine
}

function Import-VSEnv {
    Write-Log "Importing VS environment..."
    $envDump = cmd /c "`"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsarm64.bat`" && set"
    foreach ($line in $envDump) { if ($line -match "^(.*?)=(.*)$") { [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2]) } }
    # Tell Python's distutils to use the already-configured SDK
    [System.Environment]::SetEnvironmentVariable("DISTUTILS_USE_SDK", "1")
    [System.Environment]::SetEnvironmentVariable("MSSdk", "1")
}

function Initialize-Micromamba {
    Write-Log "Setting up micromamba..."
    $hookOutput = micromamba shell hook -s powershell | Out-String
    Invoke-Expression $hookOutput
    micromamba activate
    Write-Log "Micromamba activated" "SUCCESS"
}

function Publish-ToPyPI {
    param([string]$DistPath = "dist")
    
    Write-Log "Publishing to PyPI from: $DistPath"
    if (-not (Test-Path $DistPath)) { throw "Directory not found: $DistPath" }
    $wheelFiles = Get-ChildItem -Path $DistPath -Filter "*.whl"
    if ($wheelFiles.Count -eq 0) { throw "No wheel files found in $DistPath" }
    Write-Log "Found $($wheelFiles.Count) wheel file(s)" "INFO"
    if ([string]::IsNullOrEmpty($PyPIToken) -and [string]::IsNullOrEmpty($env:TWINE_PASSWORD)) { throw "PyPI token required" }
    if (-not (Get-Command "twine" -ErrorAction SilentlyContinue)) { pip install twine }
    $env:TWINE_USERNAME = "__token__"
    if (-not [string]::IsNullOrEmpty($PyPIToken)) {
        $env:TWINE_PASSWORD = $PyPIToken
    }
    else {
        $env:TWINE_PASSWORD = $env:TWINE_PASSWORD
    }
    $env:TWINE_REPOSITORY = $PyPIRepository
    $uploadArgs = @("upload", "--skip-existing", "$DistPath\*.whl")
    if ($Verbose) { $uploadArgs += "--verbose" }
    & twine @uploadArgs
    if ($LASTEXITCODE -ne 0) { throw "Twine upload failed" }
    Write-Log "Uploaded $($wheelFiles.Count) wheel file(s)" "SUCCESS"
}

function Main {
    Write-Log "Starting build process..." "SUCCESS"
    Write-Log "Version: $Version, Type: $BuildType, Jobs: $ParallelJobs, Python: $($PythonPaths.Count)" "INFO"
    if ($UploadToPyPI) { Write-Log "PyPI Upload: Enabled" "INFO" }
    
    try {
        if ($CleanBuild) {
            Write-Log "CleanBuild: Removing all .venv_py* directories and build directory..." "WARNING"
            Get-ChildItem -Path . -Filter ".venv_py*" -Directory | ForEach-Object { Remove-Item -Force -Recurse $_.FullName -ErrorAction SilentlyContinue }
            if (Test-Path "build") { Remove-Item -Recurse -Force "build" -ErrorAction SilentlyContinue }
        }
        Import-VSEnv
        Initialize-Micromamba
        if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }
        
        $successCount = 0
        foreach ($pythonPath in $PythonPaths) {
            if (Test-Path $pythonPath) {
                try {
                    Write-Log "Building for: $pythonPath" "INFO"
                    New-CleanVenv $pythonPath
                    if ($CleanBuild -and (Test-Path "build")) { Remove-Item -Recurse -Force "build" }
                    
                    # Extract base Python directory to find development files
                    $pythonDir = Split-Path $pythonPath -Parent
                    $pythonVersion = & $pythonPath --version 2>&1 | ForEach-Object { $_.Split()[1] }
                    
                    $cmakeArgs = @(
                        "-S", ".",
                        "-B", "build",
                        "-G", "Ninja",
                        "-DCMAKE_TOOLCHAIN_FILE=cmake/arm64-windows-llvm.cmake",
                        "-DPython_ROOT_DIR=$pythonDir",
                        "-DPython_EXECUTABLE=$pythonPath",
                        "-DGENIEX_PLUGIN_LLAMA_CPP=ON",
                        "-DGGML_OPENCL=ON",
                        "-DGGML_BACKEND_DL=OFF",
                        "-DGENIEX_PLUGIN_QNN=ON",
                        "-DGENIEX_BINDING_PYTHON=ON",
                        "-DGENIEX_PRODUCTION=OFF",
                        "-DGENIEX_VALIDATION=ON",
                        "-DGENIEX_DEBUG=$(if ($Debug) { "ON" } else { "OFF" })",
                        "-DGENIEX_TEST=OFF",
                        "-DGENIEX_BRIDGE_VERSION=$Version"
                    )
                    if ($Verbose) { $cmakeArgs += "-DCMAKE_VERBOSE_MAKEFILE=ON" }
                    & cmake @cmakeArgs
                    if ($LASTEXITCODE -ne 0) { throw "CMake configuration failed" }
                    
                    & cmake --build build -j $ParallelJobs
                    if ($LASTEXITCODE -ne 0) { throw "CMake build failed" }
                    
                    Write-Log "SUCCESS: Build completed for $pythonPath" "SUCCESS"
                    $successCount++
                    
                    if (-not (Test-Path "dist")) { New-Item -ItemType Directory -Path "dist" }
                    $wheelFiles = Get-ChildItem -Path ".\build\binding_python\dist\*.whl"
                    foreach ($wheel in $wheelFiles) {
                        $destPath = ".\dist\$($wheel.Name)"
                        Move-Item -Path $wheel.FullName -Destination $destPath -Force
                        Write-Log "Moved wheel: $($wheel.Name)" "INFO"
                    }
                }
                catch {
                    Write-Log "ERROR: Build failed for $pythonPath - $($_.Exception.Message)" "ERROR"
                }
                finally {
                    if ($env:VIRTUAL_ENV) { deactivate }
                }
            }
            else {
                Write-Log "SKIP: Python not found at $pythonPath" "WARNING"
            }
        }
        
        Write-Log "Multi-Python build completed. Success: $successCount/$($PythonPaths.Count)" "SUCCESS"
        if ($UploadToPyPI -and $successCount -gt 0) { Publish-ToPyPI -DistPath "dist" }
    }
    catch {
        Write-Log "Build process failed: $($_.Exception.Message)" "ERROR"
        exit 1
    }
}

Main
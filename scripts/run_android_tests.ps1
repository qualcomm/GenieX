#=============================================================================
#
#  Copyright (c) 2025 Qualcomm
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm
#
#=============================================================================

param(
    [string]$Abi = "arm64-v8a",
    [string]$BuildDir = "build-android",
    [ValidateSet("test", "app")]
    [string]$Mode = "test",
    [switch]$Debug,
    [switch]$Help
)

# Show help message
if ($Help) {
    Write-Host "Usage: .\run_android_tests.ps1 [options]"
    Write-Host "Options:"
    Write-Host "  -Abi <abi>        Target ABI (arm64-v8a). Default: arm64-v8a"
    Write-Host "  -BuildDir <dir>   Build directory. Default: build-android"
    Write-Host "  -Mode <mode>      Build mode (test, app). Default: test"
    Write-Host "                    test: plugins loaded dynamically (for unit testing)"
    Write-Host "                    app:  plugins statically linked (for Android app)"
    Write-Host "  -Debug            Build debug version. Default: Release"
    Write-Host "  -Help             Show this help message"
    exit 0
}

# Create a log directory (use absolute path to avoid issues when changing directories)
$ScriptDir = $PSScriptRoot
if (-not $ScriptDir) { $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path }
$RootDir = Split-Path -Parent $ScriptDir
$LogDir = Join-Path $RootDir "build_logs"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogDir "android_build_$Timestamp.log"

# Function to write to both console and log file
function Write-Log {
    param([string]$Message)
    Write-Host $Message
    Add-Content -Path $LogFile -Value $Message
}

Write-Log "Build started at $(Get-Date)"
Write-Log "=============================================="

# Record environment information
Write-Log "Environment Information:"
Write-Log "- PowerShell version: $($PSVersionTable.PSVersion)"
Write-Log "- OS: $([System.Environment]::OSVersion.VersionString)"

try {
    $cmakeVersion = cmake --version | Select-Object -First 1
    Write-Log "- CMake version: $cmakeVersion"
} catch {
    Write-Log "- CMake version: Could not determine"
}

Write-Log "- Environment variables:"
Get-ChildItem Env: | Where-Object { $_.Name -match "ANDROID|PATH|HOME|CMAKE" } | Sort-Object Name | ForEach-Object {
    Write-Log "  $($_.Name)=$($_.Value)"
}

# Check if NDK path is set
if (-not $env:ANDROID_NDK_ROOT) {
    Write-Log "ERROR: ANDROID_NDK_ROOT environment variable is not set."
    Write-Log "Please set it to your Android NDK installation path, for example:"
    Write-Log '  $env:ANDROID_NDK_ROOT = "C:\path\to\android-ndk"'
    exit 1
}

Write-Log "- Android NDK path: $env:ANDROID_NDK_ROOT"

$ndkPropertiesFile = Join-Path $env:ANDROID_NDK_ROOT "source.properties"
if (Test-Path $ndkPropertiesFile) {
    $ndkVersion = Get-Content $ndkPropertiesFile | Where-Object { $_ -match "Pkg.Revision" }
    Write-Log "- Android NDK version: $ndkVersion"
} else {
    Write-Log "- Android NDK version: Could not determine NDK version"
}
Write-Log "=============================================="

# Set build type
$BuildType = if ($Debug) { "Debug" } else { "Release" }

# Validate ABI
$ValidAbis = @("arm64-v8a")
if ($Abi -notin $ValidAbis) {
    Write-Log "ERROR: Invalid ABI specified: $Abi"
    Write-Log "Valid ABIs are: $($ValidAbis -join ', ')"
    exit 1
}

Write-Log "Build configuration:"
Write-Log "- ABI: $Abi"
Write-Log "- Build type: $BuildType"
Write-Log "- Build mode: $Mode"
Write-Log "- Build directory: $BuildDir"
Write-Log "=============================================="

# Create build directory
if (-not (Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir | Out-Null
}

# Change to build directory
Push-Location $BuildDir
try {
    # Configure with CMake
    Write-Log "Configuring for Android with ABI: $Abi"

    $toolchainFile = Join-Path $env:ANDROID_NDK_ROOT "build\cmake\android.toolchain.cmake"
    
    # Note: Use -DGENIEX_PLUGIN_LLAMA_CPP=OFF -DGENIEX_DL=ON -DGENIEX_DEBUG=ON for unit testing (plugins loaded dynamically)
    #       Use -DGENIEX_PLUGIN_LLAMA_CPP=ON -DGENIEX_DL=OFF -DGENIEX_DEBUG=OFF for Android App build (plugins statically linked)
    
    # Set mode-specific flags
    if ($Mode -eq "test") {
        $llamaCpp = "OFF"
        $geniexDl = "ON"
        $geniexDebug = "ON"
        $geniexTest = "ON"
    } else {
        # app mode
        $llamaCpp = "ON"
        $geniexDl = "OFF"
        $geniexDebug = "OFF"
        $geniexTest = "OFF"
    }

    $cmakeArgs = @(
        "-DCMAKE_TOOLCHAIN_FILE=$toolchainFile",
        "-DANDROID_ABI=$Abi",
        "-DANDROID_PLATFORM=android-23",
        "-DANDROID_STL=c++_static",
        "-DCMAKE_BUILD_TYPE=$BuildType",
        "-DCMAKE_VERBOSE_MAKEFILE=ON",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        "-DCMAKE_CXX_FLAGS=-Wno-error=unused-function -Wno-error=unused-local-typedef -Wno-error=for-loop-analysis -Wno-error",
        "-DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined",
        "-DGENIEX_PLUGIN_QNN=ON",
        "-DGENIEX_PLUGIN_LLAMA_CPP=$llamaCpp",
        "-DGENIEX_DL=$geniexDl",
        "-DGENIEX_DEBUG=$geniexDebug",
        "-DGENIEX_TEST=$geniexTest",
        "-DGENIEX_BINDING_PYTHON=OFF",
        ".."
    )

    Write-Log "Running: cmake $($cmakeArgs -join ' ')"
    $cmakeOutput = & cmake @cmakeArgs 2>&1
    $cmakeOutput | ForEach-Object { Write-Log $_ }
    $cmakeResult = $LASTEXITCODE

    if ($cmakeResult -ne 0) {
        Write-Log "=============================================="
        Write-Log "CMake configuration failed with error code $cmakeResult"
        Write-Log "Please check the log file: $LogFile"
        exit $cmakeResult
    }

    # Build the QNN plugin
    Write-Log "=============================================="
    Write-Log "Building GENIEX_PLUGIN_QNN for Android..."

    # Get number of processors for parallel build
    $numProcs = $env:NUMBER_OF_PROCESSORS
    if (-not $numProcs) { $numProcs = 4 }

    $buildArgs = @("--build", ".", "-j$numProcs", "--verbose")
    Write-Log "Running: cmake $($buildArgs -join ' ')"
    $buildOutput = & cmake @buildArgs 2>&1
    $buildOutput | ForEach-Object { Write-Log $_ }
    $buildResult = $LASTEXITCODE

    if ($buildResult -ne 0) {
        Write-Log "=============================================="
        Write-Log "Build failed with error code $buildResult"
        Write-Log "Please check the log file: $LogFile"
        
        Write-Log "Checking for common errors in the build log..."
        $errorSummaryFile = Join-Path $LogDir "error_summary_$Timestamp.txt"
        Get-Content $LogFile | Select-String -Pattern "error:|undefined reference|no such file|cannot find" | Set-Content $errorSummaryFile
        Write-Log "Error summary saved to: $errorSummaryFile"

        exit $buildResult
    }

    Write-Log "=============================================="
    Write-Log "Build completed successfully at $(Get-Date)"
    Write-Log "Build artifacts are located at:"
    Write-Log "  - Bridge Library: $BuildDir/out/libgeniex_bridge.so"
    Write-Log "  - QNN Plugin: $BuildDir/out/npu/libgeniex_plugin.so"
    Write-Log "  - QNN Runtime: $BuildDir/out/npu/htp-files/"
    Write-Log "  - Test Executables:"
    Write-Log "    * $BuildDir/out/tests/geniex_test_llm (liquid)"
    Write-Log "    * $BuildDir/out/tests/geniex_test_vlm (omni-neural)"
    Write-Log "    * $BuildDir/out/tests/geniex_test_asr (parakeet)"
    Write-Log "    * $BuildDir/out/tests/geniex_test_cv (paddleocr)"
    Write-Log "    * $BuildDir/out/tests/geniex_test_embedding (embed-gemma, embed-neural)"
    Write-Log "    * $BuildDir/out/tests/geniex_test_rerank (jina-rerank)"
    Write-Log ""
    Write-Log "Full build log saved to: $LogFile"
    Write-Log ""

} finally {
    Pop-Location
}

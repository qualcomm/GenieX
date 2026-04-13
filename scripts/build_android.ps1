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
    [switch]$Debug,
    [ValidateSet("test", "app")]
    [string]$Mode = "app",
    [switch]$NoHexagon,
    [switch]$Help
)

# Show help if requested
if ($Help) {
    Write-Host "Usage: .\build_android.ps1 [options]"
    Write-Host "Options:"
    Write-Host "  -Abi <abi>        Target ABI (arm64-v8a). Default: arm64-v8a"
    Write-Host "  -BuildDir <dir>   Build directory. Default: build-android"
    Write-Host "  -Debug            Build debug version. Default: Release"
    Write-Host "  -Mode <mode>      Build mode: 'test' for unit testing, 'app' for Android app. Default: app"
    Write-Host "  -NoHexagon        Disable Hexagon DSP support (for builds without Hexagon SDK)"
    Write-Host "  -Help             Show this help message"
    exit 0
}

# Create log directory (use absolute paths to avoid issues after Push-Location)
$ScriptRoot = $PSScriptRoot
$WorkspaceRoot = Split-Path $ScriptRoot -Parent
$LogDir = Join-Path $WorkspaceRoot "$BuildDir/logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogDir "android_build_$Timestamp.log"

# Start transcript for comprehensive logging (captures all console output)
Start-Transcript -Path $LogFile -Append | Out-Null
$script:TranscriptActive = $true

# Function to log and display output (transcript captures this automatically)
function Write-Log {
    param([string]$Message)
    Write-Host $Message
}

function Stop-TranscriptSafe {
    if ($script:TranscriptActive) {
        Stop-Transcript | Out-Null
        $script:TranscriptActive = $false
    }
}

Write-Log "Build started at $(Get-Date)"
Write-Log "=============================================="

# Record environment information
Write-Log "Environment Information:"
Write-Log "- PowerShell version: $($PSVersionTable.PSVersion)"
Write-Log "- OS: $([System.Environment]::OSVersion.VersionString)"

try {
    $cmakeVersion = (cmake --version | Select-Object -First 1)
    Write-Log "- CMake version: $cmakeVersion"
} catch {
    Write-Log "- CMake version: Could not determine"
}

Write-Log "- Environment variables:"
Get-ChildItem Env: | Where-Object { $_.Name -match "ANDROID|PATH|HOME|CMAKE|HEXAGON" } | Sort-Object Name | ForEach-Object {
    Write-Log "    $($_.Name)=$($_.Value)"
}

# Check if NDK path is set
if (-not $env:ANDROID_NDK_ROOT) {
    Write-Log "ERROR: ANDROID_NDK_ROOT environment variable is not set."
    Write-Log "Please set it to your Android NDK installation path, for example:"
    Write-Log "  `$env:ANDROID_NDK_ROOT = 'C:\path\to\android-ndk'"
    Stop-TranscriptSafe
    exit 1
}

Write-Log "- Android NDK path: $env:ANDROID_NDK_ROOT"

$ndkPropertiesFile = Join-Path $env:ANDROID_NDK_ROOT "source.properties"
if (Test-Path $ndkPropertiesFile) {
    $ndkVersion = (Get-Content $ndkPropertiesFile | Select-String "Pkg.Revision").ToString()
    Write-Log "- Android NDK version: $ndkVersion"
} else {
    Write-Log "- Android NDK version: Could not determine NDK version"
}

Write-Log "=============================================="

# Set build variables
$BuildType = if ($Debug) { "Debug" } else { "Release" }
$HexagonEnabled = if ($NoHexagon) { "OFF" } else { "ON" }

# Validate ABI
$ValidAbis = @("arm64-v8a")
if ($Abi -notin $ValidAbis) {
    Write-Log "ERROR: Invalid ABI specified: $Abi"
    Write-Log "Valid ABIs are: $($ValidAbis -join ', ')"
    Stop-TranscriptSafe
    exit 1
}

Write-Log "Build configuration:"
Write-Log "- ABI: $Abi"
Write-Log "- Build type: $BuildType"
Write-Log "- Build mode: $Mode"
Write-Log "- Build directory: $BuildDir"
Write-Log "- Hexagon: $HexagonEnabled"
Write-Log "=============================================="

# Create build directory
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

# Change to build directory
Push-Location $BuildDir

try {
    # Configure with CMake
    Write-Log "Configuring for Android with ABI: $Abi"

    # Configure build options based on mode
    if ($Mode -eq "test") {
        Write-Log "Configuring for unit testing (dynamic plugin loading)..."
        $DlOption = "ON"
        $DebugOption = "ON"
        $TestOption = "ON"
    } else {
        Write-Log "Configuring for Android app (static linking)..."
        $DlOption = "OFF"
        $DebugOption = "OFF"
        $TestOption = "OFF"
    }

    # Build CMake arguments (use Ninja on Windows since make is not available)
    $cmakeArgs = @(
        "-G", "Ninja",
        "-DCMAKE_TOOLCHAIN_FILE=$env:ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake",
        "-DANDROID_ABI=$Abi",
        "-DANDROID_PLATFORM=android-23",
        "-DANDROID_STL=c++_static",
        "-DCMAKE_BUILD_TYPE=$BuildType",
        "-DCMAKE_VERBOSE_MAKEFILE=ON",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        "-DCMAKE_CXX_FLAGS=-Wno-error=unused-function -Wno-error=unused-local-typedef -Wno-error=for-loop-analysis -Wno-error",
        "-DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined",
        "-DGENIEX_PLUGIN_QNN=ON",
        "-DGENIEX_PLUGIN_LLAMA_CPP=ON",
        "-DGGML_OPENCL=ON",
        "-DGGML_HEXAGON=$HexagonEnabled",
        "-DPREBUILT_LIB_DIR=android_aarch64",
        "-DGENIEX_DL=$DlOption",
        "-DGENIEX_DEBUG=$DebugOption",
        "-DGENIEX_TEST=$TestOption",
        "-DGENIEX_BINDING_PYTHON=OFF",
        ".."
    )

    # Add Hexagon SDK root if enabled
    if ($HexagonEnabled -eq "ON" -and $env:HEXAGON_SDK_ROOT) {
        $cmakeArgs = $cmakeArgs[0..($cmakeArgs.Length - 2)] + @("-DHEXAGON_SDK_ROOT=$env:HEXAGON_SDK_ROOT") + @("..")
    }

    Write-Log "Running CMake with arguments:"
    Write-Log ($cmakeArgs -join " ")

    & cmake @cmakeArgs 2>&1 | ForEach-Object { Write-Host $_ }
    $cmakeResult = $LASTEXITCODE

    if ($cmakeResult -ne 0) {
        Write-Log "=============================================="
        Write-Log "CMake configuration failed with error code $cmakeResult"
        Write-Log "Please check the log file: $LogFile"
        Stop-TranscriptSafe
        exit $cmakeResult
    }

    # Build the QNN plugin
    Write-Log "=============================================="
    Write-Log "Building GENIEX_PLUGIN_QNN for Android..."

    # Get number of processors for parallel build
    $numProcs = [Environment]::ProcessorCount

    & cmake --build . -j $numProcs --config Release --verbose 2>&1 | ForEach-Object { Write-Host $_ }
    $buildResult = $LASTEXITCODE

    if ($buildResult -ne 0) {
        Write-Log "=============================================="
        Write-Log "Build failed with error code $buildResult"
        Write-Log "Please check the log file: $LogFile"
        Write-Log "Checking for common errors in the build log..."
        
        # Stop transcript first so we can read the log file
        Stop-TranscriptSafe
        
        $errorSummaryFile = "$LogDir/error_summary_$Timestamp.txt"
        Get-Content $LogFile | Select-String -Pattern "error:|undefined reference|no such file|cannot find" | Out-File $errorSummaryFile
        Write-Host "Error summary saved to: $errorSummaryFile"

        exit $buildResult
    }

    # Copy libomp.so from NDK — required at runtime by plugins using OpenMP (qnn)
    $libompSrc = Get-ChildItem -Recurse -Filter "libomp.so" -Path $env:ANDROID_NDK_ROOT |
        Where-Object { $_.FullName -match "lib[\\/]linux[\\/]aarch64" } |
        Select-Object -First 1
    if ($libompSrc) {
        $libompDest = "out/common/lib"
        New-Item -ItemType Directory -Force -Path $libompDest | Out-Null
        Copy-Item $libompSrc.FullName -Destination "$libompDest/libomp.so"
        Write-Log "Copied libomp.so from NDK: $($libompSrc.FullName) -> $libompDest/libomp.so"
    } else {
        Write-Log "WARNING: libomp.so not found in ANDROID_NDK_ROOT. Plugins using OpenMP may fail at runtime."
    }

    Write-Log "=============================================="
    Write-Log "Build completed successfully at $(Get-Date)"
    Write-Log "Build mode: $Mode"
    Write-Log "Build artifacts are located at:"
    Write-Log "  - Bridge Library: $BuildDir/out/libgeniex_bridge.so"
    Write-Log "  - QNN Plugin: $BuildDir/out/npu/libgeniex_plugin.so"
    Write-Log "  - QNN Runtime: $BuildDir/out/npu/htp-files/"

    if ($Mode -eq "test") {
        Write-Log "  - Test Executables:"
        Write-Log "    * $BuildDir/out/tests/geniex_test_llm (liquid, granite-nano, granite4)"
        Write-Log "    * $BuildDir/out/tests/geniex_test_vlm (omni-neural)"
        Write-Log "    * $BuildDir/out/tests/geniex_test_asr (parakeet)"
        Write-Log "    * $BuildDir/out/tests/geniex_test_cv (paddleocr)"
        Write-Log "    * $BuildDir/out/tests/geniex_test_embedding (embed-gemma, embedneural)"
        Write-Log "    * $BuildDir/out/tests/geniex_test_rerank (jina-rerank)"
        Write-Log ""
    }

    Write-Log "Full build log saved to: $LogFile"

} finally {
    # Return to original directory
    Pop-Location
    Stop-TranscriptSafe
}

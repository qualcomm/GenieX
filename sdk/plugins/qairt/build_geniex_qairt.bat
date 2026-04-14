@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "CMAKELISTS=%~f1"
set "TOOLCHAIN_FILE=%~f2"
set "OUT_DIR=%~3"

for %%I in ("%CMAKELISTS%") do set "QAIRT_SRC=%%~dpI"
if "%QAIRT_SRC:~-1%"=="\" set "QAIRT_SRC=%QAIRT_SRC:~0,-1%"

set "SHORT_ROOT=%TEMP%\geniex_qairt"
set "BUILD_DIR=%SHORT_ROOT%\build"
set "VS_ROOT=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools"
set "CMAKE_EXE=%VS_ROOT%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "NINJA_EXE=%VS_ROOT%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
set "LLVM_BIN=%VS_ROOT%\VC\Tools\Llvm\x64\bin"
if not defined USERPROFILE set "USERPROFILE=%HOMEDRIVE%%HOMEPATH%"
if not defined USERPROFILE set "USERPROFILE=C:\Users\mengshen"
set "CARGO_BIN=%USERPROFILE%\.cargo\bin"
set "CARGO_EXE=%CARGO_BIN%\cargo.exe"
set "HOME=%USERPROFILE%"
set "CARGO_HOME=%USERPROFILE%\.cargo"
set "PATH=%CARGO_BIN%;%LLVM_BIN%;%VS_ROOT%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"

if not exist "%CARGO_EXE%" set "CARGO_EXE=C:\Users\mengshen\.cargo\bin\cargo.exe"

if exist "%SHORT_ROOT%" rmdir /S /Q "%SHORT_ROOT%"
mkdir "%SHORT_ROOT%"
if errorlevel 1 exit /b 1

set "TMP=%SHORT_ROOT%\tmp"
set "TEMP=%SHORT_ROOT%\tmp"
mkdir "%TMP%"
if errorlevel 1 exit /b 1

if not exist "%CARGO_EXE%" (
echo cargo.exe not found at %CARGO_EXE% >&2
exit /b 1
)

"%CMAKE_EXE%" -S "%QAIRT_SRC%" -B "%BUILD_DIR%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=%TOOLCHAIN_FILE% -DCMAKE_MAKE_PROGRAM="%NINJA_EXE%" -DCARGO_EXECUTABLE="%CARGO_EXE%" -DTOKENIZERS_CPP_CARGO_TARGET=aarch64-pc-windows-msvc -DGENIEX_BUILD_VLM=OFF -DGENIEX_BUILD_AUDIO=OFF -DGENIEX_BUILD_OCR=OFF -DBUILD_EXAMPLES=OFF
if errorlevel 1 exit /b 1

"%CMAKE_EXE%" --build "%BUILD_DIR%" --config Release --target geniex_core
if errorlevel 1 exit /b 1

copy /Y "%BUILD_DIR%\bin\geniex_core.dll" "%OUT_DIR%\geniex_core.dll" >NUL
if errorlevel 1 exit /b 1
copy /Y "%BUILD_DIR%\geniex_core.lib" "%OUT_DIR%\geniex_core.lib" >NUL
if errorlevel 1 exit /b 1

rmdir /S /Q "%SHORT_ROOT%"

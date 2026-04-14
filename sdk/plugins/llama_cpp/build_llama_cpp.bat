@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "CMAKELISTS=%~f1"
set "TOOLCHAIN_FILE=%~f2"
set "OUT_DIR=%~3"

for %%I in ("%CMAKELISTS%") do set "LLAMA_SRC=%%~dpI"
if "%LLAMA_SRC:~-1%"=="\" set "LLAMA_SRC=%LLAMA_SRC:~0,-1%"

set "BUILD_DIR=%OUT_DIR%\cmake_build"
set "VS_ROOT=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools"
set "CMAKE_EXE=%VS_ROOT%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "NINJA_EXE=%VS_ROOT%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
set "LLVM_BIN=%VS_ROOT%\VC\Tools\Llvm\x64\bin"
set "PATH=%LLVM_BIN%;%VS_ROOT%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"

if exist "%BUILD_DIR%" rmdir /S /Q "%BUILD_DIR%"

"%CMAKE_EXE%" -S "%LLAMA_SRC%" -B "%BUILD_DIR%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=%TOOLCHAIN_FILE% -DCMAKE_MAKE_PROGRAM="%NINJA_EXE%" -DGGML_BLAS=OFF -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_WEBUI=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DBUILD_SHARED_LIBS=ON
if errorlevel 1 exit /b 1

"%CMAKE_EXE%" --build "%BUILD_DIR%" --config Release --target common ggml-base ggml ggml-cpu llama mtmd
if errorlevel 1 exit /b 1

copy /Y "%BUILD_DIR%\common\common.lib" "%OUT_DIR%\common.lib" >NUL
if errorlevel 1 exit /b 1
copy /Y "%BUILD_DIR%\vendor\cpp-httplib\cpp-httplib.lib" "%OUT_DIR%\cpp-httplib.lib" >NUL
if errorlevel 1 exit /b 1
copy /Y "%BUILD_DIR%\bin\ggml-base.dll" "%OUT_DIR%\ggml-base.dll" >NUL
if errorlevel 1 exit /b 1
copy /Y "%BUILD_DIR%\ggml\src\ggml-base.lib" "%OUT_DIR%\ggml-base.lib" >NUL
if errorlevel 1 exit /b 1
copy /Y "%BUILD_DIR%\bin\ggml-cpu.dll" "%OUT_DIR%\ggml-cpu.dll" >NUL
if errorlevel 1 exit /b 1
copy /Y "%BUILD_DIR%\ggml\src\ggml-cpu.lib" "%OUT_DIR%\ggml-cpu.lib" >NUL
if errorlevel 1 exit /b 1
copy /Y "%BUILD_DIR%\bin\ggml.dll" "%OUT_DIR%\ggml.dll" >NUL
if errorlevel 1 exit /b 1
copy /Y "%BUILD_DIR%\ggml\src\ggml.lib" "%OUT_DIR%\ggml.lib" >NUL
if errorlevel 1 exit /b 1
copy /Y "%BUILD_DIR%\bin\llama.dll" "%OUT_DIR%\llama.dll" >NUL
if errorlevel 1 exit /b 1
copy /Y "%BUILD_DIR%\src\llama.lib" "%OUT_DIR%\llama.lib" >NUL
if errorlevel 1 exit /b 1
copy /Y "%BUILD_DIR%\bin\mtmd.dll" "%OUT_DIR%\mtmd.dll" >NUL
if errorlevel 1 exit /b 1
copy /Y "%BUILD_DIR%\tools\mtmd\mtmd.lib" "%OUT_DIR%\mtmd.lib" >NUL
if errorlevel 1 exit /b 1

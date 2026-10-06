# Experimental Windows x64 CUDA SDK

This source-only configuration enables NVIDIA inference through the existing
llama.cpp backend. It builds the C SDK and Python bindings. It does not provide
Windows x64 release artifacts, QAIRT support, Intel NPU support, or a validated
Go CLI/server build. Snapdragon remains the supported packaged configuration.

## Prerequisites

- Windows x64 and a CUDA-capable NVIDIA GPU with a compatible driver.
- Visual Studio 2022 C++ Build Tools, including the Windows SDK.
- Rust with the `x86_64-pc-windows-msvc` toolchain.
- CUDA Toolkit (tested with 13.0.2), including nvcc, runtime, cuBLAS and headers.
- CMake (tested with 4.4.3), Ninja and Python 3.10+.

Run the commands below in an **x64 Native Tools Command Prompt for VS 2022**
or its Developer PowerShell equivalent, with Rust, CMake, Ninja and CUDA tools
on PATH. The CUDA runtime DLL directory must also be on PATH; CUDA 13 uses
`%CUDA_PATH%\bin\x64`. Set `CUDAToolkit_ROOT` if CMake cannot locate the toolkit.

From the repository root:

```powershell
git submodule update --init third-party/llama.cpp
cmake --preset x64-windows-cuda-release -S sdk
cmake --build sdk/build-x64-windows-cuda-release --parallel 8
cmake --install sdk/build-x64-windows-cuda-release --prefix sdk/pkg-geniex

# CUDA 13 runtime dependencies for this local development install:
foreach ($pattern in @('cudart64_*.dll', 'cublas64_*.dll', 'cublasLt64_*.dll')) {
    Copy-Item "$env:CUDA_PATH\bin\x64\$pattern" sdk/pkg-geniex/lib/llama_cpp
}
```

The preset targets the local GPU. For a Blackwell sm_120 build, the tested
explicit architecture override is `-DCMAKE_CUDA_ARCHITECTURES=120a-real`.
Use a separate build directory when switching backends or architectures.

## Python

Use an isolated Python environment, then install against the local SDK:

```powershell
$env:GENIEX_SKIP_SDK_DOWNLOAD = '1'
python -m pip install -e bindings/python
$env:GENIEX_LIB_PATH = (Resolve-Path sdk/pkg-geniex/lib).Path
geniex-py devices
```

Expect `CUDA0` plus `CPU` under `llama_cpp`. In this build, `auto` and `gpu`
select CUDA0; `cpu` keeps zero GPU layers. Explicit device selection such as
`llama_cpp:CUDA1` is available for other devices. Device-list parsing is tested,
but multi-GPU inference has not been validated. QAIRT still resolves to its NPU.
Builds without `GGML_CUDA` retain the existing Snapdragon alias behavior.

If Windows reports a missing DLL, make the CUDA runtime and cuBLAS DLLs
discoverable before loading GenieX. For a local development install, copy
`cudart64_*.dll`, `cublas64_*.dll` and `cublasLt64_*.dll` from the toolkit's
runtime directory into `sdk/pkg-geniex/lib/llama_cpp`. No CUDA runtime DLLs
or model weights are included in the source contribution.

```python
from geniex import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "path/to/model.gguf",
    device_map="gpu",
    n_ctx=4096,
    n_batch=512,
    n_ubatch=256,
    # For vision models, also supply mmproj_path="path/to/mmproj.gguf".
)
try:
    prompt = model.tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 17 times 19?"}],
        add_generation_prompt=True,
    )
    print(model.generate(prompt, max_new_tokens=128).text)
finally:
    model.close()
```

## Validation

Model-free API tests against the built CUDA SDK:

```powershell
python -m pip install pytest
$env:GENIEX_TEST_CUDA_BUILD = '1'
python -m pytest tests/test_api.py
```

Leave `GENIEX_TEST_CUDA_BUILD` unset when testing Snapdragon builds. It tells
the alias tests which compile-time default to expect, even when a CUDA build
is tested on a host with no enumerated GPU.

Manually validated on an RTX PRO 2000 Blackwell Laptop GPU (8 GB), Intel x64
CPU, Windows 11, CUDA 13.0.2 and driver 596.98, using Qwen3.5-4B Q4_K_M and
its F16 vision projector. Text generation, image understanding and two-turn
chat succeeded. Logs reported 33/33 layers offloaded and the vision encoder
using CUDA0; concurrent `nvidia-smi` sampling reported up to 98% utilization
and approximately 4 GiB VRAM. Short runs are functional smoke tests, not a
cross-device performance claim. Other GPUs, mixed CUDA/OpenCL builds and
multi-GPU inference remain unvalidated.

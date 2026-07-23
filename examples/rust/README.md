# GenieX Rust Binding Example

This directory contains a functional example demonstrating how to use the GenieX Rust bindings (`geniex` crate).

## Prerequisites

1. Build and install the GenieX C SDK (`geniex.dll` / `geniex.lib` / plugins):
   ```powershell
   cd ../../sdk
   cmake -B build -DGENIEX_PLUGIN_QAIRT=OFF -DGENIEX_PLUGIN_LLAMA_CPP=ON -DGGML_OPENCL=OFF -DGGML_HEXAGON=OFF
   cmake --build build -j --config Release
   cmake --install build --prefix pkg-geniex
   ```

2. (Optional) If running outside Cargo build automation, add DLL paths to your environment (Windows PowerShell):
   ```powershell
   $env:PATH="$env:PATH;$PWD\..\..\sdk\pkg-geniex\lib;$PWD\..\..\sdk\pkg-geniex\lib\llama_cpp"
   ```

## Running the Example

### 1. Basic SDK Status & Discovery Check

Run without parameters to test runtime initialization, plugin discovery, and device resolution:

```bash
cargo run
```

### 2. LLM Model Inference

Pass the path to a GGUF model file to run text generation:

```bash
cargo run -- path/to/model.gguf
```

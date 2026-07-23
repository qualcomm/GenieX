# GenieX Rust Bindings — Windows ARM64 Guide

This document provides a comprehensive guide to building, testing, and using the GenieX Rust bindings (`geniex` crate) on **Windows ARM64** (`aarch64-pc-windows-msvc`).

---

## 1. Prerequisites & System Requirements

To build and run the Rust bindings on Windows ARM64, the following tools are required:

1. **Rust Toolchain**:
   - Target: `aarch64-pc-windows-msvc`
   - Edition: Rust 2021 (e.g., `rustc 1.97+`)
2. **LLVM / Clang**:
   - Required by `bindgen` to parse the C ABI header [`sdk/include/geniex.h`](../../../sdk/include/geniex.h).
3. **C/C++ Compiler & Build System**:
   - Visual Studio Build Tools / MSVC ARM64 + CMake (3.16+) + Ninja.
4. **Git**:
   - For fetching third-party git submodules (`llama.cpp`, `geniex-qairt`).

---

## 2. Step 1: Building the GenieX C SDK

The Rust binding dynamically links to `geniex.dll` and links against `geniex.lib` at build time.

### 2.1 Fetch Git Submodules
From the root of the GenieX repository:
```powershell
git submodule update --init --recursive
```

### 2.2 Compile the Native C/C++ SDK
Navigate to the `sdk/` directory:
```powershell
cd sdk

# Option A: Standard CPU / llama.cpp build (no Qualcomm proprietary SDKs required)
cmake -B build -DGENIEX_PLUGIN_QAIRT=OFF -DGENIEX_PLUGIN_LLAMA_CPP=ON -DGGML_OPENCL=OFF -DGGML_HEXAGON=OFF

# Option B: Full Snapdragon build (Hexagon NPU / QAIRT) if Qualcomm SDKs are installed
# cmake --preset arm64-windows-snapdragon-release -B build

# Build (Release configuration)
cmake --build build -j --config Release

# Install native artifacts to sdk/pkg-geniex/
cmake --install build --prefix pkg-geniex
```

> [!NOTE]
> Ensure that `geniex.lib` is present in `sdk/pkg-geniex/lib/` alongside `geniex.dll`. If missing from the install prefix, copy it from `sdk/build/src/geniex.lib`.

---

## 3. Step 2: Configuring the Rust Crate

The Rust build script [`build.rs`](../build.rs) locates the native C library using two lookup mechanisms:
1. Via the environment variable `CARGO_GENIEX_LIB_DIR` (if set).
2. Automatically searching default relative paths: `../../sdk/pkg-geniex/lib` and `../../sdk/build/src`.

### Environment Variables for Windows ARM64 Execution

In PowerShell:
```powershell
# Specify path to geniex.lib if located outside default lookup paths
$env:CARGO_GENIEX_LIB_DIR="C:\path\to\GenieX\sdk\pkg-geniex\lib"

# (Optional) Add DLL locations to system PATH if running custom binaries outside cargo
$env:PATH="$env:PATH;C:\path\to\GenieX\sdk\pkg-geniex\lib;C:\path\to\GenieX\sdk\pkg-geniex\lib\llama_cpp"
```

---

## 4. Step 3: Compiling and Running Rust Tests

Navigate to `bindings/rust`:
```powershell
cd bindings/rust

cargo build

cargo test -- --nocapture
```

### Test Suite Coverage
The integration test suite validates:
- `test_version`: Verifies GenieX SDK version retrieval.
- `test_config_defaults`: Ensures default configuration values (`ModelConfig`, `SamplerConfig`, `GenerationConfig`).
- `test_chat_message`: Validates chat message data structures.
- `test_error`: Verifies C error code mapping to `GeniexError`.
- `test_init_and_plugins`: Tests SDK initialization (`geniex_init`), dynamic scanning/loading of `llama_cpp` (`geniex_plugin.dll`), and teardown (`geniex_deinit`).
- `test_resolve_device`: Validates compute-unit resolution (`geniex_resolve_device`).

---

## 5. Step 4: Using the Rust Bindings in Your Project

### 5.1 Add Cargo Dependency
In your project's `Cargo.toml`:
```toml
[dependencies]
geniex = { path = "path/to/GenieX/bindings/rust" }
```

### 5.2 Example 1: SDK Initialization & Plugin Discovery
```rust
use geniex::*;

fn main() -> Result<()> {
    init()?;
    println!("GenieX Version: {}", version());

    let plugins = get_plugin_list()?;
    println!("Installed plugins: {:?}", plugins);

    let resolve_input = ResolveDeviceInput {
        plugin_id: "llama_cpp".to_string(),
        model_name: Some("model.gguf".to_string()),
        mode: Some("cpu".to_string()),
        ngl_default: -1,
    };
    let dev_out = resolve_device(&resolve_input)?;
    println!("Resolved device: {:?}, ngl = {}", dev_out.device_id, dev_out.ngl);

    deinit()?;
    Ok(())
}
```

### 5.3 Example 2: LLM Text Generation
```rust
use geniex::*;

fn main() -> Result<()> {
    init()?;

    let config = ModelConfig::default();
    let mut llm = Llm::create(
        "qwen3-0.6b.gguf",
        "llama_cpp",
        &config,
        None,
        None,
        None,
    )?;

    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: "What is the capital of France?".to_string(),
    }];

    let prompt = llm.apply_chat_template(&messages, None, false, true)?;
    
    let (text, _profile) = llm.generate::<fn(&str) -> bool>(Some(&prompt), None, None, None)?;
    println!("Response: {}", text);

    deinit()?;
    Ok(())
}
```

---

## 6. Troubleshooting on Windows ARM64

| Error | Root Cause | Solution |
|---|---|---|
| `LINK : fatal error LNK1181: cannot open input file 'geniex.lib'` | Cargo cannot locate the C import library `geniex.lib`. | Ensure the C SDK is built and set `CARGO_GENIEX_LIB_DIR` to the directory containing `geniex.lib` (e.g., `sdk/pkg-geniex/lib`). |
| `Unable to generate bindings` during `cargo build` | `bindgen` cannot find Clang/LLVM installations. | Install LLVM (ARM64 build) and ensure `clang.exe` is in your `%PATH%`. |
| Error `0xc0000135` when executing binary | Runtime loader cannot find `geniex.dll` or plugin DLLs. | Add `sdk/pkg-geniex/lib` and `sdk/pkg-geniex/lib/llama_cpp` to system `%PATH%`. |

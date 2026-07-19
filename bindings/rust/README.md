# GenieX Rust Bindings

Rust bindings for the GenieX C API (`geniex.h`).

## Prerequisites

- Rust toolchain (cargo, rustc 2021 edition)
- LLVM / Clang (required by `bindgen`)
- GenieX C SDK shared library (`geniex.dll` / `libgeniex.so` / `libgeniex.dylib`)

For Windows ARM64 specific setup and step-by-step procedure, see [docs/rust_windows_arm64.md](docs/rust_windows_arm64.md).

## Build

Build the Rust crate:

```bash
cargo build
```

If the GenieX SDK library is in a custom path, set `CARGO_GENIEX_LIB_DIR`:

```bash
CARGO_GENIEX_LIB_DIR=/path/to/sdk/lib cargo build
```

## Usage

### Core API

```rust
use geniex::*;

fn main() -> Result<()> {
    init()?;
    println!("GenieX version: {}", version());
    deinit()?;
    Ok(())
}
```

### LLM Inference

```rust
use geniex::*;

fn main() -> Result<()> {
    init()?;

    let config = ModelConfig::default();
    let mut llm = Llm::create(
        "model.bin",
        "llama_cpp",
        &config,
        None,
        None,
        None,
    )?;

    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: "Hello!".to_string(),
    }];

    let prompt = llm.apply_chat_template(&messages, None, false, true)?;
    let (text, _profile) = llm.generate::<fn(&str) -> bool>(Some(&prompt), None, None, None)?;

    println!("{}", text);

    deinit()?;
    Ok(())
}
```

### VLM Inference

```rust
use geniex::*;

fn main() -> Result<()> {
    init()?;

    let config = ModelConfig::default();
    let mut vlm = Vlm::create(
        "vlm.bin",
        "llama_cpp",
        &config,
        None,
        None,
        None,
        None,
    )?;

    let caps = vlm.get_capabilities()?;
    println!("Supports vision: {}, audio: {}", caps.supports_vision, caps.supports_audio);

    let messages = vec![VlmChatMessage {
        role: "user".to_string(),
        contents: vec![VlmContent {
            r#type: "text".to_string(),
            text: "Describe the image".to_string(),
        }],
    }];

    let prompt = vlm.apply_chat_template(&messages, None, false, false)?;
    let (text, _profile) = vlm.generate::<fn(&str) -> bool>(&prompt, None, None)?;

    println!("{}", text);

    deinit()?;
    Ok(())
}
```

# geniex — Python Bindings

Python bindings for the GenieX SDK, enabling AI model inference on Qualcomm platforms.

## Installation

The Python package requires a platform-specific native library
(`geniex.dll` on Windows, `libgeniex.so` on Linux) that is **not** bundled
in the wheel. You must provide it separately:

```bash
pip install geniex

# Point the package to the directory containing the native library
export GENIEX_LIB_PATH=/path/to/sdk/pkg-geniex/lib/   # Linux
set GENIEX_LIB_PATH=C:\path\to\sdk\pkg-geniex\lib\    # Windows
```

The native library is obtained by building the SDK from source — see
[BUILD.md](BUILD.md) for step-by-step instructions.

## Quick Start

```python
import os
os.environ["GENIEX_LIB_PATH"] = "/path/to/sdk/pkg-geniex/lib/"  # set before first import if not in env

from geniex import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "/path/to/model.gguf",
    device_map="auto",   # "auto" | "cpu" | "qairt:NPU"
)

messages = [{"role": "user", "content": "What is 2+2?"}]
text = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Batch generation
output = model.generate(text, max_new_tokens=256)
print(output.text)
print(f"[{output.profile.generated_tokens} tokens, {output.profile.decode_speed:.1f} tok/s]")

# Streaming
for token in model.generate(text, max_new_tokens=256, stream=True):
    print(token, end="", flush=True)

model.close()
```

## Supported Backends

| Backend | Device | Notes |
|---------|--------|-------|
| `llama_cpp` | CPU | Default, all platforms |
| `qairt` | NPU | Qualcomm Snapdragon only |

## Requirements

- Python 3.10+
- `huggingface_hub`, `filelock` (installed automatically)
- Native library: `geniex.dll` (Windows) / `libgeniex.so` (Linux) — built separately

## Build from Source

See [BUILD.md](BUILD.md) for build and packaging instructions.

## License

Apache 2.0 — see [LICENSE](LICENSE).

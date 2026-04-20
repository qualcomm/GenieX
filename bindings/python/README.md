# geniex — Python Bindings

Python bindings for the GenieX SDK, enabling AI model inference on Qualcomm platforms.

## Installation

```bash
pip install geniex
```

## Quick Start

```python
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
- `huggingface_hub`
- `filelock`
- Native library: `geniex.dll` (Windows) / `libgeniex.so` (Linux)

## Build from Source

See [BUILD.md](BUILD.md) for build and packaging instructions.

## License

Apache 2.0 — see [LICENSE](LICENSE).

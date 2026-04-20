# Python Bindings — Build & Run

## Prerequisites

- Python 3.10+
- CMake 3.20+
- C++ compiler (GCC / Clang / MSVC)

---

## Dev mode (in-repo)

### 1. Build the native library

```bash
cd sdk
cmake --preset default          # configure (native Linux x86_64)
cmake --build --preset default
cmake --install build-default --prefix pkg-geniex
```

> **Other platforms:**
> - ARM64 Linux:   `--preset arm64-linux-snapdragon-release`
> - ARM64 Windows: `--preset arm64-windows-snapdragon-release`
> - Android:       `--preset arm64-android-snapdragon-release`
>
> After building, always run `cmake --install <build-dir> --prefix sdk/pkg-geniex` before using the Python bindings.

### 2. Run — no env vars needed

The Python package auto-discovers the built library from `sdk/pkg-geniex/lib/`
after the install step (`cmake --install`).
`GENIEX_PLUGIN_PATH` and transitive shared-lib deps are configured automatically.

```bash
python bindings/python/examples/llm.py \
  --model /path/to/model.gguf \  
```

> **Override:** set `GENIEX_LIB_PATH=/path/to/libgeniex.so` to force a specific library.

---

## Release mode (wheel)

### 1. Build the native library (see above)

### 2. Install the SDK to `pkg-geniex`

```bash
# From repo root — adjust BUILD_DIR for your preset
BUILD_DIR=sdk/build-default

cmake --install "$BUILD_DIR" --prefix sdk/pkg-geniex
```

This produces the canonical install tree:

```
sdk/pkg-geniex/
├── include/ml.h
└── lib/
    ├── libgeniex.so        (Linux) / geniex.dll (Windows)
    └── llama_cpp/
        ├── libgeniex_plugin.so  (Linux) / geniex_plugin.dll (Windows)
        ├── libggml.so / ggml.dll
        ├── libllama.so / llama.dll
        └── ...
    └── qairt/              (only when GENIEX_PLUGIN_QAIRT=ON)
        ├── libgeniex_plugin.so / geniex_plugin.dll
        └── htp-files-v73/
```

### 3. Bundle the native libs into the package tree

Copy only the `lib/` subtree from the install prefix:

```bash
DEST=bindings/python/geniex/lib
rm -rf "$DEST"
cp -r sdk/pkg-geniex/lib "$DEST"
rm -f "$DEST/llama_cpp/"*.a           # static libs — not needed at runtime
```

### 4. Build the wheel

```bash
uv build --wheel --out-dir dist/ bindings/python/
# or: cd bindings/python && python -m build --wheel -o ../../dist/
```

### 5. Install and use

```bash
uv pip install dist/geniex-*.whl
# or: pip install dist/geniex-*.whl
```

When `geniex/lib/` was bundled in step 3, the package finds the native library
automatically — no environment variables required. If the wheel was built
**without** bundling `lib/` (e.g. the Bazel wheel or a pure-Python wheel),
set `GENIEX_LIB_PATH` to the native library before use:

```bash
# Linux
export GENIEX_LIB_PATH=/path/to/sdk/pkg-geniex/lib/libgeniex.so

# Windows
set GENIEX_LIB_PATH=C:\path\to\sdk\pkg-geniex\lib\geniex.dll
```

```python
from geniex import AutoModelForCausalLM

# llama_cpp backend (CPU, default)
model = AutoModelForCausalLM.from_pretrained(
    "/path/to/model.gguf",
    device_map="cpu",
)

# QAIRT backend (Snapdragon NPU)
# model = AutoModelForCausalLM.from_pretrained(
#     "sdk/modelfiles/qairt/my_model",
#     model_name="my_model",
#     device_map="qairt:NPU",
# )

text = model.tokenizer.apply_chat_template(
    [{"role": "user", "content": "Hello!"}],
    tokenize=False,
    add_generation_prompt=True,
)

# Batch
output = model.generate(text, max_new_tokens=256)
print(output.text)
print(f"[{output.profile.generated_tokens} tokens, {output.profile.decode_speed:.1f} tok/s]")

# Streaming
for token in model.generate(text, max_new_tokens=256, stream=True):
    print(token, end="", flush=True)

model.close()
```

---

## (Optional) Bazel build

`py_library` target (for use as a Bazel dependency):

```bash
bazel build //bindings/python:geniex_py
```

`py_wheel` target — produces a `.whl` file:

```bash
bazel build //bindings/python:geniex_wheel
# Output: bazel-bin/bindings/python/geniex-0.1.0-py3-none-any.whl
```

> The Bazel wheel does **not** bundle native libs. Run steps 2–3 above first to
> stage `geniex/lib/` before building the wheel if you need the libs included.

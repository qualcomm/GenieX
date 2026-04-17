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
cmake --build build-default --parallel $(nproc)
```

> **Other platforms:**
> - ARM64 Linux:   `--preset arm64-linux-snapdragon-release`
> - ARM64 Windows: `--preset arm64-windows-snapdragon-release`
> - Android:       `--preset arm64-android-snapdragon-release`

### 2. Run — no env vars needed

The Python package auto-discovers the built library by walking up to the repo
root and globbing `sdk/build-*/src/libgeniex.so` (newest build wins).
`GENIEX_PLUGIN_PATH` and transitive `.so` deps are configured automatically.

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
├── bin/            # test binaries
├── include/ml.h
└── lib/
    ├── libgeniex.so
    └── llama_cpp/
        ├── libgeniex_plugin.so
        ├── libggml.so  (+ versioned symlinks)
        ├── libllama.so
        └── ...
```

### 3. Bundle the native libs into the package tree

Copy only the `lib/` subtree from the install prefix:

```bash
DEST=bindings/python/geniex/lib
rm -rf "$DEST"
cp -r sdk/pkg-geniex/lib "$DEST"
rm -f "$DEST/llama_cpp/libcommon.a"   # static lib — not needed at runtime
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

After installation the package finds `libgeniex.so` automatically inside
`site-packages/geniex/lib/` — no environment variables required.

```python
from geniex import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("/path/to/model.gguf", device_map="cpu")
text = model.tokenizer.apply_chat_template(
    [{"role": "user", "content": "Hello!"}],
    tokenize=False,
    add_generation_prompt=True,
)
output = model.generate(text, max_new_tokens=256)
print(output.text)
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

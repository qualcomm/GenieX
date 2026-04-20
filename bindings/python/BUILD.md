# Python Bindings — Build & Run

## Prerequisites

- Python 3.10+, CMake 3.20+, C++ compiler (GCC / Clang / MSVC)

---

## Dev mode (in-repo)

```bash
cd sdk
cmake --preset default          # native Linux x86_64; see below for other platforms
cmake --build --preset default
cmake --install build-default --prefix pkg-geniex
```

The package auto-discovers the library from `sdk/pkg-geniex/lib/` — no env vars needed.

```bash
python bindings/python/examples/llm.py --model /path/to/model.gguf
```

**Other platforms:** use `--preset arm64-linux-snapdragon-release`, `arm64-windows-snapdragon-release`, or `arm64-android-snapdragon-release`. Always run `cmake --install` after building.

**Override:** `GENIEX_LIB_PATH=/path/to/lib/dir/` forces a specific library directory.

---

## Release wheel (setuptools)

```bash
# 1. Build and install the native library (see Dev mode above)

# 2. Bundle libs into the package tree
cp -r sdk/pkg-geniex/lib bindings/python/geniex/lib
rm -f bindings/python/geniex/lib/llama_cpp/*.a   # remove static libs

# 3. Build the wheel
uv build --wheel --out-dir dist/ bindings/python/
# or: python -m build --wheel -o dist/ bindings/python/

# 4. Install
pip install dist/geniex-*.whl
```

When `geniex/lib/` is bundled the library is found automatically after `pip install`.
Without it, set `GENIEX_LIB_PATH` to the lib directory before use:

```bash
export GENIEX_LIB_PATH=/path/to/sdk/pkg-geniex/lib/   # Linux
set GENIEX_LIB_PATH=C:\path\to\sdk\pkg-geniex\lib\    # Windows
```

See [README.md](README.md) for usage examples.

---

## Bazel wheel

```bash
bazelisk build //bindings/python:geniex_wheel                     # dev (0.0.0.dev0)
bazelisk build //bindings/python:geniex_wheel --define=VERSION=0.1.0  # release
```

> The Bazel wheel does **not** bundle native libs. Run steps 1–2 above first if you need them included.

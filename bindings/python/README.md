# geniex — Python Bindings

Python bindings for the GenieX SDK, enabling AI model inference on Qualcomm
platforms.

## Installation

The published artifact is a **source distribution (sdist)**. When `pip install`
assembles a wheel from the sdist, a custom `build_py` command downloads the
SDK zip matching your platform from the same GitHub Release tag, verifies
its SHA-256 sidecar, and bundles the `lib/` tree into the resulting wheel.

```bash
# From GitHub Release (canonical)
pip install https://github.com/qcom-ai-hub/geniex/releases/download/v0.0.3-alpha.1/geniex-0.0.3a1.tar.gz

# From TestPyPI (pre-release tags are auto-published). --extra-index-url is
# required so runtime deps (huggingface_hub, filelock) resolve from real PyPI.
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            geniex==0.0.3a1
```

### Supported platforms (automatic download)

| `sys.platform` | `machine()` | Release asset |
|----------------|-------------|---------------|
| `win32`        | `arm64`     | `geniex-sdk-windows-arm64-<tag>.zip` |
| `linux`        | `aarch64` / `arm64` | `geniex-sdk-linux-arm64-<tag>.zip` |

On any other platform `pip install` aborts with a clear message — see
[Unsupported platform](#unsupported-platform) below.

### Build-time env vars

| Var | Purpose |
|-----|---------|
| `GENIEX_SDK_DOWNLOAD_URL` | Override the SDK zip base URL (e.g. internal mirror, `file:///...` for offline testing). The asset name `geniex-sdk-<platform>-<tag>.zip` is appended. |
| `GENIEX_SKIP_SDK_DOWNLOAD` | Set to `1` to skip the download entirely — useful for unsupported platforms or when you plan to provide libs via `GENIEX_LIB_PATH` at runtime. |

### Runtime env var

| Var | Purpose |
|-----|---------|
| `GENIEX_LIB_PATH` | Directory (or file) pointing to an already-built `libgeniex.so` / `geniex.dll`. Overrides all auto-discovery. |

### Unsupported platform

`pip install` aborts with a clear error message. Two workarounds:

1. Build the SDK locally (see [Build SDK from source](#build-sdk-from-source)),
   then:
   ```bash
   GENIEX_SKIP_SDK_DOWNLOAD=1 pip install <sdist-url>
   export GENIEX_LIB_PATH=/path/to/sdk/pkg-geniex/lib/
   ```
2. Or copy `sdk/pkg-geniex/lib` into `bindings/python/geniex/lib/` before
   invoking `pip install bindings/python/` from a repo checkout.

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

## Dev mode (from a repo checkout)

Build the SDK in-tree and let `_lib.py` auto-discover it from
`sdk/pkg-geniex/lib/` — no env vars needed.

```bash
cd sdk
cmake --preset default          # native Linux x86_64
cmake --build --preset default
cmake --install build-default --prefix pkg-geniex

python bindings/python/examples/llm.py --model /path/to/model.gguf
```

**Other platforms:** `--preset arm64-linux-snapdragon-release`,
`arm64-windows-snapdragon-release`, or `arm64-android-snapdragon-release`.
Always run `cmake --install` after building.

**Override:** set `GENIEX_LIB_PATH=/path/to/lib/dir/` to force a specific
library directory.

## Build SDK from source

Prerequisites: Python 3.10+, CMake 3.20+, C++ compiler (GCC / Clang / MSVC).

Full platform-specific build instructions (Linux, Windows ARM64 + Hexagon,
Android cross-compile) live in [`docs/build.md`](../../docs/build.md).
After `cmake --install`, the libs land in `sdk/pkg-geniex/lib/` and both the
dev-mode path and `GENIEX_LIB_PATH` pick them up.

## Build the sdist locally

```bash
python -m pip install build
python -m build --sdist bindings/python --outdir dist/
# produces dist/geniex-<version>.tar.gz — no native libs inside
```

The sdist is pure Python; SDK libs are fetched later at install time.

### End-to-end local test with a file mirror

```bash
# 1. Produce a local SDK zip (example: arm64 Linux)
cd sdk && cmake --preset arm64-linux-snapdragon-release && \
  cmake --build --preset arm64-linux-snapdragon-release && \
  cmake --install build-arm64-linux-snapdragon-release --prefix pkg-geniex
mkdir -p /tmp/geniex-mirror
(cd sdk && zip -r /tmp/geniex-mirror/geniex-sdk-linux-arm64-v0.0.3-alpha.1.zip pkg-geniex)
sha256sum /tmp/geniex-mirror/geniex-sdk-linux-arm64-v0.0.3-alpha.1.zip \
  > /tmp/geniex-mirror/geniex-sdk-linux-arm64-v0.0.3-alpha.1.zip.sha256

# 2. Install the sdist pointing at the mirror
GENIEX_SDK_DOWNLOAD_URL=file:///tmp/geniex-mirror \
  pip install dist/geniex-0.0.3a1.tar.gz

# 3. Smoke test
python -c "import geniex; geniex.init(); print(geniex.version())"
```

## Bazel

```bash
bazelisk build //bindings/python:geniex_sdist                                  # dev (0.0.0.dev0)
bazelisk build //bindings/python:geniex_sdist --define=VERSION=v0.0.3-alpha.1  # release
```

Output: `bazel-bin/bindings/python/geniex_sdist.tar.gz`. Same tarball shape
as `python -m build --sdist`; same install behavior.

## License

Apache 2.0 — see [LICENSE](LICENSE).

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

## Install from the official release

The published artifact is a **source distribution (sdist)**. `pip install`
builds a wheel from the sdist, and during that build a custom `build_py`
command downloads the SDK zip matching your platform from the same GitHub
Release tag and bundles its `lib/` tree into the wheel.

```bash
pip install https://github.com/qcom-ai-hub/geniex/releases/download/v0.1.0-alpha.1/geniex-0.1.0a1.tar.gz
```

### Build-time env vars (setup.py)

| Var | Purpose |
|-----|---------|
| `GENIEX_SDK_DOWNLOAD_URL` | Override the SDK zip base URL (e.g. internal mirror, `file:///...` for offline testing). Appended with `/geniex-sdk-<platform>-<tag>.zip`. |
| `GENIEX_SKIP_SDK_DOWNLOAD` | Set to `1` to skip the download entirely — useful for unsupported platforms or when you plan to provide libs via `GENIEX_LIB_PATH` at runtime. |

### Runtime env var

| Var | Purpose |
|-----|---------|
| `GENIEX_LIB_PATH` | Directory (or file) pointing to an already-built `libgeniex.so` / `geniex.dll`. Overrides all auto-discovery. |

### Unsupported platform

`pip install` aborts with a clear error message. Two workarounds:

1. Build the SDK locally (see Dev mode above), then
   ```bash
   GENIEX_SKIP_SDK_DOWNLOAD=1 pip install <sdist-url>
   export GENIEX_LIB_PATH=/path/to/sdk/pkg-geniex/lib/
   ```
2. Or copy `sdk/pkg-geniex/lib` into `bindings/python/geniex/lib/` before invoking `pip install bindings/python`.

---

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
(cd sdk && zip -r /tmp/geniex-mirror/geniex-sdk-linux-arm64-v0.1.0-alpha.1.zip pkg-geniex)
sha256sum /tmp/geniex-mirror/geniex-sdk-linux-arm64-v0.1.0-alpha.1.zip \
  > /tmp/geniex-mirror/geniex-sdk-linux-arm64-v0.1.0-alpha.1.zip.sha256

# 2. Install the sdist pointing at the mirror
GENIEX_SDK_DOWNLOAD_URL=file:///tmp/geniex-mirror \
  pip install dist/geniex-0.1.0a1.tar.gz

# 3. Smoke test
python -c "import geniex; geniex.init(); print(geniex.version())"
```

---

## Bazel

```bash
bazelisk build //bindings/python:geniex_sdist                              # dev (0.0.0.dev0)
bazelisk build //bindings/python:geniex_sdist --define=VERSION=v0.1.0-alpha.1  # release
```

Output: `bazel-bin/bindings/python/geniex_sdist.tar.gz`. Same tarball shape
as `python -m build --sdist`; same install behavior.

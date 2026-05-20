# Python Bindings — Build & Development

Consumer install and usage instructions live in [README.md](README.md).
This document covers everything that's internal to building the package
or working from a repo checkout.

## Artifact layout

Three sibling **source distributions (sdists)** ship from the same release
tag (#554):

| sdist             | source tree                       | plugins staged               |
|-------------------|-----------------------------------|------------------------------|
| `geniex`          | `bindings/python/`                | llama.cpp **and** QAIRT      |
| `geniex-llama-cpp`| `bindings/python-llama-cpp/`      | llama.cpp only               |
| `geniex-qairt`    | `bindings/python-qairt/`          | QAIRT only                   |

Each tree carries its own `pyproject.toml` / `setup.py` / `MANIFEST.in`.
Everything else — the `geniex/` Python package, `_sdk_fetch.py`,
`_geniex_backend.py`, `README.md`, `LICENSE` — lives only in
`bindings/python/` and is mirrored into the sibling trees by
[`bindings/sync_siblings.py`](../sync_siblings.py) immediately before each
build. The mirrored copies are git-ignored.

When `pip install` assembles a wheel from any of the three sdists, the
custom `build_py` command runs `_sdk_fetch.fetch(..., backends=...)` to
pull just the requested slice of the platform-matching SDK zip via HTTP
Range requests, falling back to a full download with SHA-256 sidecar
verification when the mirror doesn't support ranges.

All three sdists are pure Python — they never contain prebuilt libs.

`pyproject.toml` declares an in-tree PEP 517 wrapper (`_geniex_backend`,
`backend-path = ["."]`) that aliases `tomli` → `tomllib` before
delegating to `setuptools.build_meta`. This is the only thing that lets
`pip install` succeed on Qualcomm Linux's stripped Python 3.12 (which
omits stdlib `tomllib`). On hosts with stdlib `tomllib` the alias path
is skipped entirely. See GH #538.

## Install sources

```bash
# From GitHub Release (canonical) — pick the distribution you need
pip install https://github.com/qcom-ai-hub/geniex/releases/download/v0.0.3-alpha.1/geniex-0.0.3a1.tar.gz
pip install https://github.com/qcom-ai-hub/geniex/releases/download/v0.0.3-alpha.1/geniex_llama_cpp-0.0.3a1.tar.gz
pip install https://github.com/qcom-ai-hub/geniex/releases/download/v0.0.3-alpha.1/geniex_qairt-0.0.3a1.tar.gz

# From TestPyPI (pre-release tags are auto-published)
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple geniex
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple geniex-llama-cpp
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple geniex-qairt
```

### Supported platforms (automatic SDK download)

| `sys.platform` | `machine()`          | Release asset                        |
|----------------|----------------------|--------------------------------------|
| `win32`        | `arm64`              | `geniex-sdk-windows-arm64-<tag>.zip` |
| `linux`        | `aarch64` / `arm64`  | `geniex-sdk-linux-arm64-<tag>.zip`   |

Any other platform aborts `pip install` with a clear error — see
[Unsupported platform](#unsupported-platform) below.

## Build-time environment variables

| Var                        | Purpose                                                                                                                                                                  |
|----------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `GENIEX_SDK_DOWNLOAD_URL`  | Override the SDK zip base URL (internal mirror, `file:///...` for offline testing). The asset name `geniex-sdk-<platform>-<tag>.zip` is appended. Pins a single source — disables the S3/GitHub fallback. |
| `GENIEX_SKIP_SDK_DOWNLOAD` | Set to `1` to skip the download — useful for unsupported platforms or when you'll provide libs via `GENIEX_LIB_PATH` at runtime.                                          |

Default fetch order: the installer tries the public S3 mirror first
(`qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-geniex/geniex-sdk-<platform>-<tag>.zip`
— flat layout, the `<tag>` in the filename disambiguates versions) and
falls back to the matching GitHub Release asset on any network, HTTP,
or SHA-256 error.

## Runtime environment variable

| Var               | Purpose                                                                             |
|-------------------|-------------------------------------------------------------------------------------|
| `GENIEX_LIB_PATH` | Directory (or file) pointing to an already-built `libgeniex.so` / `geniex.dll`. Overrides all auto-discovery. |

## Unsupported platform

`pip install` aborts with a clear error message. Two workarounds:

1. Build the SDK locally (see [Build SDK from source](#build-sdk-from-source)),
   then:
   ```bash
   GENIEX_SKIP_SDK_DOWNLOAD=1 pip install <sdist-url>
   export GENIEX_LIB_PATH=/path/to/sdk/pkg-geniex/lib/
   ```
2. Or copy `sdk/pkg-geniex/lib` into `bindings/python/geniex/lib/` before
   invoking `pip install bindings/python/` from a repo checkout.

## Dev mode (from a repo checkout)

Build the SDK in-tree and let `_lib.py` auto-discover it from
`sdk/pkg-geniex/lib/` — no env vars needed.

```bash
cd sdk
cmake --preset default                  # native Linux x86_64
cmake --build --preset default
cmake --install build-default --prefix pkg-geniex

python -m geniex.cli chat /path/to/model.gguf
```

Other platforms: `--preset arm64-linux-snapdragon-release`,
`arm64-windows-snapdragon-release`, or `arm64-android-snapdragon-release`.
Always run `cmake --install` after building.

Override: set `GENIEX_LIB_PATH=/path/to/lib/dir/` to force a specific
library directory.

## Build SDK from source

Prerequisites: Python 3.10+, CMake 3.20+, C++ compiler (GCC / Clang / MSVC).

Full platform-specific build instructions (Linux, Windows ARM64 + Hexagon,
Android cross-compile) live in [`notes/build.md`](../../notes/build.md).
After `cmake --install`, the libs land in `sdk/pkg-geniex/lib/` and both
dev mode and `GENIEX_LIB_PATH` pick them up.

## Build the sdists locally

Mirror the shared sources into the sibling trees, then build each sdist:

```bash
python -m pip install build
python bindings/sync_siblings.py
python -m build --sdist bindings/python            --outdir dist/   # geniex
python -m build --sdist bindings/python-llama-cpp  --outdir dist/   # geniex-llama-cpp
python -m build --sdist bindings/python-qairt      --outdir dist/   # geniex-qairt
ls dist/
# geniex-<version>.tar.gz
# geniex_llama_cpp-<version>.tar.gz
# geniex_qairt-<version>.tar.gz
```

All three sdists are pure Python; the corresponding SDK lib slice is
fetched at install time.

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

Output: `bazel-bin/bindings/python/geniex_sdist.tar.gz`. Same tarball
shape as `python -m build --sdist`; same install behavior.

## TestPyPI smoke-test

See [`docs/testpypi-smoketest.md`](../../docs/testpypi-smoketest.md) for
the end-to-end verification transcript.

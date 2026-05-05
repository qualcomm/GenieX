# Python Bindings — Build & Development

Consumer install and usage instructions live in [README.md](README.md).
This document covers everything that's internal to building the package
or working from a repo checkout.

## Artifact layout

The published artifact is a **source distribution (sdist)**. When
`pip install` assembles a wheel from the sdist, a custom `build_py`
command downloads the SDK zip matching the target platform from the
same GitHub Release tag, verifies its SHA-256 sidecar, and bundles the
`lib/` tree into the resulting wheel.

The sdist itself is pure Python — it never contains prebuilt libs.

## Install sources

```bash
# From GitHub Release (canonical)
pip install https://github.com/qcom-ai-hub/geniex/releases/download/v0.0.3-alpha.1/geniex-0.0.3a1.tar.gz

# From TestPyPI (pre-release tags are auto-published)
pip install -i https://test.pypi.org/simple/ geniex
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
(`qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-geniex/<tag>/...`)
and falls back to the matching GitHub Release asset on any network,
HTTP, or SHA-256 error.

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

## Build the sdist locally

```bash
python -m pip install build
python -m build --sdist bindings/python --outdir dist/
# produces dist/geniex-<version>.tar.gz — no native libs inside
```

The sdist is pure Python; SDK libs are fetched at install time.

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

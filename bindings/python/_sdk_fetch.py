# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Install-time SDK fetcher.

Invoked from setup.py during wheel assembly. Downloads the SDK zip matching
the current platform + release tag, verifies its SHA-256 sidecar, and
extracts lib/ into the package tree so package-data picks it up.

By default the fetcher tries the public S3 mirror first and falls back to
the GitHub Release asset. Set ``GENIEX_SDK_DOWNLOAD_URL`` to pin a single
source (internal mirror, ``file://`` path, etc.) — that disables fallback.

Skipped when:
  - geniex/lib/ already exists (cached / pre-staged build)
  - GENIEX_SKIP_SDK_DOWNLOAD=1 is set
"""

from __future__ import annotations

import hashlib
import io
import os
import platform
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_BASE_URL = 'https://github.com/qcom-ai-hub/geniex/releases/download'
DEFAULT_S3_BASE_URL = 'https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-geniex'

# (sys.platform, platform.machine().lower()) -> release asset platform triple.
PLATFORM_MAP = {
    ('win32', 'arm64'): 'windows-arm64',
    ('linux', 'aarch64'): 'linux-arm64',
    ('linux', 'arm64'): 'linux-arm64',
}


def _detect_platform() -> str:
    key = (sys.platform, platform.machine().lower())
    plat = PLATFORM_MAP.get(key)
    if plat is None:
        raise RuntimeError(
            f'Unsupported platform {key} for prebuilt geniex SDK.\n'
            'Build the SDK locally (see bindings/python/BUILD.md) and either:\n'
            '  - set GENIEX_SKIP_SDK_DOWNLOAD=1 before pip install, then\n'
            '    export GENIEX_LIB_PATH=/path/to/sdk/pkg-geniex/lib/ at runtime; or\n'
            '  - copy sdk/pkg-geniex/lib into bindings/python/geniex/lib/ before\n'
            '    running pip install / python -m build.'
        )
    return plat


def _try_download(url: str) -> bytes | None:
    """Fetch `url`; return bytes on success or None on network/HTTP failure."""
    try:
        with urllib.request.urlopen(url) as resp:
            return resp.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f'[geniex] source unavailable: {url} ({exc})', file=sys.stderr)
        return None


def _try_source(base: str, asset: str) -> tuple[bytes, str] | str:
    """Try one base URL. Return (zip_bytes, source_url) on success,
    or a failure-reason string on any error (network, missing sidecar,
    SHA mismatch, corrupt)."""
    zip_url = f'{base.rstrip("/")}/{asset}'
    sha_url = f'{zip_url}.sha256'

    zip_bytes = _try_download(zip_url)
    if zip_bytes is None:
        return f'{zip_url}: download failed'

    sha_bytes = _try_download(sha_url)
    if sha_bytes is None:
        return f'{sha_url}: download failed'

    want = sha_bytes.decode().strip().split()[0]
    got = hashlib.sha256(zip_bytes).hexdigest()
    if got.lower() != want.lower():
        return f'{zip_url}: SHA256 mismatch (expected {want}, got {got})'

    return zip_bytes, zip_url


def fetch(pkg_dir: Path, release_tag: str) -> None:
    """Populate pkg_dir/lib/ with the SDK libs for the current platform."""
    lib_dir = pkg_dir / 'lib'
    if lib_dir.exists() and any(lib_dir.iterdir()):
        print(f'[geniex] {lib_dir} already populated, skipping SDK download.')
        return
    if os.environ.get('GENIEX_SKIP_SDK_DOWNLOAD'):
        print('[geniex] GENIEX_SKIP_SDK_DOWNLOAD set, skipping SDK download.')
        return

    plat = _detect_platform()
    asset = f'geniex-sdk-{plat}-{release_tag}.zip'

    override = os.environ.get('GENIEX_SDK_DOWNLOAD_URL')
    if override:
        sources = [('override', override.rstrip('/'))]
    else:
        sources = [
            ('s3', f'{DEFAULT_S3_BASE_URL}/{release_tag}'),
            ('github', f'{DEFAULT_BASE_URL}/{release_tag}'),
        ]

    errors = []
    zip_bytes: bytes | None = None
    used_name = used_url = ''
    for name, base in sources:
        print(f'[geniex] Trying {name}: {base}/{asset}')
        result = _try_source(base, asset)
        if isinstance(result, tuple):
            zip_bytes, used_url = result
            used_name = name
            break
        errors.append(result)

    if zip_bytes is None:
        raise RuntimeError('Failed to fetch SDK from all sources:\n  - ' + '\n  - '.join(errors))

    print(f'[geniex] SDK fetched from {used_name}: {used_url}')

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            z.extractall(tmp_path)
        # Archive contains sdk-pkg-<platform>/lib/... (see release.yml).
        candidates = (
            list(tmp_path.rglob('libgeniex.so'))
            + list(tmp_path.rglob('geniex.dll'))
            + list(tmp_path.rglob('libgeniex.dylib'))
        )
        if not candidates:
            raise RuntimeError(f'SDK zip {asset} has no libgeniex library')
        src_lib_dir = candidates[0].parent
        lib_dir.mkdir(parents=True, exist_ok=True)
        for entry in src_lib_dir.iterdir():
            dst = lib_dir / entry.name
            if entry.is_dir():
                shutil.copytree(entry, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(entry, dst)
        # Drop static archives — only shared libs need shipping.
        for a in lib_dir.rglob('*.a'):
            a.unlink()
    print(f'[geniex] SDK libs installed at {lib_dir}')

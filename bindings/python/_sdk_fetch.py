# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Install-time SDK fetcher.

Invoked from setup.py during wheel assembly. Pulls the SDK libs matching the
current platform + release tag and stages them under ``geniex/lib/`` so
package-data picks them up.

The fetcher accepts a ``backends`` set ⊆ ``{'llama-cpp', 'qairt'}`` and only
extracts the corresponding plugin subtree (plus the always-required core
``geniex.dll`` / ``libgeniex.so``). It first attempts HTTP Range requests
against the published SDK zip — only the relevant slice is downloaded — and
falls back to a full download when the server can't fulfil ``Range:``.

By default the fetcher tries the public S3 mirror first and falls back to the
GitHub Release asset. Set ``GENIEX_SDK_DOWNLOAD_URL`` to pin a single source
(internal mirror, ``file://`` path, etc.) — that disables the inter-source
fallback but range/full fallback within the chosen source still applies.

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
import struct
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Backend = Literal['llama-cpp', 'qairt']

DEFAULT_BASE_URL = 'https://github.com/qcom-ai-hub/geniex/releases/download'
DEFAULT_S3_BASE_URL = 'https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-geniex'

# (sys.platform, platform.machine().lower()) -> release asset platform triple.
PLATFORM_MAP = {
    ('win32', 'arm64'): 'windows-arm64',
    ('linux', 'aarch64'): 'linux-arm64',
    ('linux', 'arm64'): 'linux-arm64',
}

_CORE_LIB_NAMES = ('geniex.dll', 'libgeniex.so', 'libgeniex.dylib')

_BACKEND_DIRS = {
    'llama-cpp': 'llama_cpp',
    'qairt': 'qairt',
}

_EOCD_SIG = b'PK\x05\x06'
_CD_SIG = b'PK\x01\x02'
_LFH_SIG = b'PK\x03\x04'
_EOCD_FIXED = 22
_EOCD_MAX_COMMENT = 0xFFFF
_CD_HDR_FIXED = 46
_LFH_FIXED = 30
_ZIP64_U16 = 0xFFFF
_ZIP64_U32 = 0xFFFFFFFF


class _RangeNotSupported(Exception):
    """Server didn't honor a ``Range:`` request — fall back to full download."""


class _ZIP64NotSupported(Exception):
    """Encountered a ZIP64-encoded field; the range-fetch path doesn't implement
    the ZIP64 extensions yet, so we fall back to a full download."""


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


def _file_url_to_path(url: str) -> Path:
    parsed = urllib.parse.urlparse(url)
    return Path(urllib.request.url2pathname(parsed.path))


def _fetch_range(url: str, start: int, end: int, *, exact: bool = True) -> bytes:
    """Inclusive byte range ``[start, end]``.

    With ``exact=True`` (default) the response must equal ``end - start + 1``
    bytes — used when both bounds came from the central directory. With
    ``exact=False`` the server may legitimately truncate the tail past
    end-of-resource; the caller is expected to validate via in-band
    structure (e.g. CRC32 of an extracted entry).

    Raises ``_RangeNotSupported`` when the server returns 200 (whole body)
    instead of 206.
    """
    expected = end - start + 1
    if url.startswith('file://'):
        path = _file_url_to_path(url)
        with path.open('rb') as fh:
            fh.seek(start)
            data = fh.read(expected)
        if exact and len(data) != expected:
            raise RuntimeError(f'{url}: short read {len(data)} != {expected}')
        return data
    req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
    try:
        with urllib.request.urlopen(req) as resp:
            if resp.status != 206:
                raise _RangeNotSupported(f'{url}: status {resp.status} for Range request')
            data = resp.read()
    except urllib.error.HTTPError as exc:
        raise _RangeNotSupported(f'{url}: HTTP {exc.code} on Range request') from exc
    if exact and len(data) != expected:
        raise RuntimeError(f'{url}: short read {len(data)} != {expected}')
    return data


def _fetch_suffix(url: str, n: int) -> tuple[bytes, int]:
    """Fetch the last ``n`` bytes of ``url`` and return ``(data, total_size)``.

    Uses an HTTP suffix range (``bytes=-n``) so the server's ``Content-Range``
    header gives us the resource's full length in the same response — saves
    a separate HEAD/probe round-trip.
    """
    if url.startswith('file://'):
        path = _file_url_to_path(url)
        total = path.stat().st_size
        start = max(0, total - n)
        with path.open('rb') as fh:
            fh.seek(start)
            data = fh.read(total - start)
        return data, total
    req = urllib.request.Request(url, headers={'Range': f'bytes=-{n}'})
    try:
        with urllib.request.urlopen(req) as resp:
            if resp.status != 206:
                raise _RangeNotSupported(f'{url}: status {resp.status} for suffix Range request')
            data = resp.read()
            cr = resp.headers.get('Content-Range', '')
            if '/' not in cr:
                raise _RangeNotSupported(f'{url}: 206 without parseable Content-Range')
            tail = cr.rsplit('/', 1)[-1]
            if not tail.isdigit():
                raise _RangeNotSupported(f'{url}: non-numeric total in Content-Range: {cr}')
            return data, int(tail)
    except urllib.error.HTTPError as exc:
        raise _RangeNotSupported(f'{url}: HTTP {exc.code} on suffix Range request') from exc


@dataclass(slots=True)
class _CDEntry:
    filename: str
    compression: int
    crc32: int
    compressed_size: int
    local_header_offset: int


def _parse_central_directory(cd_bytes: bytes) -> list[_CDEntry]:
    fmt = '<4s6H3L5H2L'
    fixed = struct.calcsize(fmt)
    assert fixed == _CD_HDR_FIXED
    entries: list[_CDEntry] = []
    pos = 0
    n = len(cd_bytes)
    while pos < n:
        if cd_bytes[pos : pos + 4] != _CD_SIG:
            raise RuntimeError(f'corrupt central directory at offset {pos}')
        unpacked = struct.unpack(fmt, cd_bytes[pos : pos + fixed])
        comp = unpacked[4]
        crc = unpacked[7]
        csize = unpacked[8]
        usize = unpacked[9]
        fn_len = unpacked[10]
        ex_len = unpacked[11]
        cm_len = unpacked[12]
        lho = unpacked[16]
        if csize == _ZIP64_U32 or usize == _ZIP64_U32 or lho == _ZIP64_U32:
            raise _ZIP64NotSupported('ZIP64 entry encountered — range path unimplemented')
        name_start = pos + fixed
        name = cd_bytes[name_start : name_start + fn_len].decode('utf-8', errors='replace')
        entries.append(
            _CDEntry(
                filename=name,
                compression=comp,
                crc32=crc,
                compressed_size=csize,
                local_header_offset=lho,
            )
        )
        pos += fixed + fn_len + ex_len + cm_len
    return entries


def _classify_entry(filename: str, backends: frozenset[Backend]) -> str | None:
    """Relative path under ``lib/`` for entries to keep, ``None`` otherwise.

    SDK zip entries look like ``sdk-<platform>/lib/...``. The ``lib`` segment
    anchors classification.
    """
    parts = filename.split('/')
    try:
        idx = parts.index('lib')
    except ValueError:
        return None
    rel = parts[idx + 1 :]
    if not rel or rel[-1] == '':
        return None
    if rel[-1].endswith('.a'):
        return None
    if len(rel) == 1 and rel[0] in _CORE_LIB_NAMES:
        return rel[0]
    if rel[0] == _BACKEND_DIRS['llama-cpp'] and 'llama-cpp' in backends:
        return '/'.join(rel)
    if rel[0] == _BACKEND_DIRS['qairt'] and 'qairt' in backends:
        return '/'.join(rel)
    return None


def _extract_entry(zip_url: str, entry: _CDEntry, dst: Path) -> None:
    # Pull the local file header and the compressed payload in one ranged
    # GET — they're contiguous on disk, so a single round-trip works as long
    # as we know an upper bound on the variable-length header fields. The
    # zip spec caps name and extra at 65535 bytes each; in practice they
    # are tens of bytes for SDK assets, but the upper bound keeps us correct.
    span_start = entry.local_header_offset
    span_end = span_start + _LFH_FIXED + 2 * 0xFFFF + entry.compressed_size - 1
    blob = _fetch_range(zip_url, span_start, span_end, exact=False)
    if blob[:4] != _LFH_SIG:
        raise RuntimeError(f'{entry.filename}: bad local file header signature')
    fn_len, ex_len = struct.unpack('<HH', blob[26:30])
    payload_off = _LFH_FIXED + fn_len + ex_len
    raw = blob[payload_off : payload_off + entry.compressed_size]
    if len(raw) != entry.compressed_size:
        raise RuntimeError(f'{entry.filename}: short payload {len(raw)} != {entry.compressed_size}')
    if entry.compression == 0:
        data = raw
    elif entry.compression == 8:
        data = zlib.decompress(raw, -15)
    else:
        raise RuntimeError(f'{entry.filename}: unsupported compression method {entry.compression}')
    if zlib.crc32(data) != entry.crc32:
        raise RuntimeError(f'{entry.filename}: CRC32 mismatch')
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)


def _range_fetch(zip_url: str, lib_dir: Path, backends: frozenset[Backend]) -> int:
    head_size = _EOCD_FIXED + _EOCD_MAX_COMMENT
    eocd_chunk, total = _fetch_suffix(zip_url, head_size)
    if total <= 0:
        raise _RangeNotSupported(f'{zip_url}: empty resource')
    eocd_pos = eocd_chunk.rfind(_EOCD_SIG)
    if eocd_pos == -1:
        raise RuntimeError(f'{zip_url}: end-of-central-directory record not found')
    (_sig, _disk, _disk_cd, entries_disk, total_entries, cd_size, cd_offset, _cm_len) = struct.unpack(
        '<4s4H2LH', eocd_chunk[eocd_pos : eocd_pos + _EOCD_FIXED]
    )
    if entries_disk == _ZIP64_U16 or total_entries == _ZIP64_U16 or cd_size == _ZIP64_U32 or cd_offset == _ZIP64_U32:
        raise _ZIP64NotSupported('ZIP64 archive — range path unimplemented')

    cd_bytes = _fetch_range(zip_url, cd_offset, cd_offset + cd_size - 1)
    entries = _parse_central_directory(cd_bytes)

    extracted = 0
    found_core = False
    for entry in entries:
        rel = _classify_entry(entry.filename, backends)
        if rel is None:
            continue
        if rel in _CORE_LIB_NAMES:
            found_core = True
        _extract_entry(zip_url, entry, lib_dir / rel)
        extracted += 1
    if not found_core:
        raise RuntimeError(f'{zip_url}: no core libgeniex/geniex.dll entry in central directory')
    return extracted


def _full_extract(zip_bytes: bytes, lib_dir: Path, backends: frozenset[Backend]) -> int:
    extracted = 0
    found_core = False
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            rel = _classify_entry(info.filename, backends)
            if rel is None:
                continue
            if rel in _CORE_LIB_NAMES:
                found_core = True
            dst = lib_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, dst.open('wb') as out:
                shutil.copyfileobj(src, out)
            extracted += 1
    if not found_core:
        raise RuntimeError('full-download zip has no libgeniex / geniex.dll entry')
    return extracted


def fetch(
    pkg_dir: Path,
    release_tag: str,
    *,
    backends: Iterable[Backend] = ('llama-cpp', 'qairt'),
) -> None:
    """Populate ``pkg_dir/lib/`` with SDK libs for the requested ``backends``.

    ``backends`` ⊆ ``{'llama-cpp', 'qairt'}``. The core shared library
    (``geniex.dll`` / ``libgeniex.so`` / ``libgeniex.dylib``) is always
    pulled — backends only control which plugin subtree is staged.

    Pulls just the relevant slice via HTTP Range when the source supports it
    and falls back to a full download otherwise. SHA256 sidecar verification
    applies to the full-download fallback path; the range path relies on
    per-entry CRC32 from the authenticated central directory.
    """
    lib_dir = pkg_dir / 'lib'
    if lib_dir.exists() and any(lib_dir.iterdir()):
        print(f'[geniex] {lib_dir} already populated, skipping SDK download.')
        return
    if os.environ.get('GENIEX_SKIP_SDK_DOWNLOAD'):
        print('[geniex] GENIEX_SKIP_SDK_DOWNLOAD set, skipping SDK download.')
        return

    backend_set = frozenset(backends)
    unknown = backend_set - set(_BACKEND_DIRS)
    if unknown:
        raise ValueError(f'unknown backends: {sorted(unknown)}; expected subset of {sorted(_BACKEND_DIRS)}')

    plat = _detect_platform()
    asset = f'geniex-sdk-{plat}-{release_tag}.zip'

    override = os.environ.get('GENIEX_SDK_DOWNLOAD_URL')
    if override:
        sources = [('override', override.rstrip('/'))]
    else:
        # S3 mirror is flat: every release asset sits directly under
        # qai-hub-geniex/, with the <tag> already in the filename. GitHub
        # Releases are inherently per-tag, so that path keeps /{release_tag}.
        sources = [
            ('s3', DEFAULT_S3_BASE_URL),
            ('github', f'{DEFAULT_BASE_URL}/{release_tag}'),
        ]

    errors: list[str] = []
    for name, base in sources:
        zip_url = f'{base}/{asset}'
        if _try_one_source(name, zip_url, lib_dir, backend_set, errors):
            return

    raise RuntimeError('Failed to fetch SDK from all sources:\n  - ' + '\n  - '.join(errors))


def _try_one_source(
    name: str,
    zip_url: str,
    lib_dir: Path,
    backends: frozenset[Backend],
    errors: list[str],
) -> bool:
    """Try one (mirror, asset) pair: range path first, then full-download fallback.

    Appends to ``errors`` on every failure mode and returns ``False`` so the
    outer loop can move on to the next source. Returns ``True`` after
    successfully populating ``lib_dir``.
    """
    sha_url = f'{zip_url}.sha256'
    print(f'[geniex] Trying {name}: {zip_url}')

    sha_bytes = _try_download(sha_url)
    if sha_bytes is None:
        errors.append(f'{sha_url}: download failed')
        return False
    want_sha = sha_bytes.decode().strip().split()[0]

    lib_dir.mkdir(parents=True, exist_ok=True)
    try:
        count = _range_fetch(zip_url, lib_dir, backends)
    except (_RangeNotSupported, _ZIP64NotSupported) as exc:
        print(f'[geniex] Range fetch unavailable on {name} ({exc}); falling back to full download.')
        shutil.rmtree(lib_dir, ignore_errors=True)
        lib_dir.mkdir(parents=True, exist_ok=True)
    except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
        errors.append(f'{zip_url} (range): {exc}')
        shutil.rmtree(lib_dir, ignore_errors=True)
        return False
    else:
        print(f'[geniex] Range-fetched {count} entries from {name}: {zip_url}')
        print(f'[geniex] SDK libs installed at {lib_dir}')
        return True

    zip_bytes = _try_download(zip_url)
    if zip_bytes is None:
        errors.append(f'{zip_url}: download failed')
        return False
    got = hashlib.sha256(zip_bytes).hexdigest()
    if got.lower() != want_sha.lower():
        errors.append(f'{zip_url}: SHA256 mismatch (expected {want_sha}, got {got})')
        return False
    try:
        count = _full_extract(zip_bytes, lib_dir, backends)
    except RuntimeError as exc:
        errors.append(f'{zip_url} (full): {exc}')
        shutil.rmtree(lib_dir, ignore_errors=True)
        return False
    print(f'[geniex] Full-zip extracted {count} entries from {name}: {zip_url}')
    print(f'[geniex] SDK libs installed at {lib_dir}')
    return True

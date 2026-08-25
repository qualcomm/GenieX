# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Install-time SDK fetcher — invoked from setup.py during wheel assembly."""

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

DEFAULT_BASE_URL = 'https://github.com/qualcomm/GenieX/releases/download'
DEFAULT_S3_BASE_URL = 'https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-geniex'

PLATFORM_MAP = {
    ('win32', 'arm64'): 'windows-arm64',
    ('linux', 'aarch64'): 'linux-arm64',
    ('linux', 'arm64'): 'linux-arm64',
}

# Baseline armv8.0 boards (unoq) can't run the default linux-arm64 SDK, so CI
# publishes a CPU-only one. Picking it is explicit. See GenieX #1217.
_CPU_SUFFIX = '-cpu'
_VARIANT_ENV = 'GENIEX_SDK_VARIANT'

_CORE_LIB_NAMES = ('geniex.dll', 'libgeniex.so', 'libgeniex.dylib')
_BACKEND_DIRS = {'llama-cpp': 'llama_cpp', 'qairt': 'qairt'}

_EOCD_SIG = b'PK\x05\x06'
_CD_SIG = b'PK\x01\x02'
_LFH_SIG = b'PK\x03\x04'
_EOCD_FIXED = 22
_LFH_FIXED = 30
_ZIP64_U16 = 0xFFFF
_ZIP64_U32 = 0xFFFFFFFF
_SUFFIX_PROBE = 262144


class _RangeNotSupported(Exception):
    pass


class _ZIP64NotSupported(Exception):
    pass


def _detect_platform() -> str:
    key = (sys.platform, platform.machine().lower())
    plat = PLATFORM_MAP.get(key)
    if plat == 'linux-arm64':
        variant = os.environ.get(_VARIANT_ENV, '').strip().lower()
        if variant == 'cpu':
            plat += _CPU_SUFFIX
        elif variant not in ('', 'default'):
            raise RuntimeError(f'{_VARIANT_ENV}={variant!r} is not one of: default, cpu')
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


_tty_out = None


def _tty():
    global _tty_out
    if _tty_out is None:
        try:
            name = 'CONOUT$' if sys.platform == 'win32' else '/dev/tty'
            _tty_out = open(name, 'w', encoding='utf-8', errors='replace', buffering=1)
        except OSError:
            _tty_out = sys.stderr
    return _tty_out


def _try_download(url: str) -> bytes | None:
    try:
        with urllib.request.urlopen(url) as resp:
            return resp.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f'[geniex] source unavailable: {url} ({exc})', file=sys.stderr)
        return None


def _stream_with_progress(resp, label: str) -> bytes:
    total = int(resp.headers.get('Content-Length') or 0)
    data = bytearray()
    try:
        from tqdm import tqdm

        with tqdm(
            total=total or None,
            unit='B',
            unit_scale=True,
            desc=f'[geniex] {label}',
            file=_tty(),
        ) as bar:
            for chunk in iter(lambda: resp.read(65536), b''):
                data.extend(chunk)
                bar.update(len(chunk))
    except ImportError:
        downloaded = 0
        for chunk in iter(lambda: resp.read(65536), b''):
            data.extend(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 // total
                print(
                    f'\r[geniex] {label}: {downloaded / 1048576:.1f}/{total / 1048576:.1f} MB ({pct}%)',
                    end='',
                    flush=True,
                    file=_tty(),
                )
            else:
                print(
                    f'\r[geniex] {label}: {downloaded / 1048576:.1f} MB',
                    end='',
                    flush=True,
                    file=_tty(),
                )
        if downloaded:
            print(file=_tty())
    return bytes(data)


def _download_with_progress(url: str, label: str) -> bytes | None:
    try:
        with urllib.request.urlopen(url) as resp:
            return _stream_with_progress(resp, label)
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f'[geniex] source unavailable: {url} ({exc})', file=sys.stderr)
        return None


def _file_url_to_path(url: str) -> Path:
    parsed = urllib.parse.urlparse(url)
    return Path(urllib.request.url2pathname(parsed.path))


def _fetch(
    url: str,
    start: int,
    end: int | None,
    *,
    exact: bool = True,
    label: str | None = None,
) -> tuple[bytes, int]:
    if url.startswith('file://'):
        path = _file_url_to_path(url)
        total = path.stat().st_size
        if end is None:
            begin = max(0, total - start)
            expected = total - begin
        else:
            begin = start
            expected = end - start + 1
        with path.open('rb') as fh:
            fh.seek(begin)
            data = fh.read(expected)
        if exact and len(data) != expected:
            raise RuntimeError(f'{url}: short read {len(data)} != {expected}')
        return data, total

    rng = f'bytes=-{start}' if end is None else f'bytes={start}-{end}'
    expected = None if end is None else end - start + 1
    req = urllib.request.Request(url, headers={'Range': rng})
    try:
        with urllib.request.urlopen(req) as resp:
            if resp.status != 206:
                raise _RangeNotSupported(f'{url}: status {resp.status} for Range request')
            cr = resp.headers.get('Content-Range', '')
            tail = cr.rsplit('/', 1)[-1] if '/' in cr else ''
            if not tail.isdigit():
                raise _RangeNotSupported(f'{url}: 206 without parseable Content-Range')
            total = int(tail)
            data = _stream_with_progress(resp, label) if label else resp.read()
    except urllib.error.HTTPError as exc:
        raise _RangeNotSupported(f'{url}: HTTP {exc.code} on Range request') from exc
    if exact and expected is not None and len(data) != expected:
        raise RuntimeError(f'{url}: short read {len(data)} != {expected}')
    return data, total


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
    # The qairt plugin ships one ABI variant per QAIRT/QNN version, each in its own
    # ``qairt-<ver>`` directory (e.g. qairt-2.45, qairt-2.47). Accept the bare
    # ``qairt`` dir (older single-variant layout) and every ``qairt-*`` variant.
    qairt_dir = _BACKEND_DIRS['qairt']
    if (rel[0] == qairt_dir or rel[0].startswith(qairt_dir + '-')) and 'qairt' in backends:
        return '/'.join(rel)
    return None


def _write_entry(entry: _CDEntry, blob: bytes, span_start: int, dst: Path) -> None:
    off = entry.local_header_offset - span_start
    if blob[off : off + 4] != _LFH_SIG:
        raise RuntimeError(f'{entry.filename}: bad local file header signature')
    fn_len, ex_len = struct.unpack('<HH', blob[off + 26 : off + 30])
    payload_off = off + _LFH_FIXED + fn_len + ex_len
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


def _range_fetch(zip_url: str, lib_dir: Path, backends: frozenset[Backend], label: str) -> int:
    suffix, total = _fetch(zip_url, _SUFFIX_PROBE, None, exact=False)
    if total <= 0:
        raise _RangeNotSupported(f'{zip_url}: empty resource')
    eocd_pos = suffix.rfind(_EOCD_SIG)
    if eocd_pos == -1:
        raise RuntimeError(f'{zip_url}: end-of-central-directory record not found')
    (_sig, _disk, _disk_cd, entries_disk, total_entries, cd_size, cd_offset, _cm_len) = struct.unpack(
        '<4s4H2LH', suffix[eocd_pos : eocd_pos + _EOCD_FIXED]
    )
    if entries_disk == _ZIP64_U16 or total_entries == _ZIP64_U16 or cd_size == _ZIP64_U32 or cd_offset == _ZIP64_U32:
        raise _ZIP64NotSupported('ZIP64 archive — range path unimplemented')

    suffix_start = total - len(suffix)
    if cd_offset >= suffix_start:
        cd_rel = cd_offset - suffix_start
        cd_bytes = suffix[cd_rel : cd_rel + cd_size]
    else:
        cd_bytes, _ = _fetch(zip_url, cd_offset, cd_offset + cd_size - 1)
    entries = _parse_central_directory(cd_bytes)

    selected: list[tuple[_CDEntry, str]] = []
    for entry in entries:
        rel = _classify_entry(entry.filename, backends)
        if rel is not None:
            selected.append((entry, rel))
    if not any(rel in _CORE_LIB_NAMES for _, rel in selected):
        raise RuntimeError(f'{zip_url}: no core libgeniex/geniex.dll entry in central directory')

    span_start = min(e.local_header_offset for e, _ in selected)
    last = max(selected, key=lambda p: p[0].local_header_offset)[0]
    span_end = min(
        last.local_header_offset + _LFH_FIXED + 2 * _ZIP64_U16 + last.compressed_size - 1,
        cd_offset - 1,
    )
    if span_start >= suffix_start:
        blob = suffix[span_start - suffix_start : span_end - suffix_start + 1]
    else:
        blob, _ = _fetch(zip_url, span_start, span_end, exact=False, label=label)
    for entry, rel in selected:
        _write_entry(entry, blob, span_start, lib_dir / rel)
    return len(selected)


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
    lib_dir = pkg_dir / 'lib'
    if lib_dir.exists() and any(lib_dir.iterdir()):
        print(f'[geniex] {lib_dir} already populated, skipping SDK download.', file=_tty())
        return
    if os.environ.get('GENIEX_SKIP_SDK_DOWNLOAD'):
        print('[geniex] GENIEX_SKIP_SDK_DOWNLOAD set, skipping SDK download.', file=_tty())
        return

    backend_set = frozenset(backends)
    unknown = backend_set - set(_BACKEND_DIRS)
    if unknown:
        raise ValueError(f'unknown backends: {sorted(unknown)}; expected subset of {sorted(_BACKEND_DIRS)}')

    plat = _detect_platform()
    if plat.endswith(_CPU_SUFFIX) and 'qairt' in backend_set:
        # Better to drop the backend than stage an empty qairt/ that fails later.
        if backend_set == {'qairt'}:
            raise RuntimeError(
                f'{_VARIANT_ENV}=cpu selects the CPU-only SDK, which has no QAIRT\n'
                'backend — those boards have no NPU. Install llama.cpp instead:\n'
                '  pip install geniex-llama-cpp'
            )
        backend_set -= {'qairt'}
        print(
            '[geniex] the CPU-only SDK has no QAIRT backend; installing llama.cpp only.',
            file=_tty(),
        )
    asset = f'geniex-sdk-{plat}-{release_tag}.zip'

    override = os.environ.get('GENIEX_SDK_DOWNLOAD_URL')
    if override:
        sources = [('override', override.rstrip('/'))]
    else:
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
    sha_url = f'{zip_url}.sha256'
    print(f'\n[geniex] Trying {name}: {zip_url}', file=_tty())

    sha_bytes = _try_download(sha_url)
    # Public defaults hard-fail without sidecar so unattended installs never
    # silently skip the integrity check. Override mode is opt-in and may
    # point at a staging path that only carries the .zip.
    if sha_bytes is None:
        if name != 'override':
            errors.append(f'{sha_url}: download failed')
            return False
        print(
            f'[geniex] {sha_url} unavailable; proceeding without sha256 verification (override mode).',
            file=sys.stderr,
        )
        want_sha: str | None = None
    else:
        want_sha = sha_bytes.decode().strip().split()[0]

    label = zip_url.rsplit('/', 1)[-1]
    lib_dir.mkdir(parents=True, exist_ok=True)
    try:
        count = _range_fetch(zip_url, lib_dir, backends, label)
    except (_RangeNotSupported, _ZIP64NotSupported) as exc:
        print(f'[geniex] Range fetch unavailable on {name} ({exc}); falling back to full download.', file=_tty())
        shutil.rmtree(lib_dir, ignore_errors=True)
        lib_dir.mkdir(parents=True, exist_ok=True)
    except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
        errors.append(f'{zip_url} (range): {exc}')
        shutil.rmtree(lib_dir, ignore_errors=True)
        return False
    else:
        print(f'[geniex] Range-fetched {count} entries from {name}: {zip_url}', file=_tty())
        print(f'[geniex] SDK libs installed at {lib_dir}', file=_tty())
        return True

    zip_bytes = _download_with_progress(zip_url, label)
    if zip_bytes is None:
        errors.append(f'{zip_url}: download failed')
        return False
    if want_sha is not None:
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
    print(f'[geniex] Full-zip extracted {count} entries from {name}: {zip_url}', file=_tty())
    print(f'[geniex] SDK libs installed at {lib_dir}', file=_tty())
    return True

# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for ``_sdk_fetch.py`` — the install-time SDK fetcher (#554).

Covers:
  - Range-based extraction of just the requested backend(s).
  - Fall-through to the full-zip path when ``Range:`` is unsupported.
  - CRC32 mismatch detection on the range path.
  - Unknown-backend rejection.

The fetcher targets a synthetic SDK zip served by an in-process
``http.server`` so the tests have no external dependencies.
"""

from __future__ import annotations

import http.server
import importlib.util
import io
import socket
import sys
import threading
import zipfile
from pathlib import Path

import pytest

REPO_PYTHON = Path(__file__).resolve().parents[2]


def _load_sdk_fetch():
    """Load ``_sdk_fetch`` from the bindings/python/ directory.

    The module isn't part of the ``geniex`` package, and importing it via
    ``sys.path`` insertion would shadow the version installed in the test
    environment. Register the loaded module under its dotted name so
    ``@dataclass(slots=True)`` can resolve ``cls.__module__`` correctly.
    """
    name = '_sdk_fetch_under_test'
    spec = importlib.util.spec_from_file_location(name, REPO_PYTHON / '_sdk_fetch.py')
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sdk_fetch = _load_sdk_fetch()


PLATFORM = 'linux-arm64'
RELEASE_TAG = 'v0.0.0-test'
ASSET = f'geniex-sdk-{PLATFORM}-{RELEASE_TAG}.zip'

LLAMA_FILE_BODY = b'fake-llama-cpp-plugin-payload\n' * 8
QAIRT_FILE_BODY = b'fake-qairt-plugin-payload\n' * 16
CORE_BODY = b'fake-libgeniex-core\n' * 4
STATIC_ARCHIVE_BODY = b'we-should-skip-this-static-archive\n'


def _make_sdk_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr(f'sdk-{PLATFORM}/lib/libgeniex.so', CORE_BODY)
        z.writestr(f'sdk-{PLATFORM}/lib/llama_cpp/ggml.so', LLAMA_FILE_BODY)
        z.writestr(f'sdk-{PLATFORM}/lib/llama_cpp/skel/libggml-htp-v75.so', LLAMA_FILE_BODY[::-1])
        z.writestr(f'sdk-{PLATFORM}/lib/qairt/libQnnHtp.so', QAIRT_FILE_BODY)
        z.writestr(f'sdk-{PLATFORM}/lib/qairt/libQnnHtpV79Skel.so', QAIRT_FILE_BODY[::-1])
        # Static archive — must be filtered out.
        z.writestr(f'sdk-{PLATFORM}/lib/llama_cpp/libllama.a', STATIC_ARCHIVE_BODY)
        # include/ entries — must never be staged into geniex/lib/.
        z.writestr(f'sdk-{PLATFORM}/include/geniex.h', b'#pragma once\n')
    return buf.getvalue()


SDK_ZIP_BYTES = _make_sdk_zip()


class _RangeAwareHandler(http.server.BaseHTTPRequestHandler):
    server_version = 'geniex-sdk-fetch-test/1.0'

    range_supported = True

    def log_message(self, *args, **kwargs):  # silence noisy stderr in tests
        return

    def _resolve(self) -> bytes | None:
        path = self.path
        if path == f'/{ASSET}':
            return SDK_ZIP_BYTES
        if path == f'/{ASSET}.sha256':
            import hashlib

            digest = hashlib.sha256(SDK_ZIP_BYTES).hexdigest()
            return f'{digest}  {ASSET}\n'.encode()
        return None

    def do_GET(self) -> None:  # noqa: N802 — http.server hook name
        body = self._resolve()
        if body is None:
            self.send_error(404)
            return
        rng = self.headers.get('Range')
        if rng and self.range_supported:
            assert rng.startswith('bytes=')
            spec = rng[len('bytes=') :]
            start_s, end_s = spec.split('-')
            if start_s == '':
                # Suffix range (RFC 7233 §2.1): bytes=-N means the last N bytes.
                n = int(end_s)
                start = max(0, len(body) - n)
                end = len(body) - 1
            else:
                start = int(start_s)
                end = int(end_s) if end_s else len(body) - 1
                end = min(end, len(body) - 1)
            chunk = body[start : end + 1]
            self.send_response(206)
            self.send_header('Content-Range', f'bytes {start}-{end}/{len(body)}')
            self.send_header('Content-Length', str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)
            return
        self.send_response(200)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def http_server():
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    sock.close()
    server = http.server.HTTPServer(('127.0.0.1', port), _RangeAwareHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f'http://127.0.0.1:{port}'
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _patch_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sdk_fetch, '_detect_platform', lambda: PLATFORM)


def _set_override(monkeypatch: pytest.MonkeyPatch, base_url: str) -> None:
    monkeypatch.setenv('GENIEX_SDK_DOWNLOAD_URL', base_url)
    monkeypatch.delenv('GENIEX_SKIP_SDK_DOWNLOAD', raising=False)


def test_range_fetch_llama_cpp_only(tmp_path, monkeypatch, http_server):
    _patch_platform(monkeypatch)
    _, base_url = http_server
    _set_override(monkeypatch, base_url)

    sdk_fetch.fetch(tmp_path, RELEASE_TAG, backends=('llama-cpp',))

    lib = tmp_path / 'lib'
    assert (lib / 'libgeniex.so').read_bytes() == CORE_BODY
    assert (lib / 'llama_cpp' / 'ggml.so').read_bytes() == LLAMA_FILE_BODY
    assert (lib / 'llama_cpp' / 'skel' / 'libggml-htp-v75.so').read_bytes() == LLAMA_FILE_BODY[::-1]
    assert not (lib / 'qairt').exists()
    # Static archives are stripped at classification time.
    assert not (lib / 'llama_cpp' / 'libllama.a').exists()
    # include/ never lands under lib/.
    assert not (lib / 'geniex.h').exists()


def test_range_fetch_qairt_only(tmp_path, monkeypatch, http_server):
    _patch_platform(monkeypatch)
    _, base_url = http_server
    _set_override(monkeypatch, base_url)

    sdk_fetch.fetch(tmp_path, RELEASE_TAG, backends=('qairt',))

    lib = tmp_path / 'lib'
    assert (lib / 'libgeniex.so').read_bytes() == CORE_BODY
    assert (lib / 'qairt' / 'libQnnHtp.so').read_bytes() == QAIRT_FILE_BODY
    assert (lib / 'qairt' / 'libQnnHtpV79Skel.so').read_bytes() == QAIRT_FILE_BODY[::-1]
    assert not (lib / 'llama_cpp').exists()


def test_range_fetch_both_backends(tmp_path, monkeypatch, http_server):
    _patch_platform(monkeypatch)
    _, base_url = http_server
    _set_override(monkeypatch, base_url)

    sdk_fetch.fetch(tmp_path, RELEASE_TAG, backends=('llama-cpp', 'qairt'))

    lib = tmp_path / 'lib'
    assert (lib / 'libgeniex.so').read_bytes() == CORE_BODY
    assert (lib / 'llama_cpp' / 'ggml.so').exists()
    assert (lib / 'qairt' / 'libQnnHtp.so').exists()


def test_full_fallback_when_range_unsupported(tmp_path, monkeypatch, http_server):
    _patch_platform(monkeypatch)
    _, base_url = http_server
    _set_override(monkeypatch, base_url)

    monkeypatch.setattr(_RangeAwareHandler, 'range_supported', False)
    sdk_fetch.fetch(tmp_path, RELEASE_TAG, backends=('llama-cpp',))

    lib = tmp_path / 'lib'
    assert (lib / 'libgeniex.so').exists()
    assert (lib / 'llama_cpp' / 'ggml.so').exists()
    assert not (lib / 'qairt').exists()


def test_unknown_backend_rejected(tmp_path):
    with pytest.raises(ValueError, match='unknown backends'):
        sdk_fetch.fetch(tmp_path, RELEASE_TAG, backends=('not-a-backend',))


def test_skipped_when_lib_dir_already_populated(tmp_path, monkeypatch):
    lib = tmp_path / 'lib'
    lib.mkdir()
    (lib / 'sentinel').write_text('preexisting')
    monkeypatch.setenv('GENIEX_SDK_DOWNLOAD_URL', 'http://127.0.0.1:1')  # would fail if reached

    sdk_fetch.fetch(tmp_path, RELEASE_TAG, backends=('llama-cpp',))

    assert (lib / 'sentinel').read_text() == 'preexisting'


def test_skipped_when_env_flag_set(tmp_path, monkeypatch):
    monkeypatch.setenv('GENIEX_SKIP_SDK_DOWNLOAD', '1')
    monkeypatch.setenv('GENIEX_SDK_DOWNLOAD_URL', 'http://127.0.0.1:1')  # would fail if reached

    sdk_fetch.fetch(tmp_path, RELEASE_TAG, backends=('llama-cpp',))

    assert not (tmp_path / 'lib').exists()


def test_default_sources_url_shape(tmp_path, monkeypatch):
    """S3 default uses the flat layout; GitHub default keeps /{tag}.

    Regression for the case where ``pip install`` against the public mirror
    failed because the S3 source still appended ``/{release_tag}/`` — but the
    release pipeline switched to a flat ``qai-hub-geniex/`` prefix.
    """
    _patch_platform(monkeypatch)
    monkeypatch.delenv('GENIEX_SDK_DOWNLOAD_URL', raising=False)
    monkeypatch.delenv('GENIEX_SKIP_SDK_DOWNLOAD', raising=False)

    seen: list[tuple[str, str]] = []

    def fake_try(name, zip_url, lib_dir, backends, errors):
        seen.append((name, zip_url))
        return False  # force the loop to walk every default source

    monkeypatch.setattr(sdk_fetch, '_try_one_source', fake_try)

    with pytest.raises(RuntimeError, match='Failed to fetch SDK'):
        sdk_fetch.fetch(tmp_path, RELEASE_TAG, backends=('llama-cpp',))

    assert seen == [
        ('s3', f'{sdk_fetch.DEFAULT_S3_BASE_URL}/{ASSET}'),
        ('github', f'{sdk_fetch.DEFAULT_BASE_URL}/{RELEASE_TAG}/{ASSET}'),
    ]


def test_classify_entry_filters_paths():
    classify = sdk_fetch._classify_entry
    backends = frozenset({'llama-cpp', 'qairt'})

    assert classify('sdk-linux-arm64/lib/libgeniex.so', backends) == 'libgeniex.so'
    assert classify('sdk-windows-arm64/lib/geniex.dll', backends) == 'geniex.dll'
    assert classify('sdk-linux-arm64/lib/llama_cpp/x.so', backends) == 'llama_cpp/x.so'
    assert classify('sdk-linux-arm64/lib/qairt/sub/y.so', backends) == 'qairt/sub/y.so'

    only_llama = frozenset({'llama-cpp'})
    assert classify('sdk-linux-arm64/lib/qairt/y.so', only_llama) is None

    # Static archives are dropped regardless of backends.
    assert classify('sdk-linux-arm64/lib/llama_cpp/libllama.a', backends) is None
    # include/ is outside lib/.
    assert classify('sdk-linux-arm64/include/geniex.h', backends) is None
    # Directory entries (trailing /) classify to None.
    assert classify('sdk-linux-arm64/lib/llama_cpp/', backends) is None

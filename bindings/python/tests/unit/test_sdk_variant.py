# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit coverage for CPU-only SDK asset selection (#1217)."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def fetcher(monkeypatch):
    """A fresh ``_sdk_fetch``, imported by path the way ``setup.py`` does."""
    spec = importlib.util.spec_from_file_location('_sdk_fetch_under_test', _ROOT / '_sdk_fetch.py')
    mod = importlib.util.module_from_spec(spec)
    # @dataclass looks its own module up in sys.modules while the class is built.
    monkeypatch.setitem(sys.modules, spec.name, mod)
    spec.loader.exec_module(mod)
    # Pretend we're on linux/aarch64. Swap the module's `sys` rather than the real
    # one: a global sys.platform='linux' would send pytest's tmp_path down the
    # POSIX path on Windows and into os.getuid().
    monkeypatch.setattr(mod, 'sys', types.SimpleNamespace(platform='linux', stderr=sys.stderr))
    monkeypatch.setattr(mod, 'platform', types.SimpleNamespace(machine=lambda: 'aarch64'))
    monkeypatch.delenv(mod._VARIANT_ENV, raising=False)
    return mod


@pytest.fixture
def staged(fetcher, monkeypatch):
    """Capture the asset and backend set fetch() would have downloaded."""
    seen: dict = {}

    def _capture(name, zip_url, lib_dir, backends, errors):
        seen['asset'] = zip_url.rsplit('/', 1)[-1]
        seen['backends'] = set(backends)
        return True

    monkeypatch.setattr(fetcher, '_try_one_source', _capture)
    return seen


@pytest.mark.parametrize(
    ('variant', 'expected'),
    [(None, 'linux-arm64'), ('default', 'linux-arm64'), ('cpu', 'linux-arm64-cpu')],
)
def test_variant_picks_the_asset(fetcher, monkeypatch, variant, expected):
    if variant is not None:
        monkeypatch.setenv(fetcher._VARIANT_ENV, variant)
    assert fetcher._detect_platform() == expected


def test_unknown_variant_is_rejected(fetcher, monkeypatch):
    monkeypatch.setenv(fetcher._VARIANT_ENV, 'armv9')
    with pytest.raises(RuntimeError, match='default, cpu'):
        fetcher._detect_platform()


def test_the_variant_is_linux_arm64_only(fetcher, monkeypatch):
    fetcher.sys.platform = 'win32'
    monkeypatch.setattr(fetcher, 'platform', types.SimpleNamespace(machine=lambda: 'ARM64'))
    monkeypatch.setenv(fetcher._VARIANT_ENV, 'cpu')
    assert fetcher._detect_platform() == 'windows-arm64'


def test_meta_sdist_drops_qairt(fetcher, monkeypatch, staged, tmp_path):
    monkeypatch.setenv(fetcher._VARIANT_ENV, 'cpu')
    fetcher.fetch(tmp_path, 'v1.2.3', backends=('llama-cpp', 'qairt'))
    assert staged['asset'] == 'geniex-sdk-linux-arm64-cpu-v1.2.3.zip'
    assert staged['backends'] == {'llama-cpp'}


def test_qairt_only_sdist_is_rejected(fetcher, monkeypatch, staged, tmp_path):
    monkeypatch.setenv(fetcher._VARIANT_ENV, 'cpu')
    with pytest.raises(RuntimeError, match='no QAIRT'):
        fetcher.fetch(tmp_path, 'v1.2.3', backends=('qairt',))
    assert staged == {}


def test_default_variant_keeps_qairt(fetcher, staged, tmp_path):
    fetcher.fetch(tmp_path, 'v1.2.3', backends=('llama-cpp', 'qairt'))
    assert staged['asset'] == 'geniex-sdk-linux-arm64-v1.2.3.zip'
    assert staged['backends'] == {'llama-cpp', 'qairt'}

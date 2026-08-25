# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the tqdm-backed download progress printer."""

from __future__ import annotations

import pytest

from geniex import _progress
from geniex.model_manager import FileProgress


def _files(got: int = 100, total: int = 1000, name: str = 'a.gguf') -> list[FileProgress]:
    return [FileProgress(file_name=name, downloaded_bytes=got, total_bytes=total)]


def test_resolve_false_returns_none():
    assert _progress.resolve(False) is None


def test_resolve_callable_passthrough():
    cb = lambda files: True  # noqa: E731
    assert _progress.resolve(cb) is cb


def test_resolve_none_returns_default():
    p = _progress.resolve(None)
    try:
        assert callable(p)
    finally:
        _progress.finish(p)


def test_resolve_rejects_non_callable():
    with pytest.raises(TypeError):
        _progress.resolve(123)  # type: ignore[arg-type]


def test_finish_noop_on_none():
    _progress.finish(None)


def test_finish_noop_on_plain_callable():
    _progress.finish(lambda files: True)


def test_tqdm_progress_creates_bar_per_file():
    p = _progress._TqdmProgress()
    try:
        p([FileProgress('a.gguf', 10, 100), FileProgress('b.json', 5, 20)])
        assert set(p._bars.keys()) == {'a.gguf', 'b.json'}
        assert p._bars['a.gguf'].n == 10
        assert p._bars['a.gguf'].total == 100
    finally:
        p.finish()
    assert p._bars == {}


def test_tqdm_progress_seeds_initial_from_downloaded():
    p = _progress._TqdmProgress()
    try:
        # Resuming a partial download: the first callback already reports a
        # large downloaded count. tqdm's rate must exclude those pre-existing
        # bytes, so the bar is seeded with initial=downloaded_bytes.
        p([FileProgress('a.gguf', 800, 1000)])
        bar = p._bars['a.gguf']
        assert bar.format_dict['initial'] == 800
        assert bar.n == 800
        p([FileProgress('a.gguf', 850, 1000)])
        assert bar.n == 850
    finally:
        p.finish()


def test_tqdm_progress_updates_existing_bar():
    p = _progress._TqdmProgress()
    try:
        p(_files(got=10))
        p(_files(got=50))
        assert p._bars['a.gguf'].n == 50
    finally:
        p.finish()


def test_tqdm_progress_handles_unknown_total():
    p = _progress._TqdmProgress()
    try:
        p([FileProgress('a.gguf', 10, -1)])
        assert p._bars['a.gguf'].total is None
        # Total becomes known later in the stream.
        p([FileProgress('a.gguf', 50, 100)])
        assert p._bars['a.gguf'].total == 100
    finally:
        p.finish()


def test_default_progress_printer_returns_callable():
    p = _progress.default_progress_printer()
    try:
        assert callable(p)
        assert p(_files()) is True
    finally:
        _progress.finish(p)

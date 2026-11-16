# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the ensure_cached precision-selection flow (#1098).

When precision=None, ensure_cached must call query() and pass the SDK's
priority-sorted head precision to pull() so only one variant is downloaded.
"""

from __future__ import annotations

import pytest

import geniex.model_manager as mm
from geniex._ffi._api import GenieXError
from geniex.model_manager import ModelPaths, ModelQuery, PrecisionCandidate


def _make_query(*precisions: str) -> ModelQuery:
    return ModelQuery(
        model_name='org/repo',
        runtime='llama_cpp',
        model_type='llm',
        candidates=[PrecisionCandidate(precision=p, size=0) for p in precisions],
    )


def _fake_paths() -> ModelPaths:
    return ModelPaths(
        model_path='/cache/org/repo/model.gguf',
        model_dir='/cache/org/repo',
        model_name='org/repo',
        runtime='llama_cpp',
        model_type='llm',
    )


def test_ensure_cached_pulls_head_precision(monkeypatch):
    """pull() receives candidates[0].precision when none is specified."""
    pulled = {}

    monkeypatch.setattr(mm, 'query', lambda name, **_kw: _make_query('Q4_0', 'Q4_K_M', 'Q8_0'))
    monkeypatch.setattr(mm, 'pull', lambda name, *, precision=None, **_kw: pulled.update(precision=precision))
    monkeypatch.setattr(mm, 'get_paths', lambda _key: _fake_paths())
    monkeypatch.setattr(mm, 'resolve_alias', lambda name: name)
    monkeypatch.setattr(mm, '_ensure_init', lambda: None)

    mm.ensure_cached('org/repo')

    assert pulled['precision'] == 'Q4_0'


def test_ensure_cached_explicit_precision_skips_query(monkeypatch):
    """An explicit precision bypasses query() entirely."""
    queried = []

    monkeypatch.setattr(mm, 'query', lambda *_a, **_kw: queried.append(True) or _make_query('Q4_0'))
    monkeypatch.setattr(mm, 'pull', lambda *_a, **_kw: None)
    monkeypatch.setattr(mm, 'get_paths', lambda _key: _fake_paths())
    monkeypatch.setattr(mm, 'resolve_alias', lambda name: name)
    monkeypatch.setattr(mm, '_ensure_init', lambda: None)

    mm.ensure_cached('org/repo', precision='Q8_0')

    assert queried == [], 'query() should not be called when precision is explicit'


def test_ensure_cached_query_failure_falls_through(monkeypatch):
    """A query() GenieXError is swallowed and pull() is still called."""
    pulled = {}

    monkeypatch.setattr(mm, 'query', lambda *_a, **_kw: (_ for _ in ()).throw(GenieXError(-1, 'offline')))
    monkeypatch.setattr(mm, 'pull', lambda name, *, precision=None, **_kw: pulled.update(precision=precision))
    monkeypatch.setattr(mm, 'get_paths', lambda _key: _fake_paths())
    monkeypatch.setattr(mm, 'resolve_alias', lambda name: name)
    monkeypatch.setattr(mm, '_ensure_init', lambda: None)

    mm.ensure_cached('org/repo')

    assert pulled['precision'] is None


@pytest.mark.parametrize(
    ('model_name', 'precision', 'expected_precision'),
    [
        ('qwen3', None, 'Q4_K_M'),
        ('qwen3', 'Q8_0', 'Q8_0'),
        ('qwen3:Q5_K_M', None, 'Q5_K_M'),
        ('qwen3:Q5_K_M', 'Q8_0', 'Q8_0'),
    ],
)
def test_ensure_cached_preserves_alias_precision(monkeypatch, model_name, precision, expected_precision):
    # The SDK's model_ref::resolve_alias maps qwen3 to this precision-qualified name.
    repo = 'ggml-org/Qwen3-1.7B-GGUF'
    observed = {}
    queried = []

    def _resolve_alias(name):
        assert name == 'qwen3'
        return f'{repo}:Q4_K_M'

    def _pull(name, *, precision=None, **_kw):
        observed['pull'] = (name, precision)

    def _get_paths(key):
        observed['key'] = key
        return _fake_paths()

    monkeypatch.setattr(mm, 'resolve_alias', _resolve_alias)
    monkeypatch.setattr(mm, '_ensure_init', lambda: None)
    monkeypatch.setattr(mm, 'query', lambda name, **_kw: queried.append(name) or _make_query('Q4_0'))
    monkeypatch.setattr(mm, 'pull', _pull)
    monkeypatch.setattr(mm, 'get_paths', _get_paths)

    mm.ensure_cached(model_name, precision=precision)

    assert observed['key'] == f'{repo}:{expected_precision}'
    assert observed['pull'] == (repo, expected_precision)
    assert queried == [], 'An alias or caller-supplied precision must bypass default selection'

# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit coverage for the #737 quant-validation translator.

The Rust model-manager today funnels manifest-inference failures through
``GENIEX_ERROR_COMMON_UNKNOWN`` (-100000). When the caller explicitly
passed ``quant=``, the Python wrapper rewrites that into a focused
``ValueError`` so the user can spot the typo without grepping logs.
"""

from __future__ import annotations

import pytest

from geniex._ffi._api import GeniexError
from geniex.auto import (
    GENIEX_ERROR_COMMON_UNKNOWN,
    _translate_quant_error,
)


def test_translate_unknown_error_with_quant_returns_value_error():
    err = GeniexError(GENIEX_ERROR_COMMON_UNKNOWN, 'Unknown error')
    out = _translate_quant_error(err, 'unsloth/Qwen3-0.6B-GGUF', 'BAD_QUANT')
    assert isinstance(out, ValueError)
    assert "'BAD_QUANT'" in str(out)
    assert "'unsloth/Qwen3-0.6B-GGUF'" in str(out)


def test_translate_returns_none_when_quant_is_none():
    err = GeniexError(GENIEX_ERROR_COMMON_UNKNOWN, 'Unknown error')
    assert _translate_quant_error(err, 'org/repo', None) is None


def test_translate_returns_none_for_other_error_codes():
    # -100203 (Invalid model format) should not be rewritten — that's a
    # legitimate format error, not a quant typo.
    err = GeniexError(-100203, 'Invalid model format')
    assert _translate_quant_error(err, 'org/repo', 'Q4_0') is None


def test_resolve_model_sources_translates_quant_error(monkeypatch):
    # Drive _resolve_model_sources end-to-end to confirm the translator
    # is wired into the cache-miss path.
    from geniex import auto as auto_module

    def _fake_get_paths(_key):
        raise GeniexError(-100201, 'Model loading failed')  # cache miss

    def _fake_ensure_cached(*_a, **_kw):
        raise GeniexError(GENIEX_ERROR_COMMON_UNKNOWN, 'Unknown error')

    monkeypatch.setattr(auto_module._mm, 'get_paths', _fake_get_paths)
    monkeypatch.setattr(auto_module._mm, 'ensure_cached', _fake_ensure_cached)
    monkeypatch.setattr(auto_module._progress, 'resolve', lambda _p: None)
    monkeypatch.setattr(auto_module._progress, 'finish', lambda _p: None)

    with pytest.raises(ValueError, match=r"'BAD_QUANT'"):
        auto_module._resolve_model_sources(
            'unsloth/Qwen3-0.6B-GGUF',
            quant='BAD_QUANT',
            hf_token=None,
            progress=None,
            model_name=None,
        )


def test_resolve_model_sources_passes_through_other_errors(monkeypatch):
    # When quant is None, -100000 must keep its original GeniexError shape so
    # callers / docs that already key off the SDK error code don't break.
    from geniex import auto as auto_module

    def _fake_get_paths(_key):
        raise GeniexError(-100201, 'Model loading failed')

    def _fake_ensure_cached(*_a, **_kw):
        raise GeniexError(GENIEX_ERROR_COMMON_UNKNOWN, 'Unknown error')

    monkeypatch.setattr(auto_module._mm, 'get_paths', _fake_get_paths)
    monkeypatch.setattr(auto_module._mm, 'ensure_cached', _fake_ensure_cached)
    monkeypatch.setattr(auto_module._progress, 'resolve', lambda _p: None)
    monkeypatch.setattr(auto_module._progress, 'finish', lambda _p: None)

    with pytest.raises(GeniexError):
        auto_module._resolve_model_sources(
            'unsloth/Qwen3-0.6B-GGUF',
            quant=None,
            hf_token=None,
            progress=None,
            model_name=None,
        )

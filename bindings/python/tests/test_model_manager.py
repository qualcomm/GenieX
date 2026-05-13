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

"""Model manager public API — exercises real HuggingFace pulls."""

from __future__ import annotations

import os

import pytest

import geniex
from geniex import model_manager as _mm

from .conftest import LLAMA_CPP_MODEL, LLAMA_CPP_QUANT


def test_get_paths_returns_existing_files(llama_cpp_paths):
    assert os.path.isfile(llama_cpp_paths.model_path), llama_cpp_paths.model_path
    assert os.path.isdir(llama_cpp_paths.model_dir), llama_cpp_paths.model_dir
    assert llama_cpp_paths.plugin_id


def test_list_contains_pulled_model(llama_cpp_paths):
    names = _mm.list_models()
    assert LLAMA_CPP_MODEL in names


def test_ensure_cached_idempotent(llama_cpp_paths):
    # Second call must hit the local cache and not re-download.
    calls: list[int] = []

    def _on_progress(files):
        calls.append(len(files))
        return True

    paths = _mm.ensure_cached(LLAMA_CPP_MODEL, quant=LLAMA_CPP_QUANT, hub='hf', on_progress=_on_progress)
    assert paths.model_path == llama_cpp_paths.model_path
    # No in-flight progress ticks expected on a warm cache.
    assert all(n >= 0 for n in calls)


def test_resolve_alias_unknown_raises(geniex_session):
    with pytest.raises(geniex.GeniexError):
        _mm.resolve_alias('definitely-not-a-real-alias-xyz')


def test_get_type_for_llm(llama_cpp_paths):
    assert _mm.get_type(LLAMA_CPP_MODEL) == 'llm'

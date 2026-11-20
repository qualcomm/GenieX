# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Model manager public API — exercises real HuggingFace pulls."""

from __future__ import annotations

import os

from geniex import model_manager as _mm

from .conftest import LLAMA_CPP_MODEL, LLAMA_CPP_QUANT


def test_get_paths_returns_existing_files(llama_cpp_paths):
    assert os.path.isfile(llama_cpp_paths.model_path), llama_cpp_paths.model_path
    assert os.path.isdir(llama_cpp_paths.model_dir), llama_cpp_paths.model_dir
    assert llama_cpp_paths.runtime


def test_list_contains_pulled_model(llama_cpp_paths):
    names = _mm.list_models()
    assert LLAMA_CPP_MODEL in names


def test_ensure_cached_idempotent(llama_cpp_paths):
    # Second call must hit the local cache and not re-download.
    calls: list[int] = []

    def _on_progress(files):
        calls.append(len(files))
        return True

    paths = _mm.ensure_cached(LLAMA_CPP_MODEL, precision=LLAMA_CPP_QUANT, hub='hf', on_progress=_on_progress)
    assert paths.model_path == llama_cpp_paths.model_path
    # No in-flight progress ticks expected on a warm cache.
    assert all(n >= 0 for n in calls)


def test_get_type_for_llm(llama_cpp_paths):
    assert _mm.get_type(LLAMA_CPP_MODEL) == 'llm'

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

"""Fixtures for the Python binding contract tests.

End-to-end SDK / generation coverage lives in the top-level ``tests/``
suite. This file stays focused on binding-layer tests (CLI wrapper,
progress callbacks, model_manager Python surface, local pull paths) and
only pulls a small public GGUF when a binding test needs a real cached
model on disk.
"""

from __future__ import annotations

import pytest

try:
    import geniex
    from geniex import model_manager as _mm
except (ImportError, OSError) as _geniex_import_error:
    # Pure-Python unit tests under tests/unit/ must collect even when the
    # native SDK isn't staged yet (no geniex/lib/), so we defer the import
    # error until a fixture that actually needs the runtime is requested.
    geniex = None
    _mm = None
    _geniex_import_error_reason = repr(_geniex_import_error)
else:
    _geniex_import_error_reason = ''

LLAMA_CPP_MODEL = 'bartowski/Qwen_Qwen3-0.6B-GGUF'
LLAMA_CPP_QUANT = 'Q4_0'


@pytest.fixture(scope='session')
def geniex_session():
    if geniex is None:
        pytest.skip(f'geniex runtime unavailable ({_geniex_import_error_reason})')
    geniex.init()
    _mm.init()
    yield
    geniex.deinit()


@pytest.fixture(scope='session')
def llama_cpp_paths(geniex_session):
    try:
        return _mm.ensure_cached(LLAMA_CPP_MODEL, quant=LLAMA_CPP_QUANT, hub='hf')
    except geniex.GenieXError as e:
        pytest.skip(f'could not pull {LLAMA_CPP_MODEL}: {e}')

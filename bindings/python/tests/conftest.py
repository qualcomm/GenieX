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

"""Shared fixtures for the geniex public-API test suite.

Tests talk to real SDK via the public ``geniex`` package only — never
``geniex._ffi`` — so the suite doubles as a public-surface contract check.
Set ``GENIEX_DEVICE_TEST=1`` to opt in to QAIRT / device-gated tests.
"""

from __future__ import annotations

import os
import platform

import pytest

import geniex
from geniex import model_manager as _mm

# Small, HF-public GGUF chosen for fast download.
LLAMA_CPP_MODEL = 'bartowski/Qwen_Qwen3-0.6B-GGUF'
LLAMA_CPP_QUANT = 'Q4_0'

# AI Hub alias — resolved by the SDK's registry on Snapdragon hosts.
QAIRT_ALIAS = 'granite4_micro'


def _is_snapdragon_host() -> bool:
    return platform.system() == 'Windows' and platform.machine().lower() in ('arm64', 'aarch64')


def _device_tests_enabled() -> bool:
    return bool(os.environ.get('GENIEX_DEVICE_TEST'))


@pytest.fixture(scope='session')
def tmp_datadir(tmp_path_factory) -> str:
    # Isolate from the user's real ~/.cache/geniex so tests can clean() freely.
    path = tmp_path_factory.mktemp('geniex-datadir')
    os.environ['GENIEX_DATADIR'] = str(path)
    return str(path)


@pytest.fixture(scope='session')
def geniex_session(tmp_datadir):
    geniex.init()
    _mm.init(tmp_datadir)
    yield
    geniex.deinit()


@pytest.fixture(scope='session')
def llama_cpp_paths(geniex_session):
    try:
        return _mm.ensure_cached(LLAMA_CPP_MODEL, quant=LLAMA_CPP_QUANT, hub='hf')
    except geniex.GeniexError as e:
        pytest.skip(f'could not pull {LLAMA_CPP_MODEL}: {e}')


@pytest.fixture(scope='session')
def qairt_paths(geniex_session):
    if not _device_tests_enabled() or not _is_snapdragon_host():
        pytest.skip('QAIRT tests require GENIEX_DEVICE_TEST=1 on a Snapdragon host')
    try:
        _mm.pull(QAIRT_ALIAS, hub='aihub')
    except geniex.GeniexError as e:
        pytest.skip(f'could not pull QAIRT model {QAIRT_ALIAS}: {e}')
    return _mm.get_paths(QAIRT_ALIAS)

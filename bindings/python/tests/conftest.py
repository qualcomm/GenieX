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
import sys

import pytest

import geniex
from geniex import model_manager as _mm

# Small, HF-public GGUF chosen for fast download.
LLAMA_CPP_MODEL = 'bartowski/Qwen_Qwen3-0.6B-GGUF'
LLAMA_CPP_QUANT = 'Q4_0'

# Pre-cached QAIRT models used by the NPU tests. Override with
# GENIEX_QAIRT_MODEL / GENIEX_QAIRT_VLM_MODEL to exercise other models.
QAIRT_MODEL = os.environ.get('GENIEX_QAIRT_MODEL', 'aihub/qwen3_4b')
QAIRT_VLM_MODEL = os.environ.get('GENIEX_QAIRT_VLM_MODEL', 'aihub/Qwen2.5-VL-7B-Instruct')


def _is_snapdragon_host() -> bool:
    if platform.machine().lower() not in ('arm64', 'aarch64'):
        return False
    if platform.system() == 'Windows' or hasattr(sys, 'getandroidapilevel'):
        return True
    try:
        with open('/sys/firmware/devicetree/base/compatible', 'rb') as f:
            return b'qcom' in f.read()
    except OSError:
        return False


def _device_tests_enabled() -> bool:
    return bool(os.environ.get('GENIEX_DEVICE_TEST'))


@pytest.fixture(scope='session')
def geniex_session():
    """Init the geniex runtime + model manager against the user's real cache.

    Device tests (QAIRT especially) need pre-cached models, and the SDK
    model manager only supports one data dir per process. We therefore
    run tests against the default ``~/.cache/geniex`` and avoid any
    destructive ``clean()`` calls.
    """
    geniex.init()
    _mm.init()  # default: GENIEX_DATADIR env → ~/.cache/geniex
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
        return _mm.get_paths(QAIRT_MODEL)
    except geniex.GeniexError as e:
        pytest.skip(f'QAIRT model {QAIRT_MODEL} not cached ({e}); run `geniex-py pull` first')


@pytest.fixture(scope='session')
def qairt_vlm_paths(geniex_session):
    if not _device_tests_enabled() or not _is_snapdragon_host():
        pytest.skip('QAIRT tests require GENIEX_DEVICE_TEST=1 on a Snapdragon host')
    try:
        return _mm.get_paths(QAIRT_VLM_MODEL)
    except geniex.GeniexError as e:
        pytest.skip(f'QAIRT VLM {QAIRT_VLM_MODEL} not cached ({e}); run `geniex-py pull` first')

# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Top-level pytest fixtures for the SDK end-to-end suite.

Tests drive the SDK exclusively through the public ``geniex`` Python
binding so the suite doubles as a public-surface contract check. The
fixtures here own runtime init, model caching, and host gating shared
across ``tests/api``, ``tests/plugins/llama_cpp`` and ``tests/plugins/qairt``.
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

import geniex
import pytest
from geniex import model_manager as _mm

from _models import (
    LLAMA_CPP_LLM_MODEL,
    LLAMA_CPP_LLM_PRECISION,
    LLAMA_CPP_VLM_MODEL,
    QAIRT_LLM_MODEL,
    QAIRT_VLM_MODEL,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_IMAGE_PATH = _REPO_ROOT / 'cli' / 'server' / 'docs' / 'ui' / 'favicon-32x32.png'
QUALITY_IMAGE_PATH = Path(__file__).resolve().parent / 'assets' / 'quality_dog.jpg'

_DEVICE_MARKER = {
    'cpu': 'device_cpu',
    'gpu': 'device_gpu',
    'npu': 'device_npu',
    'hybrid': 'device_hybrid',
}
_SNAPDRAGON_DEVICES = {'gpu', 'npu', 'hybrid'}


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


def device_tests_enabled() -> bool:
    return bool(os.environ.get('GENIEX_DEVICE_TEST'))


def pytest_collection_modifyitems(config, items):
    """Auto-tag generation matrix items.

    Items parametrised with ``device_map`` pick up the matching ``device_*``
    marker; items under ``tests/plugins/<plugin>/`` get the plugin marker so
    a single ``-m "qairt"`` selects everything for that plugin without
    per-test boilerplate.
    """
    for item in items:
        try:
            rel = Path(item.fspath).resolve().relative_to(_REPO_ROOT)
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) >= 3 and parts[0] == 'tests' and parts[1] == 'plugins':
            item.add_marker(getattr(pytest.mark, parts[2]))
        if len(parts) >= 2 and parts[0] == 'tests' and parts[1] == 'api':
            item.add_marker(pytest.mark.api)

        device_map = item.callspec.params.get('device_map') if hasattr(item, 'callspec') else None
        if isinstance(device_map, str):
            marker_name = _DEVICE_MARKER.get(device_map)
            if marker_name:
                item.add_marker(getattr(pytest.mark, marker_name))
            if device_map in _SNAPDRAGON_DEVICES:
                item.add_marker(pytest.mark.snapdragon)


def pytest_runtest_setup(item):
    """Skip device-gated items unless we are actually on supported hardware."""
    markers = {m.name for m in item.iter_markers()}
    if 'snapdragon' in markers or 'qairt' in markers:
        if not device_tests_enabled():
            pytest.skip('set GENIEX_DEVICE_TEST=1 to run device-gated tests')
        if not _is_snapdragon_host():
            pytest.skip('device-gated tests require a Snapdragon host')


@pytest.fixture(scope='session')
def geniex_session():
    geniex.init()
    _mm.init()
    yield
    geniex.deinit()


@pytest.fixture(scope='session')
def llama_cpp_llm_paths(geniex_session):
    try:
        return _mm.ensure_cached(LLAMA_CPP_LLM_MODEL, precision=LLAMA_CPP_LLM_PRECISION, hub='hf')
    except geniex.GenieXError as e:
        pytest.skip(f'could not pull {LLAMA_CPP_LLM_MODEL}: {e}')


@pytest.fixture(scope='session')
def llama_cpp_vlm_paths(geniex_session):
    try:
        return _mm.ensure_cached(LLAMA_CPP_VLM_MODEL, hub='hf')
    except geniex.GenieXError as e:
        pytest.skip(f'could not pull {LLAMA_CPP_VLM_MODEL}: {e}')


@pytest.fixture(scope='session')
def qairt_llm_paths(geniex_session):
    try:
        return _mm.ensure_cached(QAIRT_LLM_MODEL)
    except geniex.GenieXError as e:
        pytest.skip(f'could not pull {QAIRT_LLM_MODEL}: {e}')


@pytest.fixture(scope='session')
def qairt_vlm_paths(geniex_session):
    try:
        return _mm.ensure_cached(QAIRT_VLM_MODEL)
    except geniex.GenieXError as e:
        pytest.skip(f'could not pull {QAIRT_VLM_MODEL}: {e}')


@pytest.fixture(scope='session')
def test_image() -> str:
    if not TEST_IMAGE_PATH.is_file():
        pytest.skip(f'test image missing: {TEST_IMAGE_PATH}')
    return str(TEST_IMAGE_PATH)


@pytest.fixture(scope='session')
def quality_image() -> str:
    # Real photographic image (golden retriever) shipped under tests/assets/.
    # Used by VLM keyword-quality tests; the favicon `test_image` is too small
    # for any meaningful caption to land on dog/grass/animal vocabulary.
    if not QUALITY_IMAGE_PATH.is_file():
        pytest.skip(f'quality image missing: {QUALITY_IMAGE_PATH}')
    return str(QUALITY_IMAGE_PATH)

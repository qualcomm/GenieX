# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Top-level pytest fixtures for the SDK end-to-end suite."""

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
    LLAMA_CPP_MTP_DRAFT_MODEL,
    LLAMA_CPP_MTP_DRAFT_PRECISION,
    LLAMA_CPP_MTP_TARGET_MODEL,
    LLAMA_CPP_MTP_TARGET_PRECISION,
    LLAMA_CPP_VLM_MODEL,
    QAIRT_LLM_MODEL,
    QAIRT_VLM_MODEL,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_IMAGE_PATH = _REPO_ROOT / 'cli' / 'server' / 'docs' / 'ui' / 'favicon-32x32.png'
QUALITY_IMAGE_PATH = Path(__file__).resolve().parent / 'assets' / 'quality_dog.jpg'

_DEVICE_MARKER = {
    'cpu': 'device_cpu',
    'npu': 'device_npu',
}
_SNAPDRAGON_DEVICES = {'npu'}


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


_PLUGIN_MARKERS = {'llama_cpp', 'qairt'}


def pytest_collection_modifyitems(config, items):
    for item in items:
        try:
            rel = Path(item.fspath).resolve().relative_to(_REPO_ROOT)
        except ValueError:
            continue
        parts = rel.parts
        # tests/test_<name>.py -> derive marker from the filename stem.
        if len(parts) == 2 and parts[0] == 'tests' and parts[1].startswith('test_'):
            stem = parts[1][len('test_') : -len('.py')]
            if stem in _PLUGIN_MARKERS:
                item.add_marker(getattr(pytest.mark, stem))
            elif stem == 'api':
                item.add_marker(pytest.mark.api)

        device_map = item.callspec.params.get('device_map') if hasattr(item, 'callspec') else None
        if isinstance(device_map, str):
            marker_name = _DEVICE_MARKER.get(device_map)
            if marker_name:
                item.add_marker(getattr(pytest.mark, marker_name))
            if device_map in _SNAPDRAGON_DEVICES:
                item.add_marker(pytest.mark.snapdragon)


def pytest_runtest_setup(item):
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


# Model-manager pull failures raise, not skip — a broken hub is a real regression.
@pytest.fixture(scope='session')
def llama_cpp_llm_paths(geniex_session):
    return _mm.ensure_cached(LLAMA_CPP_LLM_MODEL, precision=LLAMA_CPP_LLM_PRECISION, hub='hf')


@pytest.fixture(scope='session')
def llama_cpp_mtp_paths(geniex_session):
    target = _mm.ensure_cached(LLAMA_CPP_MTP_TARGET_MODEL, precision=LLAMA_CPP_MTP_TARGET_PRECISION, hub='hf')
    draft = _mm.ensure_cached(LLAMA_CPP_MTP_DRAFT_MODEL, precision=LLAMA_CPP_MTP_DRAFT_PRECISION, hub='hf')
    return {'target': target, 'draft': draft}


@pytest.fixture(scope='session')
def llama_cpp_vlm_paths(geniex_session):
    return _mm.ensure_cached(LLAMA_CPP_VLM_MODEL, hub='hf')


@pytest.fixture(scope='session')
def qairt_llm_paths(geniex_session):
    return _mm.ensure_cached(QAIRT_LLM_MODEL)


@pytest.fixture(scope='session')
def qairt_vlm_paths(geniex_session):
    return _mm.ensure_cached(QAIRT_VLM_MODEL)


@pytest.fixture(scope='session')
def test_image() -> str:
    if not TEST_IMAGE_PATH.is_file():
        pytest.skip(f'test image missing: {TEST_IMAGE_PATH}')
    return str(TEST_IMAGE_PATH)


@pytest.fixture(scope='session')
def quality_image() -> str:
    if not QUALITY_IMAGE_PATH.is_file():
        pytest.skip(f'quality image missing: {QUALITY_IMAGE_PATH}')
    return str(QUALITY_IMAGE_PATH)

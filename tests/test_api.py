# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""SDK metadata + resolve APIs — no model, runs on any host."""

from __future__ import annotations

import os

import geniex
import pytest

_CUDA_BUILD = os.environ.get('GENIEX_TEST_CUDA_BUILD') == '1'
_requires_cuda_build = pytest.mark.skipif(not _CUDA_BUILD, reason='set GENIEX_TEST_CUDA_BUILD=1 for CUDA SDK builds')


def test_version_nonempty(geniex_session):
    v = geniex.version()
    assert isinstance(v, str) and v


def test_llama_cpp_plugin_version_nonempty(geniex_session):
    v = geniex.get_plugin_version('llama_cpp')
    assert isinstance(v, str) and v


def test_qairt_plugin_version_nonempty(geniex_session):
    if 'qairt' not in geniex.get_runtime_list():
        pytest.skip('QAIRT is not included in this SDK build')
    # Plugin reports its own version; available on hosts without an NPU
    # because the value comes from the shipped library, not the device.
    v = geniex.get_plugin_version('qairt')
    assert isinstance(v, str) and v


def test_runtime_list_is_non_empty_string_list(geniex_session):
    runtimes = geniex.get_runtime_list()
    assert isinstance(runtimes, list) and runtimes
    for r in runtimes:
        assert isinstance(r, str) and r


def test_runtime_list_contains_llama_cpp(geniex_session):
    assert 'llama_cpp' in geniex.get_runtime_list()


def test_compute_unit_list_shape_for_each_runtime(geniex_session):
    for runtime in geniex.get_runtime_list():
        compute_units = geniex.get_compute_unit_list(runtime)
        assert isinstance(compute_units, list)
        for entry in compute_units:
            assert isinstance(entry, tuple) and len(entry) == 2
            compute_unit, label = entry
            assert isinstance(compute_unit, str) and compute_unit
            assert isinstance(label, str)


def test_init_deinit_is_idempotent_within_session(geniex_session):
    geniex.init()
    geniex.init()


def test_set_log_level_accepts_known_levels(geniex_session):
    for level in ('trace', 'debug', 'info', 'warn', 'error', 'none'):
        geniex.set_log_level(level)


def test_public_surface_exports():
    expected = {
        'AutoModelForCausalLM',
        'AutoModelForVision2Seq',
        'GenieXError',
        'GenieXLLM',
        'GenieXVLM',
        'GenerateOutput',
        'ProfileData',
        'TextIteratorStreamer',
        'init',
        'deinit',
        'set_log_level',
        'set_qairt_runtime_path',
        'get_qairt_runtime_path',
        'version',
        'get_plugin_version',
        'get_runtime_list',
        'get_compute_unit_list',
        'resolve_device_map',
        'model_manager',
    }
    assert expected.issubset(set(geniex.__all__))
    for name in expected:
        assert hasattr(geniex, name), f'{name} missing from geniex module'


# QAIRT runtime override — set before init, process-global, and locked once
# initialized; see notes/run.md § Using a custom QNN library.


def test_qairt_runtime_path_is_locked_after_init(geniex_session):
    with pytest.raises(geniex.GenieXError):
        geniex.set_qairt_runtime_path('/nonexistent/qairt/2.XX.0')
    assert geniex.get_qairt_runtime_path() == ''


# resolve_device_map — source of truth lives in sdk/src/device.cpp. Any change
# to the alias table there must update these tests in the same PR.


def test_resolve_auto_returns_known_runtime(geniex_session):
    runtime, device_id, ngl = geniex.resolve_device_map('auto')
    assert runtime in geniex.get_runtime_list()
    assert device_id is None or isinstance(device_id, str)
    assert ngl is None or isinstance(ngl, int)


def test_resolve_cpu_alias_zeroes_gpu_layers(geniex_session):
    runtime, _, ngl = geniex.resolve_device_map('cpu')
    assert runtime == 'llama_cpp'
    assert ngl == 0


def test_resolve_hybrid_alias_offloads_all_layers(geniex_session):
    # No explicit ngl passed, so the resolver returns -1 (all layers), which
    # surfaces as None (no override).
    runtime, _, ngl = geniex.resolve_device_map('hybrid')
    assert runtime == 'llama_cpp'
    assert ngl is None


def test_resolve_llama_cpp_auto_uses_build_default(geniex_session):
    runtime, device_id, ngl = geniex.resolve_device_map('llama_cpp')
    assert runtime == 'llama_cpp'
    assert device_id == ('CUDA0' if _CUDA_BUILD else 'HTP0')
    assert ngl is None


def test_resolve_llama_cpp_npu_alias_pins_htp0(geniex_session):
    runtime, device_id, ngl = geniex.resolve_device_map('llama_cpp:npu')
    assert runtime == 'llama_cpp'
    assert device_id == 'HTP0'
    assert ngl is None


def test_resolve_qairt_npu_alias_resolves_to_qairt(geniex_session):
    runtime, device_id, _ = geniex.resolve_device_map('qairt:npu')
    assert runtime == 'qairt'
    assert isinstance(device_id, str) and device_id


@_requires_cuda_build
def test_resolve_cuda_gpu_alias(geniex_session):
    assert geniex.resolve_device_map('gpu') == ('llama_cpp', 'CUDA0', None)


@_requires_cuda_build
@pytest.mark.parametrize('mode', ['CUDA0', 'CUDA1', 'CUDA0,CUDA1', ' CUDA0 , CUDA1 '])
def test_native_resolve_explicit_cuda_devices(geniex_session, mode):
    # Exercise the native parser; Python's runtime:device shortcut bypasses it.
    from geniex._ffi._api import resolve_device

    expected = ','.join(part.strip() for part in mode.split(','))
    assert resolve_device('llama_cpp', None, mode, -1) == (expected, -1, None)


@_requires_cuda_build
@pytest.mark.parametrize('mode', ['CUDA', 'CUDA-1', 'CUDAx', 'CUDA0,', 'CUDA0,,CUDA1'])
def test_native_resolve_rejects_invalid_cuda_devices(geniex_session, mode):
    from geniex._ffi._api import resolve_device

    with pytest.raises(geniex.GenieXError):
        resolve_device('llama_cpp', None, mode, -1)


@_requires_cuda_build
def test_cuda_build_preserves_qairt_default(geniex_session):
    from geniex._ffi._api import resolve_device

    # Alias resolution itself does not require QAIRT to be loaded.
    assert resolve_device('qairt', None, 'auto', -1) == ('NPU', 0, None)

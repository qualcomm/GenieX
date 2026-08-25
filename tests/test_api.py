# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""SDK metadata + resolve APIs — no model, runs on any host."""

from __future__ import annotations

import geniex


def test_version_nonempty(geniex_session):
    v = geniex.version()
    assert isinstance(v, str) and v


def test_llama_cpp_plugin_version_nonempty(geniex_session):
    v = geniex.get_plugin_version('llama_cpp')
    assert isinstance(v, str) and v


def test_qairt_plugin_version_nonempty(geniex_session):
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


def test_resolve_llama_cpp_auto_defaults_to_npu(geniex_session):
    runtime, device_id, ngl = geniex.resolve_device_map('llama_cpp')
    assert runtime == 'llama_cpp'
    assert device_id == 'HTP0'
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

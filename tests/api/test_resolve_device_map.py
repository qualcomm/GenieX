# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""``geniex.resolve_device_map`` aliases — mirrors ``geniex_resolve_device``.

Source of truth lives in ``sdk/src/device.cpp``. Any change to the alias
table there must update these tests in the same PR.
"""

from __future__ import annotations

import geniex


def test_auto_resolves_to_a_known_runtime(geniex_session):
    runtime, device_id, ngl = geniex.resolve_device_map('auto')
    assert runtime in geniex.get_runtime_list()
    assert device_id is None or isinstance(device_id, str)
    assert ngl is None or isinstance(ngl, int)


def test_cpu_alias_zeroes_gpu_layers(geniex_session):
    runtime, _, ngl = geniex.resolve_device_map('cpu')
    assert runtime == 'llama_cpp'
    assert ngl == 0


def test_hybrid_alias_pins_gpu_layers_to_999(geniex_session):
    runtime, _, ngl = geniex.resolve_device_map('hybrid')
    assert runtime == 'llama_cpp'
    assert ngl == 999


def test_llama_cpp_npu_alias_pins_htp0(geniex_session):
    runtime, device_id, _ = geniex.resolve_device_map('llama_cpp:npu')
    assert runtime == 'llama_cpp'
    assert device_id == 'HTP0'


def test_qairt_npu_alias_resolves_to_qairt(geniex_session):
    runtime, device_id, _ = geniex.resolve_device_map('qairt:npu')
    assert runtime == 'qairt'
    assert isinstance(device_id, str) and device_id

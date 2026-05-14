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

"""Public metadata APIs: no model download required."""

from __future__ import annotations

import geniex
from geniex import ProfileData


def test_version_nonempty(geniex_session):
    v = geniex.version()
    assert isinstance(v, str) and v


def test_bundled_npu_runtime_version_nonempty(geniex_session):
    # Actually returns the bundled QAIRT runtime version; named so the
    # 'not qairt' pytest -k filter for device tests doesn't drop it.
    v = geniex.qairt_version()
    assert isinstance(v, str) and v


def test_plugin_list_has_llama_cpp(geniex_session):
    plugins = geniex.get_plugin_list()
    assert isinstance(plugins, list) and plugins
    assert 'llama_cpp' in plugins


def test_device_list_shape(geniex_session):
    for plugin in geniex.get_plugin_list():
        devices = geniex.get_device_list(plugin)
        assert isinstance(devices, list)
        for entry in devices:
            assert isinstance(entry, tuple) and len(entry) == 2
            assert isinstance(entry[0], str) and isinstance(entry[1], str)


def test_resolve_device_map_auto(geniex_session):
    plugin_id, device_id, ngl = geniex.resolve_device_map('auto')
    assert plugin_id in geniex.get_plugin_list()
    assert device_id is None or isinstance(device_id, str)
    assert ngl is None or isinstance(ngl, int)


def test_resolve_device_map_hybrid_forces_999(geniex_session):
    plugin_id, _, ngl = geniex.resolve_device_map('hybrid')
    assert plugin_id == 'llama_cpp'
    assert ngl == 999


def test_resolve_device_map_cpu_forces_0(geniex_session):
    plugin_id, _, ngl = geniex.resolve_device_map('cpu')
    assert plugin_id == 'llama_cpp'
    assert ngl == 0


def test_exports_public_surface():
    for name in [
        'AutoModelForCausalLM',
        'AutoModelForVision2Seq',
        'GeniexError',
        'GeniexLLM',
        'GeniexVLM',
        'GenerateOutput',
        'ProfileData',
        'TextIteratorStreamer',
        'init',
        'deinit',
        'set_log_level',
        'version',
        'qairt_version',
        'get_plugin_list',
        'get_device_list',
        'resolve_device_map',
        'model_manager',
    ]:
        assert name in geniex.__all__, f'{name} missing from geniex.__all__'
        assert hasattr(geniex, name), f'{name} missing from geniex'


def test_profile_data_repr_has_units():
    p = ProfileData(
        ttft=850,
        prompt_time=120_500,
        decode_time=1_850_000,
        prompt_tokens=128,
        generated_tokens=256,
        prefill_speed=533.3,
        decode_speed=138.4,
        stop_reason='eos',
    )
    s = repr(p)
    assert 'ttft=850 µs' in s
    assert 'prompt_time=120.5 ms' in s
    assert 'decode_time=1.85 s' in s
    assert 'prompt_tokens=128 tok' in s
    assert 'generated_tokens=256 tok' in s
    assert 'prefill_speed=533.3 tok/s' in s
    assert 'decode_speed=138.4 tok/s' in s
    assert 'stop_reason=eos' in s


def test_profile_data_repr_zero_defaults():
    s = repr(ProfileData())
    assert 'ttft=0 µs' in s
    assert 'prompt_time=0 µs' in s
    assert 'decode_time=0 µs' in s
    assert 'stop_reason=None' in s
